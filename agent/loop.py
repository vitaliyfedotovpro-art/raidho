"""Coder session: provider, tools, neutral history, system prompt and
(optionally) VSA memory. Two modes — chat (reasoning, no tools) and
code (agentic tool-loop).

Memory is wired in both directions: before each turn the relevant facts are
added to the system prompt (recall), and the `remember` tool lets the agent store
new facts (available only when memory is attached).

The code() method first checks for a matching deterministic procedure in VSA
memory; if found (score >= PROC_THRESHOLD) it runs the procedure via the
interpreter instead of the LLM tool-loop, saving tokens and latency.
"""
from __future__ import annotations

from pathlib import Path

from .council import Council
from .memory import REMEMBER_SPEC, AgentMemory
from .providers import Provider
from .tools import TOOLS_SPEC, Tools
from vsa.procedure_runner import Interpreter, ProcedureError
from vsa.procedure_handlers import build_handlers

DEFAULT_SYSTEM = (
    "You are a coder agent. You help read, write and edit code in the working directory.\n"
    "In coding mode use the tools (bash/read_file/write_file/list_dir) — "
    "do the task, don't just describe it. Don't invent file contents — read them. "
    "Look at the current code before editing. Explain commands briefly.\n"
    "If memory is attached — store durable facts (decisions, names, deadlines) with "
    "the remember tool; the 'Relevant memory' block in the prompt is recall."
)

# Minimum match score to trigger deterministic procedure execution (0.0–1.0).
PROC_THRESHOLD = 0.6


def _print_tool(name: str, args: dict) -> None:
    preview = str(args.get("command") or args.get("path") or args.get("subject") or "")[:70]
    print(f"  🔧 {name}({preview})")


def _print_turn(who: str, text: str) -> None:
    print(f"\n{who}> {text}\n")


class Session:
    def __init__(self, provider: Provider, workdir: str | Path = ".",
                 system: str = DEFAULT_SYSTEM, memory: AgentMemory | None = None,
                 reason_provider: Provider | None = None):
        self.provider = provider                            # execution (code, tool-loop)
        self.reason_provider = reason_provider or provider  # reasoning (chat); same by default
        self.tools = Tools(workdir)
        self.system = system
        self.memory = memory
        self.history: list[dict] = []  # neutral: [{"role","content"}]

    def _system_for(self, text: str) -> str:
        """Base prompt + recall of memory relevant to the current query."""
        if not self.memory:
            return self.system
        block = self.memory.recall(text)
        return f"{self.system}\n\n{block}" if block else self.system

    def _tools_spec(self) -> list:
        return TOOLS_SPEC + ([REMEMBER_SPEC] if self.memory else [])

    async def _run_tool(self, name: str, args: dict) -> str:
        if self.memory and name == "remember":
            return self.memory.remember(
                args.get("subject", ""), args.get("relation", ""), args.get("object", ""))
        return await self.tools.run(name, args)

    async def run_procedure(self, proc_id: str, executor) -> str:
        """Execute a procedure with automatic outcome tracking.
        On success → memory.record_outcome(proc_id, True).
        On crash (exception) → memory.record_outcome(proc_id, False) + re-raise.
        executor is an async callable that performs the procedure steps."""
        try:
            result = await executor()
            if self.memory:
                self.memory.mem.record_outcome(proc_id, True)
            return result
        except Exception:
            if self.memory:
                self.memory.mem.record_outcome(proc_id, False)
            raise

    async def chat(self, text: str) -> str:
        """Text mode: discussion/reasoning, no tools (recall active).
        Uses reason_provider — you can "think" with a smart model and "execute" with a cheap one."""
        reply = await self.reason_provider.chat(self._system_for(text), self.history, text)
        self.history += [{"role": "user", "content": text},
                         {"role": "assistant", "content": reply}]
        return reply

    async def code(self, task: str) -> str:
        """Agentic mode: the tool-loop performs the task (recall + remember active).

        Before falling into the LLM tool-loop, checks VSA memory for a matching
        deterministic procedure.  If one is found with score >= PROC_THRESHOLD it
        is executed by the procedure interpreter — saving tokens and latency.
        On any procedure failure the method falls back to the normal LLM path."""
        # ── deterministic procedure path ──
        if self.memory:
            hits = self.memory.match_procedure(task)
            if hits and hits[0]["score"] >= PROC_THRESHOLD:
                pid = hits[0]["proc_id"]
                proc = self.memory.mem.get_procedure(pid)
                if proc is not None:
                    # thin async wrappers for the interpreter
                    async def _llm(prompt, model=None, system=None):
                        return await self.provider.chat(system or "", [], prompt)

                    async def _bash(cmd):
                        return await self.tools.run("bash", {"command": cmd})

                    interp = Interpreter(build_handlers(
                        llm=_llm, bash=_bash, mem=self.memory.mem))

                    async def _exec():
                        res = await interp.arun(
                            proc, registers={"task": task, "context": task})
                        return str(res.get("registers", res))

                    print(f"  ⚡ procedure {pid} (deterministic)")

                    try:
                        result = await self.run_procedure(pid, _exec)
                        self.history += [{"role": "user", "content": task},
                                         {"role": "assistant", "content": result}]
                        return result
                    except (ProcedureError, Exception):
                        # procedure crashed — outcome already recorded by
                        # run_procedure; fall through to normal LLM tool-loop
                        pass

        # ── normal LLM tool-loop ──
        reply = await self.provider.agent_turn(
            self._system_for(task), self.history, task,
            self._tools_spec(), self._run_tool, on_tool=_print_tool)
        self.history += [{"role": "user", "content": task},
                         {"role": "assistant", "content": reply}]
        return reply

    async def council(self, question: str, rounds: int = 2) -> dict:
        """Two-provider debate → consensus. Seat A = reason_provider, seat B = the
        execution provider — set them differently (e.g. via reason_provider) for a
        Claude-vs-DeepSeek debate. Returns {'transcript', 'verdict'}."""
        c = Council(self.reason_provider, self.provider)
        return await c.consensus(question, rounds=rounds, on_turn=_print_turn)
