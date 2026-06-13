"""Coder session: provider, tools, neutral history, system prompt and
(optionally) VSA memory. Two modes — chat (reasoning, no tools) and
code (agentic tool-loop).

Memory is wired in both directions: before each turn the relevant facts are
added to the system prompt (recall), and the `remember` tool lets the agent store
new facts (available only when memory is attached).

The code() method first checks for a matching deterministic procedure in VSA
memory; if found (score >= PROC_THRESHOLD) it runs the procedure via the
interpreter instead of the LLM tool-loop, saving tokens and latency.

Context-first mode (measured: evidence/2026-06-11_opus_vs_raidho — closed the
quality gap of the procedure path at x2.6 less cost than the pure loop): a
deterministic collector packs the file tree + task-relevant sources into the
FIRST call, so the model does not spend loop iterations on discovery —
re-paying the growing context each time. Tools stay available for actions
(writes, runs) and for files the budget omitted.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .context import collect_context
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
                 reason_provider: Provider | None = None,
                 context_first: bool = False, context_budget: int = 24_000,
                 history_budget: int = 120_000):
        self.provider = provider                            # execution (code, tool-loop)
        self.reason_provider = reason_provider or provider  # reasoning (chat); same by default
        self.tools = Tools(workdir)
        self.system = system
        self.memory = memory
        self.context_first = context_first    # pack workspace context into the first call
        self.context_budget = context_budget  # char budget for the collected block
        self.history_budget = history_budget  # char budget; oldest turns dropped beyond it
        self.history: list[dict] = []  # neutral: [{"role","content"}]

    def _save_memory(self) -> None:
        """Persist memory after a turn (no-op unless a path is configured)."""
        if self.memory:
            self.memory.save()

    def _trim_history(self) -> None:
        """Keep history within the char budget by dropping the OLDEST turn pair.
        Unbounded growth otherwise ends in a context-window error on long
        sessions; durable facts belong in memory (remember), not in history."""
        def used() -> int:
            return sum(len(str(m.get("content", ""))) for m in self.history)
        trimmed = 0
        while len(self.history) > 2 and used() > self.history_budget:
            del self.history[:2]
            trimmed += 1
        if trimmed:
            print(f"  ✂ history: dropped {trimmed} oldest turn(s) (budget "
                  f"{self.history_budget} chars)")

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
        self._trim_history()
        return reply

    async def code(self, task: str, context_first: bool | None = None) -> str:
        """Agentic mode: the tool-loop performs the task (recall + remember active).

        Before falling into the LLM tool-loop, checks VSA memory for a matching
        deterministic procedure.  If one is found with score >= PROC_THRESHOLD it
        is executed by the procedure interpreter — saving tokens and latency.
        On any procedure failure the method falls back to the normal LLM path.

        context_first (per-call override of the session setting): prepend a
        deterministically collected workspace context to the first call, so
        discovery does not burn loop iterations."""
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
                        self._trim_history()
                        self._save_memory()
                        return result
                    except (ProcedureError, Exception):
                        # procedure crashed — outcome already recorded by
                        # run_procedure; fall through to normal LLM tool-loop
                        pass

        # ── context-first: hand the workspace to the FIRST call ──
        prompt = task
        use_ctx = self.context_first if context_first is None else context_first
        if use_ctx:
            block, stats = collect_context(self.tools.workdir, task,
                                           char_budget=self.context_budget)
            prompt = task + block
            print(f"  📦 context-first: {stats['files_included']} files, "
                  f"{stats['chars']} chars"
                  + (f" ({stats['files_omitted']} omitted)" if stats['files_omitted'] else ""))

        # ── normal LLM tool-loop ──
        reply = await self.provider.agent_turn(
            self._system_for(task), self.history, prompt,
            self._tools_spec(), self._run_tool, on_tool=_print_tool)
        # history keeps the bare task — the context block is per-call evidence,
        # not conversation; re-storing it would bloat every later turn
        self.history += [{"role": "user", "content": task},
                         {"role": "assistant", "content": reply}]
        self._trim_history()
        self._save_memory()
        return reply

    async def council(self, question: str, rounds: int = 2,
                      secretary: Provider | None = None, remember: bool = True) -> dict:
        """Two-provider debate → consensus. Seat A = reason_provider, seat B = the
        execution provider — set them differently (e.g. via reason_provider) for a
        Claude-vs-DeepSeek debate. Returns {'transcript', 'verdict', 'remembered'}.

        secretary: who distills the verdict. Default is seat A — note this is a
        participant, i.e. a potential bias; pass a third provider for a truly
        neutral verdict on contested questions.

        remember: if a memory is attached, distill the verdict into structural
        triples (one cheap extraction call on the EXECUTION provider) and store
        them in VSA memory — so the consensus surfaces in later turns via recall.
        Facts cost ~0 downstream (recalled only when relevant), unlike dumping
        the whole verdict into history. 'remembered' = list of stored triples."""
        c = Council(self.reason_provider, self.provider)
        res = await c.consensus(question, rounds=rounds, on_turn=_print_turn,
                                secretary=secretary)
        res["remembered"] = []
        if remember and self.memory:
            res["remembered"] = await self._remember_verdict(question, res["verdict"])
            if res["remembered"]:
                self._save_memory()
        return res

    async def _remember_verdict(self, question: str, verdict: str) -> list:
        """One cheap extraction pass: verdict → structural triples → VSA memory.
        Runs on the execution provider (the cheap seat). Best-effort: any parse
        or provider hiccup yields [] and never breaks the council result."""
        prompt = (
            "From the consensus below, extract the DURABLE decisions as structural "
            "triples (subject, relation, object) — concrete choices worth recalling "
            "later (a chosen library, an approach, a constraint). Skip vague talk.\n"
            "Return ONLY a JSON array, e.g. "
            '[{"subject":"auth","relation":"uses","object":"PyJWT"}]. '
            "Empty array if nothing durable.\n\n"
            f"Topic: {question}\n\nConsensus:\n{verdict}")
        try:
            raw = await self.provider.chat(
                "You extract structured facts. Output JSON only, no prose.", [], prompt)
        except Exception:
            return []
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            return []
        try:
            items = json.loads(m.group(0))
        except (ValueError, TypeError):
            return []
        stored = []
        for it in items if isinstance(items, list) else []:
            if not isinstance(it, dict):
                continue
            s, r, o = (str(it.get(k, "")).strip()
                       for k in ("subject", "relation", "object"))
            if s and r and o:
                self.memory.remember(s, r, o)
                stored.append((s, r, o))
        if stored:
            print(f"  🧠 council → memory: {len(stored)} fact(s) stored")
        return stored
