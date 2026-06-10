"""
token_savings.py — Honest benchmark: LLM tool-loop vs deterministic procedure.

Compares TWO paths for solving the SAME task:
  Path A = pure LLM tool-loop (Session.code without matching procedure)
  Path B = deterministic procedure (Session.code with procedure above threshold)

Measures: (1) LLM API calls, (2) tokens — from real API usage if available,
otherwise honest estimate (len/4) marked ESTIMATE.

Test task: "check if python3 is installed" — a task solvable by a deterministic
procedure (execute bash + report) vs an LLM that would do the same through
several tool-iterations.

Run:  python3 benchmarks/token_savings.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token count: chars // 4 for English text (standard heuristic)."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# MockProvider — deterministic LLM simulator for offline benchmarking
# ---------------------------------------------------------------------------

class MockProvider:
    """Simulates an LLM that uses tools.  Deterministic: always picks bash,
    runs it, then synthesises a one-line response from the output.

    Tracks internal API calls via `_api_calls` so the instrumented wrapper
    can report honest per-iteration counts.
    """

    def __init__(self, name: str = "mock") -> None:
        self.name = name
        self._api_calls: list[dict] = []   # {prompt_len, response_len, prompt_text, response_text}

    @property
    def api_call_count(self) -> int:
        return len(self._api_calls)

    async def chat(self, system: str, history: list, user_text: str) -> str:
        prompt = system + "\n" + user_text
        resp = f"[mock chat] understood: {user_text[:100]}"
        self._api_calls.append({
            "prompt_len": len(prompt), "response_len": len(resp),
            "prompt_text": prompt, "response_text": resp,
        })
        return resp

    async def agent_turn(
        self, system: str, history: list, user_text: str,
        tools_spec: list, tools: Callable, max_iters: int = 12,
        on_tool: Callable | None = None,
    ) -> str:
        """Simulate a 2-iteration LLM tool-loop:
        iteration 1 — LLM decides to call bash with a sensible command;
        iteration 2 — LLM reads tool output and returns a final answer.
        """
        # --- Iteration 1: LLM "decides" to call bash ---
        cmd = "python3 --version 2>&1"
        prompt1 = self._build_prompt(system, tools_spec, user_text)
        resp1 = f'[mock] tool_call: bash("{cmd}")'
        self._api_calls.append({
            "prompt_len": len(prompt1), "response_len": len(resp1),
            "prompt_text": prompt1, "response_text": resp1,
        })

        if on_tool:
            on_tool("bash", {"command": cmd})
        tool_output = await tools("bash", {"command": cmd})

        # --- Iteration 2: LLM synthesises final response ---
        tool_call_json = json.dumps({"tool": "bash", "command": cmd})
        prompt2 = prompt1 + f"\n[assistant tool_call]\n{tool_call_json}\n"
        prompt2 += f"[tool_result]\n{tool_output}\n"
        installed = "Python" in tool_output and len(tool_output.strip()) > 0
        if installed:
            resp2 = f"Python 3 is installed. Version info: {tool_output.strip()}"
        else:
            resp2 = f"Python 3 does not appear to be installed. Command output: {tool_output.strip()}"
        self._api_calls.append({
            "prompt_len": len(prompt2), "response_len": len(resp2),
            "prompt_text": prompt2, "response_text": resp2,
        })

        return resp2

    @staticmethod
    def _build_prompt(system: str, tools_spec: list, user_text: str) -> str:
        tools_str = json.dumps([t["name"] for t in tools_spec])
        return f"{system}\nAvailable tools: {tools_str}\nUser: {user_text}"


# ---------------------------------------------------------------------------
# UsageCaptureDeepSeekProvider — real API + usage extraction
# ---------------------------------------------------------------------------

class UsageCaptureDeepSeekProvider:
    """Wraps the real OpenAICompatProvider (DeepSeek), overriding _post to
    capture `usage` from every raw API response."""

    def __init__(self, api_key: str, model: str = "deepseek-chat") -> None:
        from agent.providers import OpenAICompatProvider, DEEPSEEK_BASE_URL
        self._inner = OpenAICompatProvider(
            api_key=api_key, model=model,
            base_url=DEEPSEEK_BASE_URL, name="deepseek",
        )
        self.name = "deepseek"
        self._api_calls: list[dict] = []   # {prompt_len, response_len, usage}
        self._usage_records: list[dict] = []

    @property
    def api_call_count(self) -> int:
        return len(self._api_calls)

    async def chat(self, system: str, history: list, user_text: str) -> str:
        prompt_text = system + " " + " ".join(
            m.get("content", "") for m in history
        ) + " " + user_text
        orig_post = self._inner._post

        captured_usage: dict = {}
        async def _post_hook(payload):
            result = await orig_post(payload)
            if "usage" in result:
                captured_usage["usage"] = result["usage"]
            return result

        self._inner._post = _post_hook  # type: ignore[method-assign]
        try:
            result = await self._inner.chat(system, history, user_text)
        finally:
            self._inner._post = orig_post  # type: ignore[method-assign]

        usage = captured_usage.get("usage", {})
        self._api_calls.append({
            "prompt_len": len(prompt_text), "response_len": len(result),
            "prompt_text": prompt_text, "response_text": result,
            "usage": usage,
        })
        if usage:
            self._usage_records.append(usage)
        return result

    async def agent_turn(
        self, system: str, history: list, user_text: str,
        tools_spec: list, tools: Callable, max_iters: int = 12,
        on_tool: Callable | None = None,
    ) -> str:
        orig_post = self._inner._post
        captured: list[dict] = []

        async def _post_hook(payload):
            result = await orig_post(payload)
            if "usage" in result:
                captured.append(dict(result["usage"]))
            return result

        self._inner._post = _post_hook  # type: ignore[method-assign]
        try:
            result = await self._inner.agent_turn(
                system, history, user_text, tools_spec, tools,
                max_iters, on_tool,
            )
        finally:
            self._inner._post = orig_post  # type: ignore[method-assign]

        for u in captured:
            self._usage_records.append(u)
            self._api_calls.append({
                "prompt_len": 0, "response_len": 0,  # unknown at this level
                "usage": u,
            })
        return result


# ---------------------------------------------------------------------------
# InstrumentedProvider — uniform wrapper, counts calls & estimates tokens
# ---------------------------------------------------------------------------

class InstrumentedProvider:
    """Wraps any provider (mock or real), counts calls and estimates tokens.

    If the inner provider stores per-call data in `_api_calls` (list of dicts
    with 'prompt_len'/'response_len' keys), those are used for accurate
    estimation.  Otherwise falls back to outer-level estimates.
    """

    def __init__(self, inner: Any, name: str = "instrumented") -> None:
        self.inner = inner
        self.name = name
        self._chat_calls = 0
        self._agent_calls = 0

    @property
    def api_call_count(self) -> int:
        """Total LLM API calls (from inner if tracked, else outer)."""
        if hasattr(self.inner, "api_call_count"):
            return self.inner.api_call_count
        return self._chat_calls + self._agent_calls

    @property
    def prompt_tokens_est(self) -> int:
        if hasattr(self.inner, "_api_calls"):
            return sum(
                estimate_tokens(c.get("prompt_text", "")) or c.get("prompt_len", 0) // 4
                for c in self.inner._api_calls
            )
        return 0

    @property
    def response_tokens_est(self) -> int:
        if hasattr(self.inner, "_api_calls"):
            return sum(
                estimate_tokens(c.get("response_text", "")) or c.get("response_len", 0) // 4
                for c in self.inner._api_calls
            )
        return 0

    @property
    def total_tokens_est(self) -> int:
        return self.prompt_tokens_est + self.response_tokens_est

    @property
    def has_real_usage(self) -> bool:
        return (
            hasattr(self.inner, "_usage_records")
            and len(self.inner._usage_records) > 0
        )

    @property
    def real_prompt_tokens(self) -> int:
        if not self.has_real_usage:
            return 0
        return sum(
            u.get("prompt_tokens", u.get("input_tokens", 0))
            for u in self.inner._usage_records
        )

    @property
    def real_response_tokens(self) -> int:
        if not self.has_real_usage:
            return 0
        return sum(
            u.get("completion_tokens", u.get("output_tokens", 0))
            for u in self.inner._usage_records
        )

    @property
    def real_total_tokens(self) -> int:
        return self.real_prompt_tokens + self.real_response_tokens

    async def chat(self, system: str, history: list, user_text: str) -> str:
        self._chat_calls += 1
        return await self.inner.chat(system, history, user_text)

    async def agent_turn(
        self, system: str, history: list, user_text: str,
        tools_spec: list, tools: Callable, max_iters: int = 12,
        on_tool: Callable | None = None,
    ) -> str:
        self._agent_calls += 1
        return await self.inner.agent_turn(
            system, history, user_text, tools_spec, tools,
            max_iters, on_tool,
        )


# ---------------------------------------------------------------------------
# Procedure body for the test task
# ---------------------------------------------------------------------------

def build_test_procedure() -> dict:
    """Return a procedure that checks if python3 is installed.

    Fully deterministic — no generative (LLM) steps:
      step 1: execute bash `python3 --version`
      step 2: validate via bash `command -v python3` (system binary check)
      step 3: report the result.

    Note: the built-in `check_installed` validator checks Python *module*
    importability (via importlib.util.find_spec), not system binary presence.
    For checking a system command we use a direct bash validation step.
    """
    return {
        "trigger": "check if python3 is installed",
        "body": {
            "registers": ["bash_output", "found"],
            "entry": "s1",
            "steps": [
                {
                    "id": "s1",
                    "op": "execute",
                    "label": "run_version_check",
                    "args": {
                        "tool": "bash",
                        "command": "python3 --version 2>&1",
                    },
                    "out": "bash_output",
                    "next": "s2",
                },
                {
                    "id": "s2",
                    "op": "execute",
                    "label": "check_binary_exists",
                    "args": {
                        "tool": "bash",
                        "command": "command -v python3 2>&1",
                    },
                    "out": "found",
                    "next": "s3",
                },
                {
                    "id": "s3",
                    "op": "report",
                    "label": "format_result",
                    "args": {},
                    "next": "END",
                },
            ],
        },
    }


# ---------------------------------------------------------------------------
# Tool runner (bash only, for the benchmark)
# ---------------------------------------------------------------------------

async def _bash_runner(cmd: str) -> str:
    """Execute a shell command and return stdout+stderr."""
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    result = (out.decode("utf-8", "replace") + err.decode("utf-8", "replace")).strip()
    return result[:8000] if result else "(no output)"


async def _tool_dispatcher(name: str, args: dict) -> str:
    if name == "bash":
        return await _bash_runner(args.get("command", ""))
    return f"(unknown tool: {name})"


# ---------------------------------------------------------------------------
# Path A: LLM tool-loop
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a coder agent. Use the bash tool to run shell commands. "
    "Do the task, do not just describe it. Be concise."
)

TOOLS_SPEC = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
]

async def run_path_a(instrumented: InstrumentedProvider, task: str) -> str:
    """Execute the task through the LLM tool-loop (agent_turn)."""
    return await instrumented.agent_turn(
        SYSTEM_PROMPT, [], task, TOOLS_SPEC, _tool_dispatcher,
    )


# ---------------------------------------------------------------------------
# Path B: deterministic procedure
# ---------------------------------------------------------------------------

async def run_path_b(instrumented: InstrumentedProvider, task: str) -> str:
    """Execute the task through the deterministic procedure interpreter.

    The procedure runner uses handlers (bash, llm, mem).  We provide a real
    bash handler and an LLM handler that counts any generative steps.
    """
    from vsa.procedure_runner import Interpreter
    from vsa.procedure_handlers import build_handlers

    procedure = build_test_procedure()

    # Track any generative LLM calls inside the procedure
    generative_calls: list[dict] = []

    async def _proc_llm(prompt: str, model: str | None = None,
                        system: str | None = None) -> str:
        """LLM handler for generative steps inside a procedure.
        Counts as an API call through the instrumented provider."""
        generative_calls.append({"prompt": prompt, "model": model})
        result = await instrumented.chat(system or "", [], prompt)
        return result

    async def _proc_bash(cmd: str) -> str:
        return await _bash_runner(cmd)

    # mem=None is safe — our procedure has no search steps
    handlers = build_handlers(
        llm=_proc_llm,
        bash=_proc_bash,
        mem=None,  # type: ignore[arg-type]
    )

    interp = Interpreter(handlers)

    result = await interp.arun(
        procedure,
        registers={"task": task, "context": task},
    )

    regs = result.get("registers", {})
    bash_output = regs.get("bash_output", "")
    found = regs.get("found", "")

    if found and "python3" in found.lower():
        return f"Python 3 is installed. Version info: {bash_output.strip()}"
    elif bash_output and "Python" in bash_output:
        return f"Python 3 is installed. Version info: {bash_output.strip()}"
    else:
        return f"Python 3 does not appear to be installed. Output: {bash_output.strip()}"


# ---------------------------------------------------------------------------
# Main: run both paths & print comparison table
# ---------------------------------------------------------------------------

def _fmt_tokens(n: int, is_estimate: bool) -> str:
    mark = " (est)" if is_estimate else ""
    return f"{n:,}{mark}"


async def main() -> None:
    task = "check if python3 is installed"

    # ---- Choose provider backend ------------------------------------------
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
    use_real = bool(deepseek_key)

    if use_real:
        print("🔑 DEEPSEEK_API_KEY found — using real DeepSeek API.\n")
        inner_a = UsageCaptureDeepSeekProvider(api_key=deepseek_key)
        # Path B may need an LLM for generative steps — use same real backend
        inner_b = UsageCaptureDeepSeekProvider(api_key=deepseek_key)
    else:
        print("ℹ️  No API key set — using MockProvider, token numbers are ESTIMATES.\n")
        inner_a = MockProvider(name="mock-path-a")
        inner_b = MockProvider(name="mock-path-b")

    provider_a = InstrumentedProvider(inner_a, name="Path-A")
    provider_b = InstrumentedProvider(inner_b, name="Path-B")

    # ---- Run both paths ---------------------------------------------------
    print(f"Task: {task!r}\n")
    print("─" * 60)

    print("\n▶ Path A: LLM tool-loop (no procedure match)")
    result_a = await run_path_a(provider_a, task)
    print(f"  Result: {result_a}")

    print("\n▶ Path B: Deterministic procedure (procedure match above threshold)")
    result_b = await run_path_b(provider_b, task)
    print(f"  Result: {result_b}")

    # ---- Collect metrics --------------------------------------------------
    calls_a = provider_a.api_call_count
    calls_b = provider_b.api_call_count

    # Per-path estimate/real decision: a path has real usage iff its
    # inner provider captured usage records from the API.
    a_real = provider_a.has_real_usage
    b_real = provider_b.has_real_usage

    # Tokens for each path — prefer real usage, fall back to estimate
    if a_real:
        tokens_a = provider_a.real_total_tokens
    else:
        tokens_a = provider_a.total_tokens_est

    if b_real:
        tokens_b = provider_b.real_total_tokens
    else:
        tokens_b = provider_b.total_tokens_est

    savings_pct = 0.0
    if tokens_a > 0:
        savings_pct = (1.0 - tokens_b / tokens_a) * 100.0

    # ---- Print table ------------------------------------------------------
    print("\n" + "=" * 72)
    print("  TOKEN SAVINGS BENCHMARK")
    print("=" * 72)
    print(f"  {'Path':<16} {'LLM calls':>10} {'Tokens':>16}  {'Savings %':>10}")
    print("  " + "-" * 58)

    # Token display: show "(est)" only when LLM calls were actually made
    # but usage wasn't captured.  0 calls → 0 tokens is exact by definition.
    show_est_a = not a_real and calls_a > 0
    show_est_b = not b_real and calls_b > 0
    tok_a_label = _fmt_tokens(tokens_a, show_est_a)
    tok_b_label = _fmt_tokens(tokens_b, show_est_b)
    print(f"  {'Path A (tool-loop)':<16} {calls_a:>10} {tok_a_label:>16}")
    print(f"  {'Path B (procedure)':<16} {calls_b:>10} {tok_b_label:>16}  {savings_pct:>9.1f}%")
    print("  " + "-" * 58)

    if a_real:
        mode_a = "Path A: REAL (DeepSeek usage)"
    elif calls_a == 0:
        mode_a = "Path A: 0 calls (exact)"
    else:
        mode_a = "Path A: ESTIMATE (chars ÷ 4)"
    if b_real:
        mode_b = "Path B: REAL (DeepSeek usage)"
    elif calls_b == 0:
        mode_b = "Path B: 0 calls (exact)"
    else:
        mode_b = "Path B: ESTIMATE (chars ÷ 4)"
    print(f"  {mode_a}")
    print(f"  {mode_b}")
    print("=" * 72)


    # ---- Honest breakdown -------------------------------------------------
    print("\n── Breakdown ──")
    print(f"  Path A — LLM tool-loop:")
    print(f"    • Orchestration decisions (which tool, what next):  per-iteration LLM calls")
    print(f"    • Final synthesis (reading tool output → user answer): last iteration")
    print(f"    • Total LLM API calls: {calls_a}")
    if a_real:
        print(f"    • Real tokens — prompt: {provider_a.real_prompt_tokens:,}, "
              f"completion: {provider_a.real_response_tokens:,}")
    else:
        print(f"    • Est. tokens — prompt: {provider_a.prompt_tokens_est:,}, "
              f"completion: {provider_a.response_tokens_est:,}")
    print(f"  Path B — Deterministic procedure:")
    print(f"    • Orchestration (step sequencing):  0 LLM calls (hardcoded DAG)")
    print(f"    • Generative steps (LLM inside procedure): 0 (fully deterministic)")
    print(f"    • Total LLM API calls: {calls_b}")

    if calls_b == 0:
        print(f"\n  → Deterministic procedure eliminates per-step LLM orchestration entirely.")
        print(f"    Orchestration tokens = 0.  All LLM cost is avoided for this task.")
    else:
        print(f"\n  → Procedure had {calls_b} generative LLM call(s) — these are counted.")
        print(f"    Orchestration tokens are still 0 (step DAG is hardcoded).")

    if not a_real and not b_real and (calls_a > 0 or calls_b > 0):
        print(f"\n  ⚠️  All token numbers are ESTIMATES (chars ÷ 4).")
        print(f"    Real measurement requires an API key and providers that expose usage.")
    elif not a_real and calls_a > 0:
        print(f"\n  ℹ️  Path A tokens are ESTIMATES — no real API usage captured.")
    elif not b_real and calls_b > 0:
        print(f"\n  ℹ️  Path B tokens are ESTIMATES — no real API usage captured.")


if __name__ == "__main__":
    asyncio.run(main())
