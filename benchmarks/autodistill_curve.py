"""autodistill_curve.py — REAL-API measurement: how much spend falls with
repetition when auto-distillation is on.

Runs the SAME read-only task N times, twice:
  baseline  — autodistill OFF: every run is a full LLM tool-loop.
  distill   — autodistill ON: run 1 learns a procedure, runs 2..N replay it
              deterministically (deterministic reads + one synthesis call).

Measures real DeepSeek token usage per run (prompt+completion from the API
`usage` field) and dollar cost, then prints the per-run curve and cumulative
savings. Read-only task → safe to auto-distill.

Pricing (deepseek-chat, verified 2026-06-11): $0.14/M in (cache-miss), $0.28/M out.
Requires DEEPSEEK_API_KEY. Budget: ~10 cheap calls, a few cents.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.loop import Session                       # noqa: E402
from agent.memory import AgentMemory                 # noqa: E402
from agent.providers import OpenAICompatProvider     # noqa: E402

PRICE_IN, PRICE_OUT = 0.14, 0.28                     # $/1M tokens, deepseek-chat
N = 5
TASK = ("List the files in this directory with list_dir, then read sample.py with "
        "read_file, then say in one line what the file defines. Use ONLY the "
        "list_dir and read_file tools, not bash.")


class MeteredDeepSeek(OpenAICompatProvider):
    """DeepSeek provider that accumulates token usage per logical run."""
    def __init__(self, key):
        super().__init__(api_key=key, model="deepseek-chat")
        self.run_in = 0
        self.run_out = 0

    def reset(self):
        self.run_in = self.run_out = 0

    async def _post(self, payload):
        data = await super()._post(payload)
        u = data.get("usage") or {}
        self.run_in += u.get("prompt_tokens", 0)
        self.run_out += u.get("completion_tokens", 0)
        return data


def cost(tin, tout):
    return (tin * PRICE_IN + tout * PRICE_OUT) / 1e6


async def measure(label, autodistill, key, workdir):
    prov = MeteredDeepSeek(key)
    rows = []
    for i in range(1, N + 1):
        # fresh Session each run (simulates separate invocations); shared memory dir
        s = Session(prov, workdir=workdir,
                    memory=AgentMemory(path=str(Path(workdir) / ".raidho" / "memory")),
                    autodistill=autodistill)
        prov.reset()
        t0 = time.perf_counter()
        await s.code(TASK)
        dt = time.perf_counter() - t0
        deterministic = bool(s.memory.mem.procedures) and i > 1 and autodistill
        rows.append((i, prov.run_in, prov.run_out, cost(prov.run_in, prov.run_out), dt))
    print(f"\n  ── {label} ──")
    print(f"  {'run':>3} | {'in':>6} | {'out':>5} | {'$/run':>8} | {'sec':>5}")
    for i, tin, tout, c, dt in rows:
        print(f"  {i:>3} | {tin:>6} | {tout:>5} | ${c:>7.5f} | {dt:>4.1f}")
    total = sum(r[3] for r in rows)
    print(f"  total {N} runs: ${total:.5f}")
    return rows, total


async def main():
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        sys.exit("set DEEPSEEK_API_KEY")
    import tempfile
    print(f"═══ Spend vs repetition: same read-only task ×{N} (deepseek-chat) ═══")
    with tempfile.TemporaryDirectory() as base:
        for d in ("baseline", "distill"):
            wd = Path(base) / d
            wd.mkdir()
            (wd / "sample.py").write_text("def greet(name):\n    return f'hi {name}'\n")
        _, base_total = await measure("baseline (autodistill OFF)", False, key,
                                      str(Path(base) / "baseline"))
        dist_rows, dist_total = await measure("distill (autodistill ON)", True, key,
                                              str(Path(base) / "distill"))

    print("\n═══ ИТОГ ═══")
    # steady-state: cost of a repeat run once the procedure is learned (runs 2..N)
    steady = sum(r[3] for r in dist_rows[1:]) / (N - 1)
    base_per = base_total / N
    print(f"  baseline per run (always full loop): ${base_per:.5f}")
    print(f"  distill run 1 (learns):              ${dist_rows[0][3]:.5f}")
    print(f"  distill steady-state (runs 2..{N}):   ${steady:.5f}  "
          f"→ ×{base_per / max(steady, 1e-9):.1f} cheaper per repeat")
    print(f"  cumulative {N} runs: baseline ${base_total:.5f} vs distill ${dist_total:.5f} "
          f"→ {(1 - dist_total / base_total) * 100:.0f}% saved")
    print(f"  break-even after ~run 2; the more a task repeats, the closer to "
          f"×{base_per / max(steady, 1e-9):.1f}.")


if __name__ == "__main__":
    asyncio.run(main())
