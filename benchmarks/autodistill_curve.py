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

LIGHT_TASK = ("List the files in this directory with list_dir, then read sample.py "
              "with read_file, then say in one line what the file defines. Use ONLY "
              "the list_dir and read_file tools, not bash.")

# Heavy task: a real package audit — many read-only steps, big context per loop
# iteration. This is where the loop's repeated-context cost really bites.
HEAVY_TASK = (
    "Audit the Python package in this directory. Use ONLY read-only bash "
    "(grep/wc/cat/find/ls — no writes). For each .py file report line count and "
    "the number of functions, count TODO/FIXME markers across the package, then "
    "give 2 short recommendations. Inspect the files before answering.")


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


async def measure(label, autodistill, key, workdir, task):
    prov = MeteredDeepSeek(key)
    rows = []
    for i in range(1, N + 1):
        # fresh Session each run (simulates separate invocations); shared memory dir
        s = Session(prov, workdir=workdir,
                    memory=AgentMemory(path=str(Path(workdir) / ".raidho" / "memory")),
                    autodistill=autodistill)
        prov.reset()
        t0 = time.perf_counter()
        await s.code(task)
        dt = time.perf_counter() - t0
        rows.append((i, prov.run_in, prov.run_out, cost(prov.run_in, prov.run_out), dt))
    print(f"\n  ── {label} ──")
    print(f"  {'run':>3} | {'in':>6} | {'out':>5} | {'$/run':>8} | {'sec':>5}")
    for i, tin, tout, c, dt in rows:
        print(f"  {i:>3} | {tin:>6} | {tout:>5} | ${c:>7.5f} | {dt:>4.1f}")
    total = sum(r[3] for r in rows)
    print(f"  total {N} runs: ${total:.5f}")
    return rows, total


def _seed_light(wd: Path):
    (wd / "sample.py").write_text("def greet(name):\n    return f'hi {name}'\n")


def _seed_heavy(wd: Path):
    # a small multi-file package with varied sizes + TODO/FIXME markers
    (wd / "core.py").write_text("# TODO: refactor\n" + "".join(
        f"def f{i}(x):\n    return x + {i}\n" for i in range(12)))
    (wd / "utils.py").write_text("import os\n# FIXME: handle errors\n" + "".join(
        f"def util{i}():\n    pass\n" for i in range(7)))
    (wd / "cli.py").write_text("".join(
        f"def cmd{i}(args):\n    '''doc'''\n    return {i}\n" for i in range(9)))
    (wd / "__init__.py").write_text("from .core import f0\n")


async def run_profile(name, task, seed, key, base):
    print(f"\n\n████ PROFILE: {name} ████")
    for d, mode in (("baseline", False), ("distill", True)):
        wd = base / name / d
        wd.mkdir(parents=True)
        seed(wd)
    _, bt = await measure("baseline (autodistill OFF)", False, key,
                          str(base / name / "baseline"), task)
    drows, dt = await measure("distill (autodistill ON)", True, key,
                              str(base / name / "distill"), task)
    steady = sum(r[3] for r in drows[1:]) / (N - 1)
    base_per = bt / N
    ratio = base_per / max(steady, 1e-9)
    print(f"\n  ── {name} summary ──")
    print(f"  baseline / run: ${base_per:.5f}   |   distill run1: ${drows[0][3]:.5f}"
          f"   |   distill repeat (2..{N}): ${steady:.5f}")
    print(f"  → ×{ratio:.1f} cheaper per repeat; cumulative {N} runs "
          f"${bt:.5f}→${dt:.5f} ({(1 - dt / bt) * 100:.0f}% saved)")
    return {"name": name, "base_per": base_per, "run1": drows[0][3],
            "steady": steady, "ratio": ratio, "base_total": bt, "dist_total": dt}


async def main():
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        sys.exit("set DEEPSEEK_API_KEY")
    import tempfile
    print(f"═══ Spend vs repetition (deepseek-chat, same task ×{N}, two profiles) ═══")
    with tempfile.TemporaryDirectory() as b:
        base = Path(b)
        light = await run_profile("light (2 reads)", LIGHT_TASK, _seed_light, key, base)
        heavy = await run_profile("heavy (package audit)", HEAVY_TASK, _seed_heavy, key, base)

    print("\n\n═══ ИТОГ (обе перспективы) ═══")
    print(f"  {'profile':<22} | {'base/run':>9} | {'repeat':>9} | {'×':>5} | {'5-run saved':>11}")
    for r in (light, heavy):
        print(f"  {r['name']:<22} | ${r['base_per']:>8.5f} | ${r['steady']:>8.5f} | "
              f"×{r['ratio']:>4.1f} | {(1 - r['dist_total'] / r['base_total']) * 100:>9.0f}%")
    print("  Чем тяжелее петля, тем больше АБСОЛЮТНАЯ экономия на повтор.")


if __name__ == "__main__":
    asyncio.run(main())
