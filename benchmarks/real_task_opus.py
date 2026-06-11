"""real_task_opus.py — REAL-API benchmark: Raidho procedure path vs pure LLM loop.

Same complex task solved twice with the SAME model (claude-opus-4-8, adaptive
thinking), measuring real tokens / dollars / wall time from API usage fields:

  Path A (Raidho)    — deterministic Interpreter procedure scans the codebase
                       locally (0 LLM calls), then ONE LLM call writes the
                       report from the aggregated JSON facts.
  Path B (pure LLM)  — manual agentic tool-loop with a bash tool; the model
                       explores the directory itself, file contents flow
                       through its context window.

Task: audit the agent/ package — per-file line counts, function counts,
TODO/FIXME markers, functions missing docstrings; summary report with top-3
recommendations.

Pricing (verified 2026-06-11): $5/M input, $25/M output;
cache write x1.25, cache read x0.1.

Budget guard: aborts if cumulative cost exceeds $3 or 15 loop iterations.
Requires ANTHROPIC_API_KEY (source .env).

Run:  python3 benchmarks/real_task_opus.py
"""
from __future__ import annotations

import ast
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vsa.procedure_runner import Interpreter  # noqa: E402

MODEL = "claude-opus-4-8"
PRICE_IN, PRICE_OUT = 5.0, 25.0          # $/M tokens
PRICE_CACHE_W, PRICE_CACHE_R = 6.25, 0.50
BUDGET_USD = 3.0
MAX_ITERS = 15
TARGET = "agent"                          # package to audit, relative to repo root

TASK = (
    f"Проведи аудит Python-пакета `{TARGET}/` (текущая директория — корень проекта). "
    "Для КАЖДОГО .py файла собери: число строк, число функций, число маркеров "
    "TODO/FIXME, список функций без docstring. Затем напиши сводный отчёт: "
    "таблица по файлам, общие итоги, и топ-3 конкретных рекомендации по улучшению "
    "качества кода. Отчёт — на русском, в markdown."
)


class Meter:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def add(self, usage) -> None:
        self.calls.append({
            "in": usage.input_tokens,
            "out": usage.output_tokens,
            "cw": getattr(usage, "cache_creation_input_tokens", 0) or 0,
            "cr": getattr(usage, "cache_read_input_tokens", 0) or 0,
        })

    def cost(self) -> float:
        return sum(c["in"] * PRICE_IN + c["out"] * PRICE_OUT
                   + c["cw"] * PRICE_CACHE_W + c["cr"] * PRICE_CACHE_R
                   for c in self.calls) / 1e6

    def totals(self) -> dict:
        return {k: sum(c[k] for c in self.calls) for k in ("in", "out", "cw", "cr")}


def guard(meter: Meter) -> None:
    if meter.cost() > BUDGET_USD:
        raise RuntimeError(f"бюджетный предохранитель: ${meter.cost():.2f} > ${BUDGET_USD}")


# ── Path A: deterministic scan (Raidho Interpreter) + one report call ────────

def scan_package(pkg_dir: Path) -> dict:
    """The mechanical part — what a procedure does locally for $0."""
    files = {}
    for p in sorted(pkg_dir.glob("*.py")):
        src = p.read_text(encoding="utf-8")
        lines = src.count("\n") + 1
        todos = sum(src.count(m) for m in ("TODO", "FIXME"))
        funcs, no_doc = [], []
        try:
            for node in ast.walk(ast.parse(src)):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    funcs.append(node.name)
                    if not ast.get_docstring(node):
                        no_doc.append(node.name)
        except SyntaxError:
            pass
        files[p.name] = {"lines": lines, "functions": len(funcs),
                         "todo_fixme": todos, "no_docstring": no_doc}
    return {"package": str(pkg_dir.name), "files": files,
            "totals": {"files": len(files),
                       "lines": sum(f["lines"] for f in files.values()),
                       "functions": sum(f["functions"] for f in files.values()),
                       "todo_fixme": sum(f["todo_fixme"] for f in files.values()),
                       "no_docstring": sum(len(f["no_docstring"]) for f in files.values())}}


async def path_a(client, meter: Meter) -> str:
    # деతерминированная процедура: реальный Interpreter Raidho, 0 LLM-вызовов
    facts_holder: dict = {}

    async def h_execute(label, args, mode, regs, model=None):
        facts_holder["facts"] = scan_package(PROJECT_ROOT / TARGET)
        return facts_holder["facts"]

    async def h_report(label, args, mode, regs, model=None):
        return regs.get("facts")

    interp = Interpreter(handlers={"execute": h_execute, "report": h_report})
    procedure = {
        "id": "audit_py_package",
        "body": {
            "steps": [
                {"id": "s1", "op": "execute", "mode": "deterministic",
                 "label": "scan package metrics", "args": {}, "out": "facts",
                 "next": "s2"},
                {"id": "s2", "op": "report", "mode": "deterministic",
                 "label": "emit facts", "args": {}, "out": "result"},
            ],
            "entry": "s1",
            "registers": ["facts", "result"],
        },
    }
    await interp.arun(procedure)
    facts = facts_holder["facts"]

    # один LLM-вызов: только написать отчёт по готовым данным
    resp = await client.messages.create(
        model=MODEL, max_tokens=8000, thinking={"type": "adaptive"},
        messages=[{"role": "user", "content":
                   TASK + "\n\nДанные уже собраны детерминированной процедурой "
                   "(доверяй им, ничего не пересчитывай):\n```json\n"
                   + json.dumps(facts, ensure_ascii=False, indent=1) + "\n```"}],
    )
    meter.add(resp.usage)
    guard(meter)
    return "".join(b.text for b in resp.content if b.type == "text")


# ── Path B: pure LLM tool-loop with bash ─────────────────────────────────────

BASH_TOOL = {
    "name": "bash",
    "description": "Выполнить bash-команду в корне проекта и вернуть stdout/stderr.",
    "input_schema": {"type": "object",
                     "properties": {"command": {"type": "string"}},
                     "required": ["command"]},
}


def run_bash(cmd: str) -> str:
    try:
        r = subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT, timeout=30,
                           capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip() or "(пустой вывод)"
    except subprocess.TimeoutExpired:
        out = "(timeout 30s)"
    return out[:6000]


async def path_b(client, meter: Meter) -> str:
    messages = [{"role": "user", "content": TASK}]
    for i in range(MAX_ITERS):
        resp = await client.messages.create(
            model=MODEL, max_tokens=8000, thinking={"type": "adaptive"},
            tools=[BASH_TOOL], messages=messages,
        )
        meter.add(resp.usage)
        guard(meter)
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                print(f"    [iter {i+1}] bash: {b.input.get('command','')[:90]}")
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": run_bash(b.input.get("command", ""))})
        messages.append({"role": "user", "content": results})
    return "(достигнут предел итераций)"


# ── main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic()  # ключ из ANTHROPIC_API_KEY

    print(f"═══ Та же задача, тот же {MODEL}: Raidho-процедура vs чистый tool-loop ═══\n")
    results = {}
    for name, fn in (("A: Raidho (процедура + 1 вызов)", path_a),
                     ("B: чистый Opus 4.8 (tool-loop)", path_b)):
        meter = Meter()
        t0 = time.perf_counter()
        report = await fn(client, meter)
        dt = time.perf_counter() - t0
        t = meter.totals()
        results[name] = {"meter": meter, "dt": dt, "report": report, "t": t}
        print(f"\n  ── {name} ──")
        print(f"  LLM-вызовов: {len(meter.calls)} | время: {dt:.1f}s | "
              f"токены: in={t['in']} out={t['out']} cache_w={t['cw']} cache_r={t['cr']} "
              f"| стоимость: ${meter.cost():.4f}\n")

    a, b = results.values()
    ratio_cost = b["meter"].cost() / max(a["meter"].cost(), 1e-9)
    ratio_tok = (b["t"]["in"] + b["t"]["out"]) / max(a["t"]["in"] + a["t"]["out"], 1)
    print("═══ ИТОГ ═══")
    print(f"  стоимость: B/A = ×{ratio_cost:.1f} | токены: B/A = ×{ratio_tok:.1f} | "
          f"время: {b['dt']:.0f}s vs {a['dt']:.0f}s")
    print(f"  суммарно потрачено: ${a['meter'].cost() + b['meter'].cost():.3f}")

    out = Path("/tmp/raidho_bench_reports.md")
    out.write_text("# Path A (Raidho)\n\n" + a["report"]
                   + "\n\n---\n\n# Path B (pure LLM)\n\n" + b["report"],
                   encoding="utf-8")
    print(f"  оба отчёта (для оценки качества): {out}")


if __name__ == "__main__":
    asyncio.run(main())
