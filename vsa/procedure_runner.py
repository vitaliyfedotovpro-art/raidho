"""
procedure_runner.py — procedure interpreter (procedural memory).

Role split:
  - VSA (vsa/memory.py) STORES the procedure and MATCHES the trigger
    («which procedure fits»).
  - This module EXECUTES the body. The body is a structured program:
      linear steps (next) · branches (branch.on) · registers ($ref) · await flags.

Alphabet — 7 generic opcodes; specificity lives in args + label, not new opcodes:
  search     — lookup (memory/web)
  transform  — data transform (deterministic or generative)
  validate   — condition check → result
  execute    — tool invocation (bash / API / file / ask_user)
  compose    — assemble result from parts
  branch     — conditional jump (if/else) — routing, not computation
  report     — output result

transform.mode ∈ {deterministic, generative}:
  deterministic — pure function, cacheable, no confirmation needed;
  generative    — LLM call, NOT cacheable, may require confirmation
                  (a signal to the executor, not a separate opcode).

ask_user = execute(tool="ask_user"). In a bot (async, Telegram turns) you
cannot wait synchronously, so run() SUSPENDS the procedure: returns
{"status":"awaiting", ...} with serializable state. The external await-machine
saves it, catches the user response and resumes:
run(procedure, resume={**awaiting, "answer": ...}). For tests/CLI you can pass
a synchronous ask_user callback — then no suspension.

Real handlers (bash, LLM, memory lookups) live in bot.py and are injected
from outside via Interpreter(handlers=..., ask_user=...). Here — the engine
and the contract.
"""

from __future__ import annotations

from typing import Any, Callable

OPCODES = ("search", "transform", "validate", "execute", "compose", "branch", "report")

# Opcode handler: (label, args, mode, registers, model) -> value
# model — model tier for the step (flash|pro|None); generative steps choose
# cheap/strong model based on it, deterministic/bash steps ignore it.
Handler = Callable[[str | None, dict, str | None, dict, str | None], Any]


class ProcedureError(RuntimeError):
    """Structural procedure error (broken jump, no handler, step budget exceeded)."""


def _resolve(args: dict, registers: dict) -> dict:
    """$ref → register value; everything else — literal. Flat at top level;
    list values are resolved element-wise (for args like criteria: [$a, "x"])."""
    out: dict = {}
    for k, v in (args or {}).items():
        out[k] = _resolve_value(v, registers)
    return out


def _resolve_value(v: Any, registers: dict) -> Any:
    if isinstance(v, str) and v.startswith("$"):
        return registers.get(v[1:])
    if isinstance(v, list):
        return [_resolve_value(x, registers) for x in v]
    return v


def _branch_key(cond: Any) -> str:
    """Branch key from condition. bool/truthy → 'true'/'false'; otherwise —
    str(cond) (supports multi-way branches on:{'pip':..,'conda':..})."""
    if isinstance(cond, bool):
        return "true" if cond else "false"
    if cond in (None, "", 0):
        return "false"
    return str(cond)


class Interpreter:
    """Executes a procedure body. Opcode handlers are injected from outside."""

    def __init__(
        self,
        handlers: dict[str, Handler] | None = None,
        ask_user: Callable[[str], Any] | None = None,
        max_steps: int = 100,
    ) -> None:
        self.handlers: dict[str, Handler] = dict(handlers or {})
        self.ask_user = ask_user
        self.max_steps = max_steps

    def register(self, opcode: str, fn: Handler) -> None:
        if opcode not in OPCODES:
            raise ValueError(f"unknown opcode: {opcode!r} (not one of the 7 generic)")
        self.handlers[opcode] = fn

    def _init(self, procedure: dict, registers: dict | None, resume: dict | None):
        """Initial execution state (shared by run/arun). Returns
        (steps, regs, cur, trace)."""
        body = procedure["body"]
        steps = {s["id"]: s for s in body["steps"]}
        if resume is not None:
            # Resume: user answer → out-register of ask_user step, continue
            # from its next.
            regs = dict(resume["registers"])
            astep = steps.get(resume["step_id"])
            if astep is None:
                raise ProcedureError(f"resume: no step id={resume['step_id']!r}")
            if astep.get("out"):
                regs[astep["out"]] = resume.get("answer")
            trace = list(resume.get("trace", []))
            trace.append({"id": resume["step_id"], "op": "execute",
                          "label": astep.get("label"), "resumed": True,
                          "value": _summary(resume.get("answer"))})
            cur = astep.get("next", "END")
        else:
            regs = {r: None for r in body.get("registers", [])}
            regs.update(registers or {})
            cur = body.get("entry", body["steps"][0]["id"])
            trace = []
        return steps, regs, cur, trace

    def _prologue(self, steps: dict, cur, trace: list, regs: dict):
        """One control step BEFORE handler invocation. Returns
        ('suspend', awaiting) | ('branch', next_id) | ('call', step, op, args)."""
        step = steps.get(cur)
        if step is None:
            raise ProcedureError(f"jump to non-existent step id={cur!r}")
        op = step["op"]
        if op not in OPCODES:
            raise ProcedureError(f"step {cur}: opcode {op!r} not one of the 7 generic")
        args = _resolve(step.get("args", {}), regs)
        if op == "branch":
            key = _branch_key(args.get("cond"))
            on = step.get("on", {})
            if key not in on:
                raise ProcedureError(f"step {cur} branch: no branch {key!r} in on={list(on)}")
            trace.append({"id": cur, "op": op, "label": step.get("label"), "branch": key})
            return ("branch", on[key])
        # ask_user without synchronous callback → SUSPEND (await-machine outside).
        if op == "execute" and args.get("tool") == "ask_user" and self.ask_user is None:
            return ("suspend", {"status": "awaiting", "step_id": cur,
                                "question": args.get("question", ""), "out": step.get("out"),
                                "registers": regs, "trace": trace})
        return ("call", step, op, args)

    @staticmethod
    def _record(trace: list, cur, step: dict, value) -> None:
        trace.append({"id": cur, "op": step["op"], "label": step.get("label"),
                      "mode": step.get("mode"), "out": step.get("out"),
                      "value": _summary(value)})

    def run(self, procedure: dict, registers: dict | None = None,
            resume: dict | None = None) -> dict:
        """Execute body SYNCHRONOUSLY (handlers — regular functions; tests/CLI).

        Returns {"status":"done", "registers","trace"} or
        {"status":"awaiting", "step_id","question","out","registers","trace"} —
        if we hit execute(tool=ask_user) without a sync ask_user callback
        (state is serializable; resume via run(proc, resume={**awaiting,"answer":…})."""
        steps, regs, cur, trace = self._init(procedure, registers, resume)
        budget = self.max_steps
        while cur not in ("END", None):
            if budget <= 0:
                raise ProcedureError(f"step budget exceeded ({self.max_steps}) — loop in procedure?")
            budget -= 1
            kind, *rest = self._prologue(steps, cur, trace, regs)
            if kind == "suspend":
                return rest[0]
            if kind == "branch":
                cur = rest[0]
                continue
            step, op, args = rest
            value = self._dispatch(step, op, args, regs)
            if step.get("out"):
                regs[step["out"]] = value
            self._record(trace, cur, step, value)
            cur = step.get("next", "END")
        return {"status": "done", "registers": regs, "trace": trace}

    async def arun(self, procedure: dict, registers: dict | None = None,
                   resume: dict | None = None) -> dict:
        """Execute body ASYNCHRONOUSLY (handlers — coroutines).
        Control flow (branches/suspension/resume) identical to run()."""
        steps, regs, cur, trace = self._init(procedure, registers, resume)
        budget = self.max_steps
        while cur not in ("END", None):
            if budget <= 0:
                raise ProcedureError(f"step budget exceeded ({self.max_steps}) — loop in procedure?")
            budget -= 1
            kind, *rest = self._prologue(steps, cur, trace, regs)
            if kind == "suspend":
                return rest[0]
            if kind == "branch":
                cur = rest[0]
                continue
            step, op, args = rest
            if op == "execute" and args.get("tool") == "ask_user":
                value = await self.ask_user(args.get("question", "")) if self.ask_user else None
            else:
                fn = self.handlers.get(op)
                if fn is None:
                    raise ProcedureError(f"step {step['id']}: no handler for opcode {op!r}")
                value = await fn(step.get("label"), args, step.get("mode"), regs, step.get("model"))
            if step.get("out"):
                regs[step["out"]] = value
            self._record(trace, cur, step, value)
            cur = step.get("next", "END")
        return {"status": "done", "registers": regs, "trace": trace}

    def _dispatch(self, step: dict, op: str, args: dict, regs: dict) -> Any:
        # ask_user reaches here ONLY with synchronous callback (tests/CLI):
        # otherwise run() already returned "awaiting" and suspended.
        if op == "execute" and args.get("tool") == "ask_user":
            return self.ask_user(args.get("question", ""))
        fn = self.handlers.get(op)
        if fn is None:
            raise ProcedureError(f"step {step['id']}: no handler for opcode {op!r}")
        return fn(step.get("label"), args, step.get("mode"), regs, step.get("model"))


def _summary(value: Any) -> Any:
    """Short form for trace (don't drag large objects)."""
    if isinstance(value, str):
        return value if len(value) <= 80 else value[:77] + "..."
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return f"[{len(value)} items]"
    if isinstance(value, dict):
        return f"{{{len(value)} keys}}"
    return type(value).__name__
