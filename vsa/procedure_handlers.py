"""
procedure_handlers.py — real handlers for the 7 generic opcodes, wired to bot
capabilities (path B deterministic executor).

Backends are injected (so the module is testable without network / Telegram):
    llm(prompt:str, model:str|None) -> str  — one-shot LLM; model=flash|pro|None —
                                              tier (cheap default / strong when needed)
    bash(cmd:str) -> str              — shell (execute(tool=bash), validate via bash)
    mem                               — OdinnMemory (search over memory)
    web(query:str) -> str | None      — optional (search scope=web)

Each step may carry a model field (flash|pro). Generative steps pick the model
accordingly; deterministic/bash steps ignore it. Default (None) — cheap flash.

Mapping:
    transform(mode=deterministic) → pure function from DETERMINISTIC[label]
    transform(mode=generative)    → llm(...)            (step 3)
    validate                      → VALIDATORS[label] (bash) or LLM judgement (true/false)
    execute(tool=bash)            → bash(...)
    execute(tool=ask_user)        → suspension (handled by interpreter, never reaches here)
    search(scope=memory|web)      → mem.search / web
    compose / report              → llm / formatting

All handlers are async — awaited by Interpreter.arun().
"""

from __future__ import annotations

import re
from typing import Any, Awaitable, Callable

from .procedure_runner import ProcedureError

_SAFE_TOKEN = re.compile(r"^[\w][\w.\-]*$")  # module/package name: no shell metacharacters


# ── Deterministic transform primitives (pure, no network) ────────────────────
def _extract_module_name(args: dict, regs: dict) -> str | None:
    """Name of the missing module from a traceback (ModuleNotFoundError/ImportError)."""
    src = str(args.get("source") or "")
    m = re.search(r"No module named ['\"]([\w.]+)['\"]", src)
    if not m:
        m = re.search(r"(?:ImportError|ModuleNotFoundError)[^\n]*?['\"]([\w.]+)['\"]", src)
    return m.group(1).split(".")[0] if m else None


DETERMINISTIC: dict[str, Callable[[dict, dict], Any]] = {
    "extract_module_name": _extract_module_name,
}


# ── Bash-based validators (deterministic environment checks) ────────────────
async def _check_installed(args: dict, regs: dict, bash: Callable[[str], Awaitable[str]]) -> bool:
    mod = str(args.get("target") or "").strip()
    if not _SAFE_TOKEN.match(mod):
        return False
    out = await bash(
        f"python3 -c \"import importlib.util as u; "
        f"print('YES' if u.find_spec('{mod}') else 'NO')\""
    )
    return "YES" in out


async def _check_version(args: dict, regs: dict, bash: Callable[[str], Awaitable[str]]) -> bool:
    """Return True on version MISMATCH against req.
    If req is empty / not set — treat as no mismatch (False)."""
    mod = str(args.get("target") or "").strip()
    req = str(args.get("req") or "").strip()
    if not _SAFE_TOKEN.match(mod) or not req:
        return False
    out = await bash(
        f"python3 -c \"from importlib.metadata import version; print(version('{mod}'))\""
    )
    cur = out.strip().splitlines()[-1].strip() if out.strip() else ""
    regs["v_cur"] = cur  # side-effect: store current version into register for later steps
    req_num = req.lstrip("=<>~^ ")
    return bool(cur) and cur != req_num


VALIDATORS: dict[str, Callable[[dict, dict, Callable], Awaitable[Any]]] = {
    "check_installed": _check_installed,
    "check_version": _check_version,
}


def _gen_prompt(op: str, label: str, args: dict, regs: dict) -> str:
    """Prompt for a generative step: what we do (op/label) + arguments."""
    parts = [f"You are executing a procedure step. Operation: {op}, action: {label}."]
    if args:
        parts.append("Arguments: " + ", ".join(f"{k}={v!r}" for k, v in args.items() if k != "tool"))
    parts.append("Return only the step result, concise and to the point, no preamble.")
    return "\n".join(parts)


def build_handlers(
    *,
    llm: Callable[[str], Awaitable[str]],
    bash: Callable[[str], Awaitable[str]],
    mem: Any,
    web: Callable[[str], Awaitable[str]] | None = None,
) -> dict[str, Callable]:
    """Assemble the registry of async handlers for Interpreter.arun()."""

    async def h_transform(label, args, mode, regs, model=None):
        if mode == "deterministic":
            fn = DETERMINISTIC.get(label)
            if fn is None:
                raise ProcedureError(
                    f"transform(deterministic) '{label}': no pure function in DETERMINISTIC"
                )
            return fn(args, regs)
        # generative (or mode not set — treated as generative)
        return (await llm(_gen_prompt("transform", label, args, regs), model)).strip()

    async def h_validate(label, args, mode, regs, model=None):
        v = VALIDATORS.get(label)
        if v is not None:
            return await v(args, regs, bash)
        # no registered check → boolean LLM judgement
        out = (await llm(_gen_prompt("validate", label, args, regs) +
                         "\nAnswer STRICTLY with a single word: true or false.", model)).strip().lower()
        return out.startswith("true") or out.startswith("yes")

    async def h_execute(label, args, mode, regs, model=None):
        tool = args.get("tool")
        if tool == "bash":
            return await bash(str(args.get("command", "")))
        # ask_user is intercepted by the interpreter (suspension) and never reaches here
        raise ProcedureError(f"execute: unsupported tool {tool!r}")

    async def h_search(label, args, mode, regs, model=None):
        scope = args.get("scope", "memory")
        query = str(args.get("query", ""))
        if scope == "web":
            if web is None:
                raise ProcedureError("search(scope=web): web backend not provided")
            return await web(query)
        return mem.search(query, int(args.get("top_k", 6)))

    async def h_compose(label, args, mode, regs, model=None):
        if "template" in args:
            return args["template"]
        return (await llm(_gen_prompt("compose", label, args, regs), model)).strip()

    async def h_report(label, args, mode, regs, model=None):
        # report — formatting: by default return content as-is (deterministic)
        return args.get("content", "")

    return {
        "transform": h_transform,
        "validate": h_validate,
        "execute": h_execute,
        "search": h_search,
        "compose": h_compose,
        "report": h_report,
    }
