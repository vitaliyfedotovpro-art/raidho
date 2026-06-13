"""Auto-distillation of procedures from successful tool-loops (opt-in, gated).

The token-saving lever in Raidho is deterministic procedures: a task that
matches a stored procedure runs without the LLM tool-loop (the loop re-pays the
growing context every iteration — see evidence/2026-06-11). This module turns a
successful LLM tool-loop into such a procedure automatically, so the SAME task
costs a fraction next time.

Safety is the whole game — a distilled procedure later runs deterministically,
so a wrong or destructive one would act silently. Defense in depth:

  1. Opt-in only (Session(autodistill=True) / CODER_AUTODISTILL=1).
  2. Only successful LLM-path runs (never re-distill an existing procedure,
     never a failed run).
  3. READ-ONLY only: every captured tool call must be read_file / list_dir, or
     a bash command whose leading word is on an allowlist AND which contains no
     mutation/redirect tokens. Any write (write_file, mutating bash) → skip.
     Writes stay on the LLM path forever; we never auto-replay them.
  4. Bounded & non-trivial: 2..MAX_STEPS tool calls.
  5. The distilled body is data-collection (deterministic reads) + ONE
     generative synthesis step — it produces a real answer (1 cheap LLM call),
     not raw dumps, and replaces the expensive multi-iteration loop.
  6. A cheap LLM safety-verify gate (fail-closed) on top of the static filter.
  7. Stored with neutral fitness; if it ever misbehaves, record_outcome sinks it
     and code() falls back to the LLM path on any procedure crash.
"""
from __future__ import annotations

import json
import re
import shlex

MAX_STEPS = 15

# Leading commands considered read-only / non-destructive.
_SAFE_LEADING = {
    "cat", "ls", "find", "grep", "rg", "head", "tail", "wc", "pwd", "tree",
    "file", "stat", "cut", "sort", "uniq", "diff", "basename", "dirname", "echo",
    "true", "test", "[",
}
# Substrings that, if present anywhere, disqualify a command (mutation/escape).
_DENY = (
    ">", "<", "|", ";", "&", "`", "$(", "rm ", "rmdir", "mv ", "cp ", "dd ",
    "mkfs", "truncate", "shred", "tee", "sudo", "chmod", "chown", "kill",
    "shutdown", "reboot", "ln ", "touch", "mkdir", "install", "apt", "brew",
    "pip ", "npm ", "yarn", "git push", "git reset", "git checkout", "git clean",
    "git commit", "git rm", "sed -i", "curl", "wget", "ssh", "scp", "nc ",
    "python", "perl", "ruby", "node", "bash", "sh ", "eval", "export", "set ",
    ":(){",
)


def _bash_is_readonly(cmd: str) -> bool:
    """Conservative: leading word allowlisted AND no mutation/escape token.
    Dual-use interpreters (python/sed -i/etc.) are excluded — fail closed."""
    c = cmd.strip()
    if not c:
        return False
    low = c.lower()
    if any(tok in low for tok in _DENY):
        return False
    lead = shlex.split(c)[0] if c else ""
    # allow `git status|log|diff|show` (read-only git) even though git is special
    if lead == "git":
        sub = (shlex.split(c)[1:2] or [""])[0]
        return sub in {"status", "log", "diff", "show", "ls-files", "blame"}
    return lead in _SAFE_LEADING


def _call_to_command(name: str, args: dict) -> str | None:
    """Map a captured read-only tool call to an equivalent read-only bash command,
    or None if the call is not safe to replay."""
    if name == "read_file":
        p = str(args.get("path", "")).strip()
        return f"cat -- {shlex.quote(p)}" if p else None
    if name == "list_dir":
        p = str(args.get("path", ".")).strip() or "."
        return f"ls -la -- {shlex.quote(p)}"
    if name == "bash":
        cmd = str(args.get("command", ""))
        return cmd if _bash_is_readonly(cmd) else None
    # write_file, remember, anything else → not replayable read-only
    return None


def distillable(trajectory: list[tuple[str, dict]]) -> tuple[bool, str, list[str]]:
    """Decide if a captured trajectory can become a read-only procedure.
    Returns (ok, reason, commands)."""
    if not trajectory:
        return False, "no tool calls", []
    if len(trajectory) > MAX_STEPS:
        return False, f"too many steps ({len(trajectory)} > {MAX_STEPS})", []
    cmds = []
    for name, args in trajectory:
        cmd = _call_to_command(name, args)
        if cmd is None:
            return False, f"non-read-only call: {name}", []
        cmds.append(cmd)
    if len(cmds) < 2:
        return False, "trivial (<2 read steps)", []
    return True, "ok", cmds


def build_body(task: str, commands: list[str]) -> dict:
    """Deterministic read steps that collect data, then ONE generative synthesis
    step that answers the original task from what was collected."""
    steps, regs = [], []
    for i, cmd in enumerate(commands):
        rid = f"r{i}"
        regs.append(rid)
        steps.append({
            "id": f"s{i}", "op": "execute", "mode": "deterministic",
            "label": f"collect step {i}",
            "args": {"tool": "bash", "command": cmd},
            "out": rid, "next": f"s{i + 1}",
        })
    synth_args = {"task": task}
    synth_args.update({rid: f"${rid}" for rid in regs})
    steps.append({
        "id": f"s{len(commands)}", "op": "compose", "mode": "generative",
        "label": "synthesize the final answer for the task from the collected data",
        "args": synth_args, "out": "result",
    })
    return {"steps": steps, "entry": "s0", "registers": regs + ["result"]}


_VERIFY_PROMPT = (
    "A procedure will be REPLAYED automatically for tasks like the one below. "
    "It runs ONLY these shell commands (already checked to be read-only), then "
    "asks a model to synthesize the answer. Confirm it is safe AND generalizes "
    "to that task class (no one-off paths/values that would be wrong next time).\n"
    "Return ONLY JSON: {{\"safe\": true|false, \"reason\": \"...\"}}.\n\n"
    "Task: {task}\n\nCommands:\n{cmds}"
)


async def verify_safe(llm, task: str, commands: list[str]) -> tuple[bool, str]:
    """Cheap fail-closed LLM gate on top of the static filter."""
    prompt = _VERIFY_PROMPT.format(task=task, cmds="\n".join(commands))
    try:
        raw = await llm("You are a strict safety reviewer. JSON only.", [], prompt)
    except Exception as e:  # provider hiccup → do not store
        return False, f"verify call failed: {e}"
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return False, "verifier returned no JSON"
    try:
        v = json.loads(m.group(0))
    except (ValueError, TypeError):
        return False, "verifier JSON parse failed"
    return bool(v.get("safe")), str(v.get("reason", ""))


def proc_id_for(task: str) -> str:
    """Stable-ish id from the task text."""
    slug = re.sub(r"[^a-z0-9]+", "-", task.lower()).strip("-")[:40] or "task"
    return f"auto-{slug}"
