"""Tests for deterministic execution of procedures via Interpreter.run().

Covers:
  1. Linear deterministic procedure (2-3 steps) → status=done, correct registers.
  2. Branch: validate → branch.on picks correct branch by condition.
  3. $ref register resolving between steps.
  4. Suspend on execute(tool=ask_user) without callback → awaiting, then resume → done.
  5. Step budget exceeded → ProcedureError on a looping procedure.
"""

import pytest

from vsa.procedure_runner import Interpreter, ProcedureError


# ── helpers ────────────────────────────────────────────────────────

def _handler_echo(_label, args, _mode, _registers, _model):
    """Deterministic transform: return args['value'] as-is."""
    return args["value"]


def _handler_uppercase(_label, args, _mode, _registers, _model):
    """Deterministic transform: uppercase the given string."""
    return args["text"].upper()


def _handler_validate(_label, args, _mode, _registers, _model):
    """Fake validate: compare two values, return True/False."""
    return args.get("left") == args.get("right")


def _handler_compose(_label, args, _mode, _registers, _model):
    """Compose: concatenate parts with a separator."""
    sep = args.get("sep", " ")
    parts = args.get("parts", [])
    return sep.join(parts)


# ── 1. Simple linear procedure (deterministic) ─────────────────────

def test_linear_two_steps_done_correct_registers():
    """Two deterministic steps produce status=done and expected registers."""
    procedure = {
        "body": {
            "steps": [
                {
                    "id": "s1",
                    "op": "transform",
                    "mode": "deterministic",
                    "label": "echo",
                    "args": {"value": "hello"},
                    "out": "greeting",
                    "next": "s2",
                },
                {
                    "id": "s2",
                    "op": "report",
                    "label": "final",
                    "args": {"value": "$greeting"},
                    "out": "result",
                },
            ],
            "entry": "s1",
            "registers": ["greeting", "result"],
        }
    }

    interp = Interpreter(handlers={
        "transform": _handler_echo,
        "report": _handler_echo,
    })

    result = interp.run(procedure)

    assert result["status"] == "done"
    assert result["registers"]["greeting"] == "hello"
    assert result["registers"]["result"] == "hello"
    assert len(result["trace"]) == 2
    assert result["trace"][0]["id"] == "s1"
    assert result["trace"][1]["id"] == "s2"


def test_linear_three_steps_sequential_flow():
    """Three deterministic steps: transform → transform → report."""
    procedure = {
        "body": {
            "steps": [
                {
                    "id": "a",
                    "op": "transform",
                    "mode": "deterministic",
                    "args": {"text": "world"},
                    "out": "raw",
                    "next": "b",
                },
                {
                    "id": "b",
                    "op": "transform",
                    "mode": "deterministic",
                    "args": {"text": "$raw"},
                    "out": "upper",
                    "next": "c",
                },
                {
                    "id": "c",
                    "op": "report",
                    "args": {"value": "$upper"},
                    "out": "final",
                },
            ],
            "entry": "a",
            "registers": ["raw", "upper", "final"],
        }
    }

    interp = Interpreter(handlers={
        "transform": _handler_uppercase,
        "report": _handler_echo,
    })

    result = interp.run(procedure)

    assert result["status"] == "done"
    assert result["registers"]["raw"] == "WORLD"
    assert result["registers"]["upper"] == "WORLD"  # _handler_uppercase receives "$raw" resolved
    assert result["registers"]["final"] == "WORLD"
    assert len(result["trace"]) == 3


# ── 2. Branch: validate → branch.on ────────────────────────────────

def test_branch_true_path():
    """validate returns True → branch.on['true'] is taken."""
    procedure = {
        "body": {
            "steps": [
                {
                    "id": "chk",
                    "op": "validate",
                    "args": {"left": 7, "right": 7},
                    "out": "valid",
                    "next": "fork",
                },
                {
                    "id": "fork",
                    "op": "branch",
                    "args": {"cond": "$valid"},
                    "on": {"true": "yes", "false": "no"},
                },
                {
                    "id": "yes",
                    "op": "transform",
                    "mode": "deterministic",
                    "args": {"value": "taken-true"},
                    "out": "result",
                },
                {
                    "id": "no",
                    "op": "transform",
                    "mode": "deterministic",
                    "args": {"value": "taken-false"},
                    "out": "result",
                },
            ],
            "entry": "chk",
            "registers": ["valid", "result"],
        }
    }

    interp = Interpreter(handlers={
        "validate": _handler_validate,
        "transform": _handler_echo,
    })

    result = interp.run(procedure)

    assert result["status"] == "done"
    assert result["registers"]["result"] == "taken-true"
    assert result["registers"]["valid"] is True


def test_branch_false_path():
    """validate returns False → branch.on['false'] is taken."""
    procedure = {
        "body": {
            "steps": [
                {
                    "id": "chk",
                    "op": "validate",
                    "args": {"left": 3, "right": 9},
                    "out": "valid",
                    "next": "fork",
                },
                {
                    "id": "fork",
                    "op": "branch",
                    "args": {"cond": "$valid"},
                    "on": {"true": "yes", "false": "no"},
                },
                {
                    "id": "yes",
                    "op": "transform",
                    "mode": "deterministic",
                    "args": {"value": "taken-true"},
                    "out": "result",
                },
                {
                    "id": "no",
                    "op": "transform",
                    "mode": "deterministic",
                    "args": {"value": "taken-false"},
                    "out": "result",
                },
            ],
            "entry": "chk",
            "registers": ["valid", "result"],
        }
    }

    interp = Interpreter(handlers={
        "validate": _handler_validate,
        "transform": _handler_echo,
    })

    result = interp.run(procedure)

    assert result["status"] == "done"
    assert result["registers"]["result"] == "taken-false"
    assert result["registers"]["valid"] is False


def test_branch_missing_key_raises():
    """Branch condition maps to a key not in 'on' → ProcedureError."""
    procedure = {
        "body": {
            "steps": [
                {
                    "id": "chk",
                    "op": "validate",
                    "args": {"left": 1, "right": 2},
                    "out": "valid",
                    "next": "fork",
                },
                {
                    "id": "fork",
                    "op": "branch",
                    "args": {"cond": "$valid"},
                    "on": {"true": "yes"},
                    # 'false' missing
                },
                {"id": "yes", "op": "report", "args": {"value": "ok"}, "out": "x"},
            ],
            "entry": "chk",
            "registers": ["valid", "x"],
        }
    }

    interp = Interpreter(handlers={
        "validate": _handler_validate,
        "report": _handler_echo,
    })

    with pytest.raises(ProcedureError, match="no branch"):
        interp.run(procedure)


# ── 3. $ref register resolving between steps ───────────────────────

def test_ref_resolve_flat_args():
    """$ref in args is replaced by register value from a previous step."""
    procedure = {
        "body": {
            "steps": [
                {
                    "id": "put",
                    "op": "transform",
                    "mode": "deterministic",
                    "args": {"value": 42},
                    "out": "x",
                    "next": "use",
                },
                {
                    "id": "use",
                    "op": "transform",
                    "mode": "deterministic",
                    "args": {"value": "$x"},
                    "out": "y",
                },
            ],
            "entry": "put",
            "registers": ["x", "y"],
        }
    }

    interp = Interpreter(handlers={"transform": _handler_echo})
    result = interp.run(procedure)

    assert result["status"] == "done"
    assert result["registers"]["x"] == 42
    assert result["registers"]["y"] == 42  # resolved from $x


def test_ref_resolve_in_list_arg():
    """$ref inside a list arg is resolved element-wise."""
    procedure = {
        "body": {
            "steps": [
                {
                    "id": "a",
                    "op": "transform",
                    "mode": "deterministic",
                    "args": {"value": "alpha"},
                    "out": "p1",
                    "next": "b",
                },
                {
                    "id": "b",
                    "op": "transform",
                    "mode": "deterministic",
                    "args": {"value": "beta"},
                    "out": "p2",
                    "next": "c",
                },
                {
                    "id": "c",
                    "op": "compose",
                    "args": {"parts": ["$p1", "X", "$p2"], "sep": "-"},
                    "out": "joined",
                },
            ],
            "entry": "a",
            "registers": ["p1", "p2", "joined"],
        }
    }

    interp = Interpreter(handlers={
        "transform": _handler_echo,
        "compose": _handler_compose,
    })

    result = interp.run(procedure)
    assert result["status"] == "done"
    assert result["registers"]["joined"] == "alpha-X-beta"


def test_ref_unresolved_returns_none():
    """$ref to a missing register resolves to None."""
    procedure = {
        "body": {
            "steps": [
                {
                    "id": "use",
                    "op": "transform",
                    "mode": "deterministic",
                    "args": {"value": "$missing"},
                    "out": "z",
                },
            ],
            "entry": "use",
            "registers": ["z"],
        }
    }

    interp = Interpreter(handlers={"transform": _handler_echo})
    result = interp.run(procedure)

    assert result["status"] == "done"
    assert result["registers"]["z"] is None


def test_resolve_with_initial_registers():
    """Initial registers passed to run() are used for $ref resolution."""
    procedure = {
        "body": {
            "steps": [
                {
                    "id": "greet",
                    "op": "transform",
                    "mode": "deterministic",
                    "args": {"value": "$name"},
                    "out": "greeting",
                },
            ],
            "entry": "greet",
            "registers": ["greeting"],
        }
    }

    interp = Interpreter(handlers={"transform": _handler_echo})
    result = interp.run(procedure, registers={"name": "Max"})

    assert result["status"] == "done"
    assert result["registers"]["greeting"] == "Max"


# ── 4. Suspend / resume on execute(tool=ask_user) ──────────────────

def test_ask_user_without_callback_suspends():
    """execute(tool=ask_user) without synchronous ask_user → status=awaiting."""
    procedure = {
        "body": {
            "steps": [
                {
                    "id": "ask",
                    "op": "execute",
                    "args": {"tool": "ask_user", "question": "Name?"},
                    "out": "answer",
                    "next": "done",
                },
                {
                    "id": "done",
                    "op": "report",
                    "args": {"value": "$answer"},
                    "out": "final",
                },
            ],
            "entry": "ask",
            "registers": ["answer", "final"],
        }
    }

    interp = Interpreter(handlers={})  # no ask_user callback
    result = interp.run(procedure)

    assert result["status"] == "awaiting"
    assert result["step_id"] == "ask"
    assert result["question"] == "Name?"
    assert result["out"] == "answer"
    assert "registers" in result
    assert "trace" in result


def test_ask_user_resume_to_done():
    """Resume an awaiting state with an answer → status=done with correct registers."""
    procedure = {
        "body": {
            "steps": [
                {
                    "id": "ask",
                    "op": "execute",
                    "args": {"tool": "ask_user", "question": "Name?"},
                    "out": "answer",
                    "next": "done",
                },
                {
                    "id": "done",
                    "op": "report",
                    "args": {"value": "$answer"},
                    "out": "final",
                },
            ],
            "entry": "ask",
            "registers": ["answer", "final"],
        }
    }

    interp = Interpreter(handlers={"report": _handler_echo})

    # First run — suspends
    awaiting = interp.run(procedure)
    assert awaiting["status"] == "awaiting"

    # Resume with user answer
    resumed = interp.run(procedure, resume={**awaiting, "answer": "Alice"})

    assert resumed["status"] == "done"
    assert resumed["registers"]["answer"] == "Alice"
    assert resumed["registers"]["final"] == "Alice"


def test_ask_user_with_sync_callback_does_not_suspend():
    """When Interpreter has ask_user callback, execute(tool=ask_user) runs synchronously."""
    procedure = {
        "body": {
            "steps": [
                {
                    "id": "ask",
                    "op": "execute",
                    "args": {"tool": "ask_user", "question": "Name?"},
                    "out": "answer",
                    "next": "done",
                },
                {
                    "id": "done",
                    "op": "report",
                    "args": {"value": "$answer"},
                    "out": "final",
                },
            ],
            "entry": "ask",
            "registers": ["answer", "final"],
        }
    }

    interp = Interpreter(
        handlers={"report": _handler_echo},
        ask_user=lambda q: "SyncBob",
    )

    result = interp.run(procedure)

    assert result["status"] == "done"
    assert result["registers"]["answer"] == "SyncBob"
    assert result["registers"]["final"] == "SyncBob"


# ── 5. Step budget exceeded ────────────────────────────────────────

def test_step_budget_exceeded_raises():
    """A procedure that loops (no progress toward END) raises ProcedureError."""
    procedure = {
        "body": {
            "steps": [
                {
                    "id": "loop1",
                    "op": "transform",
                    "mode": "deterministic",
                    "args": {"value": "ping"},
                    "out": "x",
                    "next": "loop2",
                },
                {
                    "id": "loop2",
                    "op": "transform",
                    "mode": "deterministic",
                    "args": {"value": "$x"},
                    "out": "y",
                    "next": "loop1",  # back to loop1 → infinite
                },
            ],
            "entry": "loop1",
            "registers": ["x", "y"],
        }
    }

    interp = Interpreter(
        handlers={"transform": _handler_echo},
        max_steps=5,
    )

    with pytest.raises(ProcedureError, match="step budget exceeded"):
        interp.run(procedure)


def test_budget_ok_for_normal_procedure():
    """A normal 2-step procedure with budget=2 succeeds (no false positive on budget)."""
    procedure = {
        "body": {
            "steps": [
                {
                    "id": "a",
                    "op": "transform",
                    "mode": "deterministic",
                    "args": {"value": "ok"},
                    "out": "v",
                    "next": "b",
                },
                {
                    "id": "b",
                    "op": "report",
                    "args": {"value": "$v"},
                    "out": "r",
                },
            ],
            "entry": "a",
            "registers": ["v", "r"],
        }
    }

    interp = Interpreter(
        handlers={"transform": _handler_echo, "report": _handler_echo},
        max_steps=2,
    )

    result = interp.run(procedure)
    assert result["status"] == "done"


def test_branch_step_counts_toward_budget():
    """Branch step consumes budget; a branch loop also triggers budget error."""
    procedure = {
        "body": {
            "steps": [
                {
                    "id": "chk",
                    "op": "validate",
                    "args": {"left": 1, "right": 1},
                    "out": "v",
                    "next": "fork",
                },
                {
                    "id": "fork",
                    "op": "branch",
                    "args": {"cond": "$v"},
                    "on": {"true": "chk"},  # loops back to chk → infinite
                },
            ],
            "entry": "chk",
            "registers": ["v"],
        }
    }

    interp = Interpreter(
        handlers={"validate": _handler_validate},
        max_steps=4,
    )

    with pytest.raises(ProcedureError, match="step budget exceeded"):
        interp.run(procedure)
