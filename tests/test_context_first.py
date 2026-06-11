"""Context-first mode: deterministic collector + Session.code wiring.
Measured motivation: evidence/2026-06-11_opus_vs_raidho (loop re-pays context
every iteration; one call with collected evidence closed the gap at x2.6 less)."""
import asyncio

import pytest

from agent.context import collect_context
from agent.loop import Session


# ── collector ────────────────────────────────────────────────────────────────

def _make_tree(tmp_path):
    (tmp_path / "auth.py").write_text("def login():\n    pass  # token check\n")
    (tmp_path / "billing.py").write_text("def charge():\n    pass\n")
    (tmp_path / "README.md").write_text("project docs\n")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "auth.cpython-311.pyc").write_bytes(b"\x00\x01junk")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\x00\x00")
    (tmp_path / "data.bin").write_bytes(b"\x00" * 100)
    return tmp_path


def test_collector_includes_tree_and_relevant_first(tmp_path):
    _make_tree(tmp_path)
    block, stats = collect_context(tmp_path, "fix the login token bug in auth")
    assert "### File tree" in block and "auth.py" in block
    # relevant file content included, and BEFORE the irrelevant one
    assert block.index("===== auth.py =====") < block.index("===== billing.py =====")
    assert "def login" in block
    assert stats["files_included"] >= 2


def test_collector_skips_noise_and_binaries(tmp_path):
    _make_tree(tmp_path)
    block, _ = collect_context(tmp_path, "anything")
    assert "__pycache__" not in block
    assert "logo.png" not in block
    assert "data.bin" not in block         # NUL-byte binary not inlined


def test_collector_respects_budget(tmp_path):
    for i in range(30):
        (tmp_path / f"f{i:02d}.py").write_text("x = 1\n" * 50)
    block, stats = collect_context(tmp_path, "task", char_budget=2000)
    assert len(block) < 2000 + 4000        # tree + header overhead only
    assert stats["files_omitted"] > 0
    assert "omitted by budget" in block


# ── Session wiring ───────────────────────────────────────────────────────────

class FakeProvider:
    name = "fake"

    def __init__(self):
        self.seen_user_texts = []

    async def chat(self, system, history, user_text):
        return "chat-reply"

    async def agent_turn(self, system, history, user_text, tools_spec, tools,
                         max_iters=12, on_tool=None):
        self.seen_user_texts.append(user_text)
        return "done"


def test_context_first_injects_block_into_first_call(tmp_path):
    _make_tree(tmp_path)
    p = FakeProvider()
    s = Session(p, workdir=tmp_path, context_first=True)
    reply = asyncio.run(s.code("audit auth login"))
    assert reply == "done"
    sent = p.seen_user_texts[0]
    assert sent.startswith("audit auth login")
    assert "## Workspace context" in sent and "def login" in sent


def test_context_block_not_stored_in_history(tmp_path):
    _make_tree(tmp_path)
    s = Session(FakeProvider(), workdir=tmp_path, context_first=True)
    asyncio.run(s.code("audit auth"))
    assert s.history[0]["content"] == "audit auth"      # bare task, no block


def test_per_call_override_beats_session_setting(tmp_path):
    _make_tree(tmp_path)
    p = FakeProvider()
    s = Session(p, workdir=tmp_path, context_first=False)
    asyncio.run(s.code("task one", context_first=True))
    asyncio.run(s.code("task two"))                     # session default: off
    assert "## Workspace context" in p.seen_user_texts[0]
    assert "## Workspace context" not in p.seen_user_texts[1]


def test_off_by_default(tmp_path):
    _make_tree(tmp_path)
    p = FakeProvider()
    s = Session(p, workdir=tmp_path)
    asyncio.run(s.code("task"))
    assert "## Workspace context" not in p.seen_user_texts[0]
