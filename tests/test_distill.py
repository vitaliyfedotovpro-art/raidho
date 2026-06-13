"""Auto-distillation: a successful read-only tool-loop becomes a deterministic
procedure (+ one synth step); writes and unsafe commands are refused; the gate
is opt-in. Safety is the point — most of these assert REJECTION."""
import asyncio

from agent import distill
from agent.loop import Session
from agent.memory import AgentMemory, hash_embed


# ── static safety filter ─────────────────────────────────────────────────────

def test_readonly_bash_accepted():
    ok, _, cmds = distill.distillable([("bash", {"command": "grep -rn TODO ."}),
                                       ("bash", {"command": "wc -l setup.py"})])
    assert ok and len(cmds) == 2


def test_read_file_and_list_dir_mapped():
    ok, _, cmds = distill.distillable([("read_file", {"path": "a.py"}),
                                       ("list_dir", {"path": "src"})])
    assert ok
    assert cmds[0].startswith("cat -- ") and cmds[1].startswith("ls -la -- ")


def test_write_file_rejected():
    ok, reason, _ = distill.distillable([("read_file", {"path": "a"}),
                                         ("write_file", {"path": "b", "content": "x"})])
    assert not ok and "non-read-only" in reason


def test_mutating_bash_rejected():
    for bad in ["rm -rf build", "echo x > f", "pip install foo", "git push",
                "python3 -c 'open(1,2)'", "cat a && rm b", "sed -i s/a/b/ f"]:
        ok, _, _ = distill.distillable([("bash", {"command": "ls"}),
                                        ("bash", {"command": bad})])
        assert not ok, f"should reject: {bad}"


def test_readonly_git_allowed_mutating_git_rejected():
    ok, _, _ = distill.distillable([("bash", {"command": "git status"}),
                                    ("bash", {"command": "git diff"})])
    assert ok
    ok2, _, _ = distill.distillable([("bash", {"command": "git status"}),
                                     ("bash", {"command": "git reset --hard"})])
    assert not ok2


def test_trivial_and_oversized_rejected():
    assert not distill.distillable([("bash", {"command": "ls"})])[0]          # <2
    big = [("bash", {"command": "ls"})] * (distill.MAX_STEPS + 1)
    assert not distill.distillable(big)[0]                                    # too many


def test_build_body_shape():
    body = distill.build_body("audit", ["ls -la", "cat a.py"])
    ops = [s["op"] for s in body["steps"]]
    assert ops == ["execute", "execute", "compose"]                # reads + 1 synth
    assert body["steps"][-1]["mode"] == "generative"
    assert body["entry"] == "s0" and "result" in body["registers"]


# ── Session integration ──────────────────────────────────────────────────────

class ReadOnlyProvider:
    """agent_turn makes two read-only tool calls; chat (verify/synth) returns canned."""
    name = "ro"

    async def chat(self, system, history, user_text):
        if "safety reviewer" in system.lower():
            return '{"safe": true, "reason": "read-only, generalizes"}'
        return "synthesized answer"

    async def agent_turn(self, system, history, user_text, tools_spec, tools,
                         max_iters=12, on_tool=None):
        for nm, a in [("list_dir", {"path": "."}), ("read_file", {"path": "x"})]:
            if on_tool:
                on_tool(nm, a)
            await tools(nm, a)
        return "done"


def _sess(tmp_path, autodistill=True, provider=None):
    return Session(provider or ReadOnlyProvider(), workdir=str(tmp_path),
                   memory=AgentMemory(embed_fn=hash_embed), autodistill=autodistill)


def test_successful_readonly_run_learns_procedure(tmp_path):
    (tmp_path / "x").write_text("hi")
    s = _sess(tmp_path)
    asyncio.run(s.code("list files and read x"))
    procs = s.memory.mem.procedures
    assert any(p.startswith("auto-") for p in procs)


def test_disabled_by_default(tmp_path):
    (tmp_path / "x").write_text("hi")
    s = _sess(tmp_path, autodistill=False)
    asyncio.run(s.code("list files and read x"))
    assert s.memory.mem.procedures == []


def test_safety_gate_blocks_storage(tmp_path):
    class Unsafe(ReadOnlyProvider):
        async def chat(self, system, history, user_text):
            if "safety reviewer" in system.lower():
                return '{"safe": false, "reason": "one-off path"}'
            return "x"
    (tmp_path / "x").write_text("hi")
    s = _sess(tmp_path, provider=Unsafe())
    asyncio.run(s.code("do it"))
    assert s.memory.mem.procedures == []                # verifier said no → not stored


def test_writing_run_not_distilled(tmp_path):
    class Writer(ReadOnlyProvider):
        async def agent_turn(self, system, history, user_text, tools_spec, tools,
                             max_iters=12, on_tool=None):
            if on_tool:
                on_tool("write_file", {"path": "out", "content": "x"})
            await tools("write_file", {"path": "out", "content": "x"})
            return "done"
    s = _sess(tmp_path, provider=Writer())
    asyncio.run(s.code("write a file"))
    assert s.memory.mem.procedures == []                # writes never auto-replayed
