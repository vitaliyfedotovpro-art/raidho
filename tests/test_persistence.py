"""Memory persistence across runs: AgentMemory(path=...) loads on start and
save() writes back; a fresh instance at the same path recalls prior facts."""
import asyncio

from agent.loop import Session
from agent.memory import AgentMemory, hash_embed


def test_save_then_load_roundtrip(tmp_path):
    p = str(tmp_path / "mem")
    m1 = AgentMemory(embed_fn=hash_embed, path=p)
    m1.remember("auth", "uses", "PyJWT")
    assert m1.save() is True

    m2 = AgentMemory(embed_fn=hash_embed, path=p)   # new "run", same path
    assert m2.mem.n_facts == 1
    triples = [(m2.mem._names[s], m2.mem._names[r], m2.mem._names[o])
               for (s, r, o) in m2.mem._fact_idx]
    assert ("auth", "uses", "PyJWT") in triples


def test_no_path_is_ephemeral(tmp_path):
    m = AgentMemory(embed_fn=hash_embed)            # no path
    m.remember("a", "b", "c")
    assert m.save() is False                         # nothing persisted


def test_missing_file_starts_empty(tmp_path):
    m = AgentMemory(embed_fn=hash_embed, path=str(tmp_path / "nope"))
    assert m.mem.n_facts == 0                        # no crash on absent file


def test_council_facts_persist_across_sessions(tmp_path):
    p = str(tmp_path / "mem")

    class Scripted:
        name = "scripted"
        async def chat(self, system, history, user_text):
            if "Output JSON only" in system:
                return '[{"subject":"db","relation":"is","object":"postgres"}]'
            if "secretary" in system.lower():
                return "Consensus: use postgres."
            return "AGREE."

    s1 = Session(Scripted(), workdir=str(tmp_path),
                 memory=AgentMemory(embed_fn=hash_embed, path=p))
    asyncio.run(s1.council("which db?", rounds=1))
    assert s1.memory.mem.n_facts == 1

    # brand-new session at the same path sees the council's decision
    s2 = Session(Scripted(), workdir=str(tmp_path),
                 memory=AgentMemory(embed_fn=hash_embed, path=p))
    assert s2.memory.mem.n_facts == 1


def test_code_turn_persists_remember_tool_fact(tmp_path):
    p = str(tmp_path / "mem")

    class NoToolProvider:
        name = "x"
        async def chat(self, system, history, user_text):
            return "ok"
        async def agent_turn(self, system, history, user_text, tools_spec, tools,
                             max_iters=12, on_tool=None):
            await tools("remember", {"subject": "x", "relation": "y", "object": "z"})
            return "done"

    s = Session(NoToolProvider(), workdir=str(tmp_path),
                memory=AgentMemory(embed_fn=hash_embed, path=p))
    asyncio.run(s.code("do it"))
    assert AgentMemory(embed_fn=hash_embed, path=p).mem.n_facts == 1   # persisted
