"""Council → VSA memory: the verdict is distilled into structural triples and
stored, so consensus surfaces in later recall without bloating history."""
import asyncio

from agent.loop import Session
from agent.memory import AgentMemory, hash_embed


class ScriptedProvider:
    """Debate seats return canned text; the extraction call returns triples JSON."""
    name = "scripted"

    def __init__(self, extract_json):
        self._extract = extract_json

    async def chat(self, system, history, user_text):
        if "Output JSON only" in system:           # the extraction pass
            return self._extract
        if "secretary" in system.lower():           # the verdict pass
            return "Consensus / recommendation: use PyJWT for auth."
        return "AGREE — fine."                       # debate turns


def _session(extract_json, with_memory=True):
    p = ScriptedProvider(extract_json)
    mem = AgentMemory(embed_fn=hash_embed) if with_memory else None
    return Session(p, workdir=".", memory=mem)


def test_verdict_triples_stored_in_memory():
    s = _session('[{"subject":"auth","relation":"uses","object":"PyJWT"}]')
    res = asyncio.run(s.council("which auth lib?", rounds=1))
    assert res["remembered"] == [("auth", "uses", "PyJWT")]
    assert s.memory.n_facts == 1


def test_stored_fact_is_recalled_later():
    # Recall quality is the embedder's job; the hash fallback is bag-of-words and
    # unreliable (documented). Test the intended path with the real embedder when
    # the `embed` extra is present; skip otherwise.
    import importlib.util
    import pytest
    if importlib.util.find_spec("sentence_transformers") is None:
        pytest.skip("semantic embedder (raidho[embed]) not installed")
    p = ScriptedProvider('[{"subject":"auth","relation":"uses","object":"PyJWT"}]')
    s = Session(p, workdir=".", memory=AgentMemory())   # auto-picks the real model
    asyncio.run(s.council("which auth lib?", rounds=1))
    block = s.memory.recall("authentication library")   # paraphrase, no exact "uses/PyJWT"
    assert "PyJWT" in block


def test_remember_false_skips_extraction():
    s = _session('[{"subject":"x","relation":"y","object":"z"}]')
    res = asyncio.run(s.council("q?", rounds=1, remember=False))
    assert res["remembered"] == []
    assert s.memory.n_facts == 0


def test_no_memory_is_safe():
    s = _session('[{"subject":"x","relation":"y","object":"z"}]', with_memory=False)
    res = asyncio.run(s.council("q?", rounds=1))
    assert res["remembered"] == []                  # no crash without memory


def test_bad_extraction_output_is_tolerated():
    s = _session("sorry, I cannot produce JSON here")
    res = asyncio.run(s.council("q?", rounds=1))
    assert res["remembered"] == []                  # best-effort, never breaks council
    assert res["verdict"]                            # verdict still returned


def test_partial_triples_skipped():
    s = _session('[{"subject":"auth","relation":"uses","object":"PyJWT"},'
                 '{"subject":"","relation":"x","object":"y"}]')
    res = asyncio.run(s.council("q?", rounds=1))
    assert res["remembered"] == [("auth", "uses", "PyJWT")]   # incomplete one dropped
