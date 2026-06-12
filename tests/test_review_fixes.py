"""Fixes from the 2026-06-11 external review (verified subset): embedder
auto-pickup, retry/backoff in OpenAICompatProvider, history budget in Session,
council secretary passthrough."""
import asyncio

import agent.memory as agent_memory
import agent.providers as providers_mod
from agent.loop import Session
from agent.memory import AgentMemory, hash_embed
from agent.providers import OpenAICompatProvider


# ── embedder auto-pickup ─────────────────────────────────────────────────────

def test_memory_falls_back_to_hash_without_extra(monkeypatch, capsys):
    monkeypatch.setattr(agent_memory, "_semantic_embedder_available", lambda: False)
    m = AgentMemory()
    assert m.embed_fn is hash_embed
    assert "raidho[embed]" in capsys.readouterr().out      # honest notice shown


def test_memory_uses_real_embedder_when_available(monkeypatch):
    monkeypatch.setattr(agent_memory, "_semantic_embedder_available", lambda: True)
    m = AgentMemory()
    assert m.embed_fn is None        # VSAMemory lazy-loads sentence-transformers


def test_memory_explicit_embed_fn_wins(monkeypatch):
    monkeypatch.setattr(agent_memory, "_semantic_embedder_available", lambda: True)
    fn = lambda t: hash_embed(t)     # noqa: E731
    m = AgentMemory(embed_fn=fn)
    assert m.embed_fn is fn


# ── retry/backoff in _post ───────────────────────────────────────────────────

class FakeResponse:
    def __init__(self, status, json_data=None, text="", headers=None):
        self.status_code = status
        self._json = json_data
        self.text = text
        self.headers = headers or {}

    def json(self):
        if self._json is None:
            raise ValueError("not json")
        return self._json


class FakeClient:
    """httpx.AsyncClient stand-in returning a scripted sequence of responses."""
    script: list = []
    posts_made = 0

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, *a, **kw):
        FakeClient.posts_made += 1
        item = FakeClient.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _patched_provider(monkeypatch, script):
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(OpenAICompatProvider, "_BACKOFF_BASE", 0.0)
    FakeClient.script = list(script)
    FakeClient.posts_made = 0
    return OpenAICompatProvider(api_key="k")


def test_post_retries_transient_then_succeeds(monkeypatch):
    import httpx
    p = _patched_provider(monkeypatch, [
        FakeResponse(429, text="rate limited"),
        httpx.ConnectError("boom"),
        FakeResponse(200, json_data={"choices": [{"message": {"content": "ok"}}]}),
    ])
    data = asyncio.run(p._post({"x": 1}))
    assert data["choices"][0]["message"]["content"] == "ok"
    assert FakeClient.posts_made == 3


def test_post_gives_up_with_error_dict(monkeypatch):
    p = _patched_provider(monkeypatch, [FakeResponse(503, text="down")] * 4)
    data = asyncio.run(p._post({"x": 1}))
    assert "error" in data and "503" in data["error"]
    assert FakeClient.posts_made == 4          # 1 + 3 retries, no exception


def test_post_does_not_retry_client_errors(monkeypatch):
    p = _patched_provider(monkeypatch, [FakeResponse(401, text="bad key")])
    data = asyncio.run(p._post({"x": 1}))
    assert "error" in data and "401" in data["error"]
    assert FakeClient.posts_made == 1          # fail fast


def test_post_handles_non_json_body(monkeypatch):
    p = _patched_provider(monkeypatch, [FakeResponse(200, text="<html>gateway</html>")])
    data = asyncio.run(p._post({"x": 1}))
    assert "error" in data and "non-JSON" in data["error"]


# ── history budget ───────────────────────────────────────────────────────────

class FakeProvider:
    name = "fake"

    async def chat(self, system, history, user_text):
        return "r" * 500

    async def agent_turn(self, system, history, user_text, tools_spec, tools,
                         max_iters=12, on_tool=None):
        return "r" * 500


def test_history_trimmed_to_budget(tmp_path):
    s = Session(FakeProvider(), workdir=tmp_path, history_budget=3000)
    for i in range(10):
        asyncio.run(s.chat(f"q{i} " + "x" * 500))
    used = sum(len(m["content"]) for m in s.history)
    assert used <= 3000
    assert s.history[-1]["content"] == "r" * 500       # newest kept
    assert s.history[0]["content"].startswith("q")      # oldest dropped, pairs intact
    assert s.history[0]["content"] != "q0 " + "x" * 500


def test_history_keeps_last_pair_even_over_budget(tmp_path):
    s = Session(FakeProvider(), workdir=tmp_path, history_budget=10)
    asyncio.run(s.chat("a long question over budget"))
    assert len(s.history) == 2                          # never trimmed to empty


# ── council secretary passthrough ────────────────────────────────────────────

class NamedProvider:
    def __init__(self, name):
        self.name = name

    async def chat(self, system, history, user_text):
        return f"AGREE [{self.name}]"


def test_council_secretary_passthrough(tmp_path):
    a, b, sec = NamedProvider("A"), NamedProvider("B"), NamedProvider("SEC")
    s = Session(b, workdir=tmp_path, reason_provider=a)
    res = asyncio.run(s.council("q?", rounds=1, secretary=sec))
    assert "[SEC]" in res["verdict"]


def test_council_secretary_defaults_to_seat_a(tmp_path):
    a, b = NamedProvider("A"), NamedProvider("B")
    s = Session(b, workdir=tmp_path, reason_provider=a)
    res = asyncio.run(s.council("q?", rounds=1))
    assert "[A]" in res["verdict"]
