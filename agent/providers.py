"""Provider-pluggable LLM backend.

The canonical tool-spec (tools.py) is translated into the provider's format:
  • Anthropic — {name, description, input_schema}; tool-loop by stop_reason="tool_use".
  • OpenAI-compatible (DeepSeek etc.) — {type:"function", function:{...}}; tool_calls.

History is neutral: list[{"role": "user"|"assistant", "content": str}]. Tool rounds
in the agent loop run in the internal (native) format and do not leak outward —
only the final text is returned.

Auth: api_key may be a string OR a zero-arg callable (a hook — e.g. so an advanced
user can supply their own OAuth token). It is resolved at call time.
"""
from __future__ import annotations

import json
from typing import Awaitable, Callable

ANTHROPIC_DEFAULT_MODEL = "claude-opus-4-8"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/chat/completions"
OPENAI_BASE_URL = "https://api.openai.com/v1/chat/completions"

OnTool = Callable[[str, dict], None] | None
ToolRunner = Callable[[str, dict], Awaitable[str]]


def _resolve_key(key) -> str:
    """String or callable hook → key string."""
    return key() if callable(key) else (key or "")


def _preview(args: dict) -> str:
    return str(args.get("command") or args.get("path") or "")[:70]


class Provider:
    """Base interface. chat — text mode, agent_turn — coding with a tool-loop."""

    name = "base"

    async def chat(self, system: str, history: list, user_text: str) -> str:
        raise NotImplementedError

    async def agent_turn(self, system: str, history: list, user_text: str,
                         tools_spec: list, tools: ToolRunner,
                         max_iters: int = 12, on_tool: OnTool = None) -> str:
        raise NotImplementedError


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, api_key=None, model: str = ANTHROPIC_DEFAULT_MODEL):
        self._key = api_key
        self.model = model

    def _client(self):
        from anthropic import AsyncAnthropic  # lazy import — dependency is optional
        return AsyncAnthropic(api_key=_resolve_key(self._key) or None)

    @staticmethod
    def _tools(tools_spec: list) -> list:
        return [{"name": t["name"], "description": t["description"],
                 "input_schema": t["parameters"]} for t in tools_spec]

    @staticmethod
    def _text(content) -> str:
        return "".join(b.text for b in content if b.type == "text") or "(empty)"

    async def chat(self, system: str, history: list, user_text: str) -> str:
        client = self._client()
        msgs = list(history) + [{"role": "user", "content": user_text}]
        resp = await client.messages.create(
            model=self.model, max_tokens=16000, system=system,
            thinking={"type": "adaptive"}, messages=msgs)
        return self._text(resp.content)

    async def agent_turn(self, system, history, user_text, tools_spec, tools,
                         max_iters=12, on_tool=None) -> str:
        client = self._client()
        atools = self._tools(tools_spec)
        msgs = list(history) + [{"role": "user", "content": user_text}]
        for _ in range(max_iters):
            resp = await client.messages.create(
                model=self.model, max_tokens=16000, system=system,
                tools=atools, thinking={"type": "adaptive"}, messages=msgs)
            if resp.stop_reason != "tool_use":
                return self._text(resp.content)
            msgs.append({"role": "assistant", "content": resp.content})
            results = []
            for b in resp.content:
                if b.type == "tool_use":
                    if on_tool:
                        on_tool(b.name, dict(b.input))
                    out = await tools(b.name, dict(b.input))
                    results.append({"type": "tool_result",
                                    "tool_use_id": b.id, "content": out})
            msgs.append({"role": "user", "content": results})
        return "(agent iteration limit reached)"


class OpenAICompatProvider(Provider):
    """OpenAI-compatible chat/completions: DeepSeek, OpenAI, local gateways."""

    def __init__(self, api_key=None, model: str = "deepseek-chat",
                 base_url: str = DEEPSEEK_BASE_URL, name: str = "openai-compat"):
        self._key = api_key
        self.model = model
        self.base_url = base_url
        self.name = name

    @staticmethod
    def _tools(tools_spec: list) -> list:
        return [{"type": "function", "function": {
            "name": t["name"], "description": t["description"],
            "parameters": t["parameters"]}} for t in tools_spec]

    # transient: rate limit / server-side; everything else fails fast
    _RETRY_STATUS = {429, 500, 502, 503, 529}
    _RETRIES = 3            # attempts after the first try
    _BACKOFF_BASE = 1.5     # seconds; doubled per attempt (tests set it to 0)

    async def _post(self, payload: dict) -> dict:
        """POST with status check + exponential backoff on transient errors.
        Never raises: exhausted retries / non-JSON bodies come back as
        {"error": ...} — the callers already render that as [LLM error: ...].
        (The Anthropic provider gets the equivalent from the official SDK,
        which auto-retries 429/5xx itself.)"""
        import asyncio
        import httpx  # lazy import
        last: dict = {"error": "no attempts made"}
        async with httpx.AsyncClient(timeout=120) as c:
            for attempt in range(1 + self._RETRIES):
                retry_after = None
                try:
                    r = await c.post(
                        self.base_url,
                        headers={"Authorization": f"Bearer {_resolve_key(self._key)}"},
                        json=payload)
                except httpx.HTTPError as e:        # network/timeout — transient
                    last = {"error": f"{type(e).__name__}: {e}"}
                else:
                    if r.status_code not in self._RETRY_STATUS:
                        try:
                            return r.json()
                        except ValueError:
                            return {"error": f"HTTP {r.status_code}: non-JSON body "
                                             f"{r.text[:200]!r}"}
                    last = {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
                    retry_after = r.headers.get("retry-after")
                if attempt < self._RETRIES:
                    try:
                        delay = float(retry_after)
                    except (TypeError, ValueError):
                        delay = self._BACKOFF_BASE * (2 ** attempt)
                    await asyncio.sleep(min(delay, 30))
        return last

    async def chat(self, system: str, history: list, user_text: str) -> str:
        msgs = [{"role": "system", "content": system}] + list(history) + \
               [{"role": "user", "content": user_text}]
        data = await self._post({"model": self.model, "temperature": 0, "messages": msgs})
        if "choices" not in data:
            return f"[LLM error: {data.get('error', data)}]"
        return data["choices"][0]["message"].get("content") or "(empty)"

    async def agent_turn(self, system, history, user_text, tools_spec, tools,
                         max_iters=12, on_tool=None) -> str:
        otools = self._tools(tools_spec)
        msgs = [{"role": "system", "content": system}] + list(history) + \
               [{"role": "user", "content": user_text}]
        for _ in range(max_iters):
            data = await self._post({"model": self.model, "temperature": 0,
                                     "messages": msgs, "tools": otools})
            if "choices" not in data:
                return f"[LLM error: {data.get('error', data)}]"
            m = data["choices"][0]["message"]
            msgs.append(m)
            tcs = m.get("tool_calls")
            if not tcs:
                return m.get("content") or "(empty)"
            for tc in tcs:
                fn = tc["function"]["name"]
                try:
                    fargs = json.loads(tc["function"]["arguments"] or "{}")
                except Exception:
                    fargs = {}
                if on_tool:
                    on_tool(fn, fargs)
                out = await tools(fn, fargs)
                msgs.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                             "content": out})
        return "(agent iteration limit reached)"


def get_provider(config: dict) -> Provider:
    """Factory from a config dict.

    config = {
        "provider": "anthropic" | "deepseek" | "openai" | "openai-compat",
        "api_key":  str | callable,   # callable = pluggable auth hook (e.g. OAuth)
        "model":    str,              # optional
        "base_url": str,              # for openai-compat
    }
    """
    kind = (config.get("provider") or "anthropic").lower()
    key = config.get("api_key")
    model = config.get("model")
    if kind == "anthropic":
        return AnthropicProvider(api_key=key, model=model or ANTHROPIC_DEFAULT_MODEL)
    if kind == "deepseek":
        return OpenAICompatProvider(api_key=key, model=model or "deepseek-chat",
                                    base_url=DEEPSEEK_BASE_URL, name="deepseek")
    if kind == "openai":
        return OpenAICompatProvider(api_key=key, model=model or "gpt-4o",
                                    base_url=OPENAI_BASE_URL, name="openai")
    if kind == "openai-compat":
        return OpenAICompatProvider(api_key=key, model=model or "default",
                                    base_url=config.get("base_url", OPENAI_BASE_URL),
                                    name="openai-compat")
    raise ValueError(f"unknown provider: {kind}")
