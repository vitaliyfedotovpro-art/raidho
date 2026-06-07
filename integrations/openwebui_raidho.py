"""
title: Raidho
author: Raidho
description: Drive Raidho from Open WebUI as selectable models — chat (reasoning),
  council (two-provider debate → consensus), and optionally code (agentic tool loop).
  Providers and keys are configured in the Valves. NOTE: the code model runs an
  UNSANDBOXED shell on the Open WebUI host; it is disabled by default (enable_code).
version: 0.1.0
required_open_webui_version: 0.5.0

Install: `pip install -e .` (this repo) into the environment Open WebUI runs in, then
add this file as a Function in Open WebUI. See docs/OPENWEBUI.md.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from agent.council import Council
from agent.loop import Session
from agent.memory import AgentMemory
from agent.providers import get_provider


def _provider(name: str, model: str, key: str, base_url: str):
    cfg = {"provider": (name or "deepseek").lower(), "api_key": key or None}
    if model:
        cfg["model"] = model
    if base_url and cfg["provider"] == "openai-compat":
        cfg["base_url"] = base_url
    return get_provider(cfg)


def _neutral(messages: list) -> tuple[list, str]:
    """Open WebUI messages → (prior neutral history, last user text)."""
    hist = [{"role": m["role"], "content": m.get("content", "")}
            for m in messages if m.get("role") in ("user", "assistant")]
    last = hist.pop()["content"] if hist and hist[-1]["role"] == "user" else \
        (messages[-1].get("content", "") if messages else "")
    return hist, last


class Pipe:
    class Valves(BaseModel):
        # execution backend (code mode, and the second council seat)
        provider: str = Field(default="deepseek",
                              description="Execution provider: anthropic | deepseek | openai | openai-compat")
        model: str = Field(default="deepseek-chat", description="Execution model id")
        api_key: str = Field(default="", description="API key for the execution provider")
        base_url: str = Field(default="", description="Endpoint URL for openai-compat")
        # reasoning backend (chat mode, and the first council seat). Falls back to execution.
        reason_provider: str = Field(default="", description="Reasoning provider (blank = same as execution)")
        reason_model: str = Field(default="", description="Reasoning model id")
        reason_api_key: str = Field(default="", description="API key for the reasoning provider")
        # behavior
        council_rounds: int = Field(default=2, description="Council debate rounds")
        enable_code: bool = Field(default=False,
                                  description="⚠️ Expose the code model — runs UNSANDBOXED shell on this host")
        workdir: str = Field(default=".", description="Working directory for the code model")

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self):
        models = [
            {"id": "raidho-chat", "name": "Raidho · chat"},
            {"id": "raidho-council", "name": "Raidho · council"},
        ]
        if self.valves.enable_code:
            models.append({"id": "raidho-code", "name": "Raidho · code ⚠️"})
        return models

    def _providers(self):
        v = self.valves
        execute = _provider(v.provider, v.model, v.api_key, v.base_url)
        if v.reason_provider:
            reason = _provider(v.reason_provider, v.reason_model,
                               v.reason_api_key or v.api_key, v.base_url)
        else:
            reason = execute
        return execute, reason

    async def pipe(self, body, __user__=None, __metadata__=None, __event_emitter__=None):
        pid = (body.get("model") or "").rsplit(".", 1)[-1]
        messages = body.get("messages", [])
        history, last = _neutral(messages)
        execute, reason = self._providers()

        if pid == "raidho-council":
            council = Council(reason, execute)
            res = await council.consensus(last, rounds=self.valves.council_rounds)
            lines = [f"**{t['who']}**\n{t['text']}" for t in res["transcript"]]
            return "## Debate\n\n" + "\n\n---\n\n".join(lines) + \
                   "\n\n## Consensus\n\n" + res["verdict"]

        if pid == "raidho-code":
            if not self.valves.enable_code:
                return "⚠️ The code model is disabled. Enable `enable_code` in the Valves " \
                       "(it runs an unsandboxed shell on this host)."
            session = Session(execute, workdir=self.valves.workdir,
                              memory=AgentMemory(), reason_provider=reason)
            session.history = history
            return await session.code(last)

        # default: raidho-chat (reasoning, no tools)
        session = Session(execute, memory=AgentMemory(), reason_provider=reason)
        session.history = history
        return await session.chat(last)
