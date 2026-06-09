"""Coder CLI: two modes (text / code), provider-pluggable backend.

Config from environment variables:
  CODER_PROVIDER         anthropic | deepseek | openai | openai-compat (default anthropic)
  CODER_MODEL            override the execution model
  CODER_BASE_URL         endpoint for openai-compat
  CODER_REASON_PROVIDER  (opt.) separate provider for reasoning mode (text)
  CODER_REASON_MODEL     (opt.) reasoning provider's model
  ANTHROPIC_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY / CODER_API_KEY

The "smart model thinks / cheap model executes" split: set different providers for
execution (CODER_PROVIDER) and reasoning (CODER_REASON_PROVIDER). The key is taken
provider-specific (ANTHROPIC_API_KEY etc.), otherwise CODER_API_KEY.

Auth — the end user's key (BYO). OAuth login is not implemented; see providers.py.
"""
from __future__ import annotations

import asyncio
import os
import sys

from .loop import Session
from .memory import AgentMemory
from .providers import Provider, get_provider

_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "deepseek": "DEEPSEEK_API_KEY",
            "openai": "OPENAI_API_KEY"}


def _key_for(provider: str) -> str | None:
    """Provider-specific key, otherwise the shared CODER_API_KEY."""
    return (os.environ.get(_KEY_ENV.get(provider, ""), "")
            or os.environ.get("CODER_API_KEY")
            or None)


def _config(provider: str, model_env: str) -> dict:
    cfg = {"provider": provider, "api_key": _key_for(provider)}
    if os.environ.get(model_env):
        cfg["model"] = os.environ[model_env]
    if provider == "openai-compat" and os.environ.get("CODER_BASE_URL"):
        cfg["base_url"] = os.environ["CODER_BASE_URL"]
    return cfg


def _main_config() -> dict:
    return _config((os.environ.get("CODER_PROVIDER") or "anthropic").lower(), "CODER_MODEL")


def _reason_provider() -> Provider | None:
    """Optional separate provider for reasoning (text mode)."""
    name = os.environ.get("CODER_REASON_PROVIDER")
    if not name:
        return None
    return get_provider(_config(name.lower(), "CODER_REASON_MODEL"))


def _make_session(workdir: str) -> Session:
    return Session(get_provider(_main_config()), workdir=workdir,
                   memory=AgentMemory(), reason_provider=_reason_provider())


async def repl(workdir: str = ".") -> None:
    session = _make_session(workdir)
    mode = "code"  # code | text
    reason, exe = session.reason_provider.name, session.provider.name
    backend = f"reason={reason} / exec={exe}" if reason != exe else f"provider={exe}"
    print(f"Coder ready ({backend}, mode={mode}, workdir={workdir}).")
    print("/text — discuss, /code — agentic coding, /council <q> — debate "
          "between two providers → consensus, /quit — exit.\n")
    while True:
        try:
            line = input(f"[{mode}] › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/text":
            mode = "text"
            continue
        if line == "/code":
            mode = "code"
            continue
        if line.startswith("/council "):
            res = await session.council(line[len("/council "):].strip())
            print(f"\n══ consensus ══\n{res['verdict']}\n")
            continue
        reply = await (session.code(line) if mode == "code" else session.chat(line))
        print(f"\n{reply}\n")


async def run_once(task: str, workdir: str = ".") -> str:
    """A single task in agentic mode → result (for delegation/scripts)."""
    return await _make_session(workdir).code(task)


def main() -> None:
    # `coder "<task>"` — headless single task; with no argument — interactive REPL.
    if len(sys.argv) > 1:
        print(asyncio.run(run_once(" ".join(sys.argv[1:]))))
    else:
        asyncio.run(repl())


if __name__ == "__main__":
    main()
