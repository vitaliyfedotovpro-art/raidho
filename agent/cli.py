"""CLI кодера: два режима (text / code), provider-pluggable бэкенд.

Конфиг из переменных окружения:
  CODER_PROVIDER         anthropic | deepseek | openai | openai-compat (default anthropic)
  CODER_MODEL            переопределить модель исполнения
  CODER_BASE_URL         endpoint для openai-compat
  CODER_REASON_PROVIDER  (опц.) отдельный провайдер для reasoning-режима (text)
  CODER_REASON_MODEL     (опц.) модель reasoning-провайдера
  ANTHROPIC_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY / CODER_API_KEY

Сплит «умная модель думает / дешёвая исполняет»: задай разные провайдеры для
исполнения (CODER_PROVIDER) и рассуждения (CODER_REASON_PROVIDER). Ключ берётся
провайдер-специфичный (ANTHROPIC_API_KEY и т.п.), иначе CODER_API_KEY.

Auth — ключ конечного пользователя (BYO). OAuth-логин не реализован; см. providers.py.
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
    """Провайдер-специфичный ключ, иначе общий CODER_API_KEY."""
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
    """Опциональный отдельный провайдер для reasoning (режим text)."""
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
    print(f"Кодер готов ({backend}, mode={mode}, workdir={workdir}).")
    print("/text — режим обсуждения, /code — агентный кодинг, /quit — выход.\n")
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
        reply = await (session.code(line) if mode == "code" else session.chat(line))
        print(f"\n{reply}\n")


async def run_once(task: str, workdir: str = ".") -> str:
    """Одна задача в агентном режиме → результат (для делегирования/скриптов)."""
    return await _make_session(workdir).code(task)


def main() -> None:
    # `coder "<task>"` — headless одна задача; без аргумента — интерактивный REPL.
    if len(sys.argv) > 1:
        print(asyncio.run(run_once(" ".join(sys.argv[1:]))))
    else:
        asyncio.run(repl())


if __name__ == "__main__":
    main()
