"""CLI кодера: два режима (text / code), provider-pluggable бэкенд.

Конфиг из переменных окружения:
  CODER_PROVIDER   anthropic | deepseek | openai | openai-compat  (по умолч. anthropic)
  CODER_MODEL      переопределить модель (опционально)
  CODER_BASE_URL   для openai-compat
  ANTHROPIC_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY / CODER_API_KEY

Auth — ключ конечного пользователя (BYO). OAuth-логин не реализован: при желании
подставьте свой токен через callable-хук в get_provider (см. providers.py).
"""
from __future__ import annotations

import asyncio
import os
import sys

from .loop import Session
from .memory import AgentMemory
from .providers import get_provider


def _config_from_env() -> dict:
    provider = (os.environ.get("CODER_PROVIDER") or "anthropic").lower()
    key = (os.environ.get("CODER_API_KEY")
           or os.environ.get("ANTHROPIC_API_KEY")
           or os.environ.get("DEEPSEEK_API_KEY")
           or os.environ.get("OPENAI_API_KEY"))
    cfg = {"provider": provider, "api_key": key}
    if os.environ.get("CODER_MODEL"):
        cfg["model"] = os.environ["CODER_MODEL"]
    if os.environ.get("CODER_BASE_URL"):
        cfg["base_url"] = os.environ["CODER_BASE_URL"]
    return cfg


async def repl(workdir: str = ".") -> None:
    cfg = _config_from_env()
    provider = get_provider(cfg)
    session = Session(provider, workdir=workdir, memory=AgentMemory())
    mode = "code"  # code | text
    print(f"Кодер готов (provider={provider.name}, mode={mode}, workdir={workdir}).")
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
    session = Session(get_provider(_config_from_env()), workdir=workdir,
                      memory=AgentMemory())
    return await session.code(task)


def main() -> None:
    # `coder "<task>"` — headless одна задача; без аргумента — интерактивный REPL.
    if len(sys.argv) > 1:
        print(asyncio.run(run_once(" ".join(sys.argv[1:]))))
    else:
        asyncio.run(repl())


if __name__ == "__main__":
    main()
