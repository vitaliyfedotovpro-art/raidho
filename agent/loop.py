"""Сессия кодера: держит провайдера, инструменты, нейтральную историю и системный
промпт. Два режима — chat (reasoning, без инструментов) и code (агентный tool-loop).
"""
from __future__ import annotations

from pathlib import Path

from .providers import Provider
from .tools import TOOLS_SPEC, Tools

DEFAULT_SYSTEM = (
    "Ты — кодер-агент. Помогаешь читать, писать и править код в рабочей директории.\n"
    "В режиме кодинга используй инструменты (bash/read_file/write_file/list_dir) — "
    "выполняй задачу, а не описывай её. Не выдумывай содержимое файлов — читай их. "
    "Перед правкой смотри текущий код. Команды объясняй кратко."
)


def _print_tool(name: str, args: dict) -> None:
    preview = str(args.get("command") or args.get("path") or "")[:70]
    print(f"  🔧 {name}({preview})")


class Session:
    def __init__(self, provider: Provider, workdir: str | Path = ".",
                 system: str = DEFAULT_SYSTEM):
        self.provider = provider
        self.tools = Tools(workdir)
        self.system = system
        self.history: list[dict] = []  # нейтральная: [{"role","content"}]

    async def chat(self, text: str) -> str:
        """Текстовый режим: обсуждение/reasoning, без инструментов."""
        reply = await self.provider.chat(self.system, self.history, text)
        self.history += [{"role": "user", "content": text},
                         {"role": "assistant", "content": reply}]
        return reply

    async def code(self, task: str) -> str:
        """Агентный режим: tool-loop выполняет задачу в рабочей директории."""
        reply = await self.provider.agent_turn(
            self.system, self.history, task, TOOLS_SPEC, self.tools.run,
            on_tool=_print_tool)
        self.history += [{"role": "user", "content": task},
                         {"role": "assistant", "content": reply}]
        return reply
