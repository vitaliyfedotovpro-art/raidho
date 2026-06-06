"""Сессия кодера: провайдер, инструменты, нейтральная история, системный промпт
и (опционально) VSA-память. Два режима — chat (reasoning, без инструментов) и
code (агентный tool-loop).

Память вшита двусторонне: перед каждым ходом релевантные факты добавляются в
системный промпт (recall), а инструмент `remember` позволяет агенту сохранять
новые факты (доступен только когда память подключена).
"""
from __future__ import annotations

from pathlib import Path

from .memory import REMEMBER_SPEC, AgentMemory
from .providers import Provider
from .tools import TOOLS_SPEC, Tools

DEFAULT_SYSTEM = (
    "Ты — кодер-агент. Помогаешь читать, писать и править код в рабочей директории.\n"
    "В режиме кодинга используй инструменты (bash/read_file/write_file/list_dir) — "
    "выполняй задачу, а не описывай её. Не выдумывай содержимое файлов — читай их. "
    "Перед правкой смотри текущий код. Команды объясняй кратко.\n"
    "Если подключена память — устойчивые факты (решения, имена, сроки) сохраняй "
    "инструментом remember; блок «Релевантная память» в промпте — это recall."
)


def _print_tool(name: str, args: dict) -> None:
    preview = str(args.get("command") or args.get("path") or args.get("subject") or "")[:70]
    print(f"  🔧 {name}({preview})")


class Session:
    def __init__(self, provider: Provider, workdir: str | Path = ".",
                 system: str = DEFAULT_SYSTEM, memory: AgentMemory | None = None):
        self.provider = provider
        self.tools = Tools(workdir)
        self.system = system
        self.memory = memory
        self.history: list[dict] = []  # нейтральная: [{"role","content"}]

    def _system_for(self, text: str) -> str:
        """Базовый промпт + recall релевантной памяти под текущий запрос."""
        if not self.memory:
            return self.system
        block = self.memory.recall(text)
        return f"{self.system}\n\n{block}" if block else self.system

    def _tools_spec(self) -> list:
        return TOOLS_SPEC + ([REMEMBER_SPEC] if self.memory else [])

    async def _run_tool(self, name: str, args: dict) -> str:
        if self.memory and name == "remember":
            return self.memory.remember(
                args.get("subject", ""), args.get("relation", ""), args.get("object", ""))
        return await self.tools.run(name, args)

    async def chat(self, text: str) -> str:
        """Текстовый режим: обсуждение/reasoning, без инструментов (recall активен)."""
        reply = await self.provider.chat(self._system_for(text), self.history, text)
        self.history += [{"role": "user", "content": text},
                         {"role": "assistant", "content": reply}]
        return reply

    async def code(self, task: str) -> str:
        """Агентный режим: tool-loop выполняет задачу (recall + remember активны)."""
        reply = await self.provider.agent_turn(
            self._system_for(task), self.history, task,
            self._tools_spec(), self._run_tool, on_tool=_print_tool)
        self.history += [{"role": "user", "content": task},
                         {"role": "assistant", "content": reply}]
        return reply
