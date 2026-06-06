"""Инструменты кодера (bash/read/write/list) + канонический tool-spec.

Канонический формат — нейтральный JSON Schema на инструмент; провайдеры
транслируют его в свой формат. Исполнение — в РАБОЧЕЙ директории (workdir),
без песочницы: инструмент создаётся под конкретный workdir вызывающим.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

MAX_OUTPUT = 8000  # обрезка вывода, чтобы не раздувать контекст

# Канонический tool-spec: name / description / parameters(JSON Schema).
# Провайдеры переводят это в input_schema (Anthropic) или function (OpenAI).
TOOLS_SPEC = [
    {
        "name": "bash",
        "description": "Выполнить shell-команду в рабочей директории "
                       "(полный доступ). Для сборки, запуска, git, поиска, установки.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "shell-команда"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Прочитать файл (путь относительно workdir или абсолютный).",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Записать/перезаписать файл содержимым.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_dir",
        "description": "Список файлов в директории (по умолчанию workdir).",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": [],
        },
    },
]


class Tools:
    """Исполнитель инструментов, привязанный к рабочей директории."""

    def __init__(self, workdir: str | Path = "."):
        self.workdir = Path(workdir).resolve()

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.workdir / p

    async def bash(self, command: str) -> str:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.workdir),
        )
        out, err = await proc.communicate()
        res = (out.decode("utf-8", "replace") + err.decode("utf-8", "replace")).strip()
        return res[:MAX_OUTPUT] if res else "(no output)"

    async def run(self, name: str, args: dict) -> str:
        """Диспетчер: имя инструмента + аргументы → строковый результат."""
        try:
            if name == "bash":
                return await self.bash(args.get("command", ""))
            if name == "read_file":
                p = self._resolve(args["path"])
                if not p.exists():
                    return f"(нет файла {p})"
                return p.read_text(encoding="utf-8", errors="replace")[:MAX_OUTPUT]
            if name == "write_file":
                p = self._resolve(args["path"])
                p.parent.mkdir(parents=True, exist_ok=True)
                content = args.get("content", "")
                p.write_text(content, encoding="utf-8")
                return f"записано: {p} ({len(content)} символов)"
            if name == "list_dir":
                p = self._resolve(args.get("path", "."))
                items = sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir())
                return "\n".join(items) or "(пусто)"
            return f"(неизвестный инструмент {name})"
        except Exception as e:  # noqa: BLE001 — инструмент не должен ронять петлю
            return f"[ошибка инструмента {name}: {type(e).__name__}: {e}]"
