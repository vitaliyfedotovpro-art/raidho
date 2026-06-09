"""Coder tools (bash/read/write/list) + the canonical tool-spec.

The canonical format is a neutral JSON Schema per tool; providers translate it
into their own format. Execution happens in the WORKING directory (workdir),
unsandboxed: the tool is created for a specific workdir by the caller.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

MAX_OUTPUT = 8000  # truncate output so it does not bloat the context

# Canonical tool-spec: name / description / parameters(JSON Schema).
# Providers translate this into input_schema (Anthropic) or function (OpenAI).
TOOLS_SPEC = [
    {
        "name": "bash",
        "description": "Run a shell command in the working directory "
                       "(full access). For building, running, git, search, install.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "shell command"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file (path relative to workdir or absolute).",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write/overwrite a file with the given content.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_dir",
        "description": "List files in a directory (workdir by default).",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": [],
        },
    },
]


class Tools:
    """Tool executor bound to a working directory."""

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
        """Dispatcher: tool name + args → string result."""
        try:
            if name == "bash":
                return await self.bash(args.get("command", ""))
            if name == "read_file":
                p = self._resolve(args["path"])
                if not p.exists():
                    return f"(no file {p})"
                return p.read_text(encoding="utf-8", errors="replace")[:MAX_OUTPUT]
            if name == "write_file":
                p = self._resolve(args["path"])
                p.parent.mkdir(parents=True, exist_ok=True)
                content = args.get("content", "")
                p.write_text(content, encoding="utf-8")
                return f"written: {p} ({len(content)} chars)"
            if name == "list_dir":
                p = self._resolve(args.get("path", "."))
                items = sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir())
                return "\n".join(items) or "(empty)"
            return f"(unknown tool {name})"
        except Exception as e:  # noqa: BLE001 — a tool must not crash the loop
            return f"[tool error {name}: {type(e).__name__}: {e}]"
