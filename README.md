# Raidho ᚱ

**A coding agent with composition-episodic VSA memory and a provider-pluggable LLM backend.**

Raidho is a small, dependency-light coding agent. It runs a tool-using agent loop
(`bash` / `read` / `write` / `list`) over your codebase, remembers durable facts in a
Vector Symbolic Architecture (VSA) memory, and talks to whichever LLM you give it a
key for — Claude (default), DeepSeek, or any OpenAI-compatible endpoint.

> The name is the rune *Raidho* (ᚱ) — "journey / movement".

![license](https://img.shields.io/badge/license-AGPL--3.0%20%2F%20Commercial-blue)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![status](https://img.shields.io/badge/status-alpha-orange)

> **Status: alpha.** Tested end-to-end live against DeepSeek. The Claude path is
> built on the official Anthropic SDK. Public APIs may change before 1.0.

## Why

Most agent "memory" is RAG over a vector database. Raidho's memory is **structural**:

- **facts** are stored as role-binding hypervectors (subject / relation / object);
- **episodes** as permutations (order-preserving sequences);
- entity identity is decided by string normalization + an alias table (not cosine);
- similarity is computed with **bit-packed popcount** — 32× less RAM than float,
  with bit-identical ranking.

Before each turn the agent recalls relevant facts into its system prompt, and it can
persist new facts itself through a `remember` tool.

## Features

- 🔌 **Provider-pluggable** — Claude (default), DeepSeek, OpenAI, or any
  OpenAI-compatible endpoint. Bring your own API key.
- 🧠 **VSA memory** — facts + episodes, bit-packed similarity (×32 RAM), recall into
  the prompt, a `remember` tool. Pluggable embedder.
- 🛠 **Tool-using agent loop** — `bash`, `read_file`, `write_file`, `list_dir`.
- 💬 **Two modes** — `text` (reasoning chat, no tools) and `code` (agentic loop).
- 🪶 **Light core** — the memory engine depends only on `numpy`.

## Install

```bash
pip install -e '.[anthropic]'      # Claude backend (official Anthropic SDK)
pip install -e '.[openai-compat]'  # DeepSeek / OpenAI-compatible (httpx)
pip install -e '.[dev]'            # + pytest
```

Python ≥ 3.11.

## Quickstart

### Claude (default)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
coder "create a FastAPI hello-world app and run it"
```

### DeepSeek (or any OpenAI-compatible endpoint)

```bash
export CODER_PROVIDER=deepseek
export DEEPSEEK_API_KEY=sk-...
coder            # interactive REPL
```

## Usage

### CLI

```
coder                 # interactive REPL (default mode: code)
coder "<task>"        # headless: run one task, print result, exit
```

In the REPL: `/code` agentic coding, `/text` reasoning chat, `/quit` to exit.

### Library

```python
import asyncio
from agent.providers import get_provider
from agent.loop import Session
from agent.memory import AgentMemory

provider = get_provider({
    "provider": "deepseek",
    "api_key": "sk-...",
    "model": "deepseek-chat",
})
session = Session(provider, workdir=".", memory=AgentMemory())

asyncio.run(session.code("add a /health endpoint and a test for it"))
asyncio.run(session.chat("what does this module do?"))
```

## Configuration

| Variable | Meaning | Default |
|---|---|---|
| `CODER_PROVIDER` | `anthropic` \| `deepseek` \| `openai` \| `openai-compat` | `anthropic` |
| `CODER_MODEL` | override the model id | provider default |
| `CODER_BASE_URL` | endpoint URL for `openai-compat` | — |
| `CODER_API_KEY` | API key (used if no provider-specific key is set) | — |
| `ANTHROPIC_API_KEY` / `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` | provider keys | — |

See [docs/PROVIDERS.md](docs/PROVIDERS.md) for adding a provider and the auth hook.

## How it works

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — components and data flow.
- [docs/MEMORY.md](docs/MEMORY.md) — the VSA memory model and bit-packing.

## Security

The `bash` tool runs **unsandboxed** in the working directory. Run Raidho only on
code and tasks you trust — ideally inside a container or a throwaway directory.
See [SECURITY.md](SECURITY.md).

## License

Dual-licensed: **AGPL-3.0-or-later** for open-source / research / non-commercial use,
or a commercial license — see [COMMERCIAL.md](COMMERCIAL.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and pull requests welcome.
