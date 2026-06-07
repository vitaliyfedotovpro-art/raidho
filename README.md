# Raidho ᚱ

**A coding agent that plans with one model, executes with another, and remembers what it learns.**

Most coding agents are one model in a tool loop. Raidho splits the work: use a
**smart, expensive model to reason and plan**, a **cheap, fast model to execute**,
and a **durable memory** that carries facts across the whole session — all
provider-agnostic, with your own API key.

> The name is the rune *Raidho* (ᚱ) — "journey / movement".

![license](https://img.shields.io/badge/license-AGPL--3.0%20%2F%20Commercial-blue)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![status](https://img.shields.io/badge/status-alpha-orange)

> **Status: alpha.** Tested end-to-end live against DeepSeek; the Claude path uses
> the official Anthropic SDK. No published benchmarks yet (see [Roadmap](#roadmap)).
> APIs may change before 1.0.

## What makes it different

- **Reasoning ≠ execution.** `text` mode (reasoning, no tools) and `code` mode
  (agentic tool loop) can run on **different providers**. Plan on Claude, grind on
  DeepSeek — you choose where the expensive thinking happens and where the cheap
  doing happens.
- **Council mode.** Have two providers *debate* a question and a neutral pass distill
  the consensus (points of agreement, residual disagreements, recommendation) — e.g.
  Claude vs DeepSeek. Depersonalized and provider-pluggable; no built-in personas.
- **Durable, structural memory.** The agent remembers `(subject, relation, object)`
  facts and recalls the relevant ones into its prompt each turn — and can save new
  ones itself via a `remember` tool. It's a Vector Symbolic Architecture (VSA), not
  RAG: facts are composed algebraically, similarity is bit-packed (32× less RAM than
  float, identical ranking). You don't need to know any of that to use it — see
  [docs/MEMORY.md](docs/MEMORY.md) if you want to.
- **Tiny and hackable.** The memory core depends only on `numpy`; the whole agent is
  a handful of files. Swap providers, tools, or the embedder without fighting a
  framework.
- **Bring your own key.** Claude (default), DeepSeek, OpenAI, or any
  OpenAI-compatible endpoint.

## Install

```bash
pip install -e '.[anthropic]'      # Claude backend (official Anthropic SDK)
pip install -e '.[openai-compat]'  # DeepSeek / OpenAI-compatible (httpx)
pip install -e '.[dev]'            # + pytest
```

Python ≥ 3.11.

## Quickstart

### Single provider

```bash
export CODER_PROVIDER=deepseek
export DEEPSEEK_API_KEY=sk-...
coder "create a FastAPI hello-world app and run it"
```

### Plan with Claude, execute with DeepSeek (the point)

```bash
export CODER_PROVIDER=deepseek          # execution (code mode, tool loop)
export DEEPSEEK_API_KEY=sk-...
export CODER_REASON_PROVIDER=anthropic  # reasoning (text mode)
export ANTHROPIC_API_KEY=sk-ant-...
coder                                    # REPL: /text plans on Claude, /code executes on DeepSeek
```

The expensive model is used only where it earns its keep; the token-heavy tool loop
runs on the cheap one.

## Usage

### CLI

```
coder                 # interactive REPL (default mode: code)
coder "<task>"        # headless: run one task, print result, exit
```

In the REPL: `/code` agentic coding, `/text` reasoning chat, `/council <question>`
two-provider debate → consensus, `/quit` to exit.

### Library

```python
import asyncio
from agent.providers import get_provider
from agent.loop import Session
from agent.memory import AgentMemory

reason = get_provider({"provider": "anthropic", "api_key": "sk-ant-..."})        # smart
execute = get_provider({"provider": "deepseek",  "api_key": "sk-...",            # cheap
                        "model": "deepseek-chat"})

session = Session(execute, workdir=".", memory=AgentMemory(), reason_provider=reason)

asyncio.run(session.chat("plan how to add auth to this app"))   # → reason provider
asyncio.run(session.code("implement the plan and add a test"))  # → execution provider
```

Omit `reason_provider` and both modes use the single provider.

### Council: debate → consensus

```python
from agent.council import Council

council = Council(reason, execute, name_a="claude", name_b="deepseek")
result = await council.consensus("pin exact deps or use ranges?", rounds=2)
print(result["verdict"])      # points of agreement / residual disagreements / recommendation
# result["transcript"] holds the full exchange
```

Or `Session(...).council("...")`, which seats `reason_provider` vs `provider`.

## Configuration

| Variable | Meaning | Default |
|---|---|---|
| `CODER_PROVIDER` | execution provider: `anthropic` \| `deepseek` \| `openai` \| `openai-compat` | `anthropic` |
| `CODER_MODEL` | override execution model | provider default |
| `CODER_REASON_PROVIDER` | optional separate provider for `text`/reasoning | = `CODER_PROVIDER` |
| `CODER_REASON_MODEL` | reasoning model | provider default |
| `CODER_BASE_URL` | endpoint URL for `openai-compat` | — |
| `ANTHROPIC_API_KEY` / `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / `CODER_API_KEY` | API keys (provider-specific first, then `CODER_API_KEY`) | — |

See [docs/PROVIDERS.md](docs/PROVIDERS.md) for adding a provider and the auth hook.

## How it works

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — components and data flow.
- [docs/MEMORY.md](docs/MEMORY.md) — the VSA memory model and bit-packing.
- [docs/OPENWEBUI.md](docs/OPENWEBUI.md) — drive Raidho from Open WebUI (chat /
  council / code as selectable models).

## Roadmap

- Reproducible benchmark (success rate vs. single-model baseline) and a workflow demo.
- Optional persistent memory (save/load across runs).
- Pluggable real embedder out of the box (the default is a light hash embedder).

## Security

The `bash` tool runs **unsandboxed** in the working directory; in `code` mode the
model decides which commands to run. Use Raidho only on code and tasks you trust —
ideally inside a container or a throwaway directory. See [SECURITY.md](SECURITY.md).

## License

Dual-licensed: **AGPL-3.0-or-later** for open-source / research / non-commercial use,
or a commercial license — see [COMMERCIAL.md](COMMERCIAL.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and pull requests welcome.
