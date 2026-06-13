# Raidho ᚱ

**A coding agent that plans with one model, executes with another, and remembers what it learns.**

Most coding agents are one model in a tool loop. Raidho splits the work: use a
**smart, expensive model to reason and plan**, a **cheap, fast model to execute**,
and a **durable memory** that carries facts across runs — all
provider-agnostic, with your own API key.

> The name is the rune *Raidho* (ᚱ) — "journey / movement".

![license](https://img.shields.io/badge/license-AGPL--3.0%20%2F%20Commercial-blue)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![status](https://img.shields.io/badge/status-alpha-orange)

> **Status: alpha.** Tested end-to-end live against both backends (DeepSeek and
> Claude through the official Anthropic SDK, including the agentic tool-loop).
> A reproducible real-API benchmark ships in `benchmarks/real_task_opus.py` with
> full evidence (`evidence/2026-06-11_opus_vs_raidho/`): same task, same model —
> deterministic procedure $0.05 / context-first hybrid $0.116 / pure tool-loop
> $0.301; the hybrid matched the loop's report quality at ×2.6 less cost.
> APIs may change before 1.0.

## What makes it different

- **Reasoning ≠ execution.** `text` mode (reasoning, no tools) and `code` mode
  (agentic tool loop) can run on **different providers**. Plan on Claude, grind on
  DeepSeek — you choose where the expensive thinking happens and where the cheap
  doing happens.
- **Council mode.** Have two providers *debate* a question and a neutral pass distill
  the consensus (points of agreement, residual disagreements, recommendation) — e.g.
  Claude vs DeepSeek. Depersonalized and provider-pluggable; no built-in personas.
- **Durable, structural memory — persists across runs.** The agent remembers
  `(subject, relation, object)` facts and recalls the relevant ones into its prompt
  each turn; it saves new ones itself via a `remember` tool, and **council verdicts
  are distilled into facts automatically**. Memory is written to disk per project
  (`<workdir>/.raidho/memory`) and reloaded next run — so a decision reached today
  resurfaces tomorrow, recalled only when relevant (cheap; no history bloat) and
  across languages (a Russian query finds an English fact). It's a Vector Symbolic
  Architecture (VSA), not RAG: facts are composed algebraically, similarity is
  bit-packed (32× less RAM than float, identical ranking). You don't need to know
  any of that to use it — see [docs/MEMORY.md](docs/MEMORY.md) if you want to.
- **Gets cheaper with repetition (opt-in).** Turn on auto-distillation and a successful read-only tool-loop is captured as a deterministic procedure: the next similar task replaces the multi-iteration LLM loop with deterministic data-collection + one synthesis call. Heavily gated for safety (read-only commands and pipelines only, a safety-verify pass, neutral fitness that sinks a bad procedure; writes always stay on the LLM path). Measured live (deepseek-chat, `evidence/2026-06-12_autodistill_curve/`): the win scales with **iteration overhead**, not task size — a repeated multi-step task over small data dropped **×9.6 per repeat (70% over 5 runs)**, while a data-heavy audit (cost dominated by file contents, few iterations to cut) saved ~nothing. Honest rule: it removes repeated per-iteration context cost, not the cost of the data itself.
- **Tiny and hackable.** The memory core depends only on `numpy`; the whole agent is
  a handful of files. Swap providers, tools, or the embedder without fighting a
  framework.
- **Bring your own key.** Claude (default), DeepSeek, OpenAI, or any
  OpenAI-compatible endpoint.

## Install

**Guided (recommended)** — one interactive script that explains every step,
verifies your API key live, runs a real smoke test and shows how to use the
agent (concept: [MavKa](https://github.com/MozgAI/MavKa) by MozgAI):

```bash
bash install.sh
```

**Manual:**

```bash
pip install -e '.[anthropic]'      # Claude backend (official Anthropic SDK)
pip install -e '.[openai-compat]'  # DeepSeek / OpenAI-compatible (httpx)
pip install -e '.[embed]'          # semantic memory (sentence-transformers)
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

In the REPL: `/code` agentic coding, `/text` reasoning chat, `/ctx` toggle
context-first, `/learn` toggle auto-distill, `/council <question>` two-provider debate → consensus, `/quit` to
exit. Memory persists per project at `<workdir>/.raidho/memory` — the REPL shows
how many facts it loaded on start.

### Library

```python
import asyncio
from agent.providers import get_provider
from agent.loop import Session
from agent.memory import AgentMemory

reason = get_provider({"provider": "anthropic", "api_key": "sk-ant-..."})        # smart
execute = get_provider({"provider": "deepseek",  "api_key": "sk-...",            # cheap
                        "model": "deepseek-chat"})

# path=... makes memory persist across runs (omit it for an in-RAM, ephemeral memory)
memory = AgentMemory(path=".raidho/memory")
session = Session(execute, workdir=".", memory=memory, reason_provider=reason)

asyncio.run(session.chat("plan how to add auth to this app"))   # → reason provider
asyncio.run(session.code("implement the plan and add a test"))  # → execution provider
# facts the agent stored are now on disk; a new Session(path=...) reloads them
```

Omit `reason_provider` and both modes use the single provider.

### Council: debate → consensus

```python
from agent.council import Council

council = Council(reason, execute, name_a="claude", name_b="deepseek")
result = await council.consensus("pin exact deps or use ranges?", rounds=2)
print(result["verdict"])      # points of agreement / residual disagreements / recommendation
# result["transcript"] holds the full exchange

# Via a Session with memory, the verdict is auto-distilled into facts and stored:
res = await session.council("pin exact deps or use ranges?")
print(res["remembered"])      # e.g. [("dependencies", "pinned", "exact")] — recalled later
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
| `CODER_CONTEXT_FIRST` | `1` packs the workspace into the first call (fewer tool iterations) | off |
| `CODER_AUTODISTILL` | `1` learns read-only procedures from successful runs (gets cheaper with repetition) | off |
| `CODER_MEMORY` | memory file path; `off` disables persistence | `<workdir>/.raidho/memory` |
| `ANTHROPIC_API_KEY` / `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / `CODER_API_KEY` | API keys (provider-specific first, then `CODER_API_KEY`) | — |

See [docs/PROVIDERS.md](docs/PROVIDERS.md) for adding a provider and the auth hook.

## How it works

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — components and data flow.
- [docs/MEMORY.md](docs/MEMORY.md) — the VSA memory model and bit-packing.
- [docs/OPENWEBUI.md](docs/OPENWEBUI.md) — drive Raidho from Open WebUI (chat /
  council / code as selectable models).

## Roadmap

- Broader benchmark coverage (success rate on a task set vs. single-model baseline; SWE-bench-style eval) — a first real-API cost benchmark with evidence is already in `benchmarks/` + `evidence/`.
- Streaming responses in the Open WebUI plugin (currently the reply lands at once).

Recently shipped: persistent memory across runs · council verdicts saved as facts ·
context-first mode · auto-picked semantic embedder · automatic Open WebUI setup.

## Security

The `bash` tool runs **unsandboxed** in the working directory; in `code` mode the
model decides which commands to run. Use Raidho only on code and tasks you trust —
ideally inside a container or a throwaway directory. See [SECURITY.md](SECURITY.md).

## License

Dual-licensed: **AGPL-3.0-or-later** for open-source / research / non-commercial use,
or a commercial license — see [COMMERCIAL.md](COMMERCIAL.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and pull requests welcome.

## Acknowledgments

- **[Open WebUI](https://github.com/open-webui/open-webui)** — the web interface
  Raidho plugs into. It's an excellent, polished chat UI and a perfect fit for
  this agent; rather than reinvent it, Raidho ships a Pipe plugin and the
  installer can wire itself in automatically. Thanks to the Open WebUI team.
- **[Oles Lytvyn (MozgAI)](https://github.com/MozgAI)** — this project's critic
  throughout its path: his reviews shaped the retry layer, the embedder honesty,
  the history budget and more. The guided installer (`install.sh`) follows the
  concept he pioneered in **[MavKa](https://github.com/MozgAI/MavKa)** —
  an installer that explains everything out of the box ("AI installs itself").
