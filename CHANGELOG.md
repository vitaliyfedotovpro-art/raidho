# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/) once it reaches 1.0.

## [Unreleased]

## [0.0.1] — alpha

### Added
- **VSA memory engine** (`vsa/`): MAP primitives (bind/bundle/permute/ground),
  facts (role-binding) and episodes (permutation), `search` / `query` / `save` /
  `load`. Depends only on `numpy`; the embedder is injected (`embed_fn`).
- **Bit-packed similarity** — facts stored as packed bits, compared with
  `popcount(XOR)`; bit-identical ranking to the float dot product at 1/32 the RAM.
  Regression-tested (`tests/test_bitpack.py`).
- **Provider-pluggable backend** (`agent/providers.py`): Claude via the official
  Anthropic SDK (default, `claude-opus-4-8`), and OpenAI-compatible providers
  (DeepSeek, OpenAI, custom endpoints). A canonical tool spec is translated per
  provider. Bring-your-own API key, with a callable auth hook for custom tokens.
- **Tool-using agent loop** (`agent/tools.py`, `agent/loop.py`): `bash`,
  `read_file`, `write_file`, `list_dir`; two modes — `chat` (reasoning) and `code`
  (agentic). A neutral conversation history keeps provider details out of the loop.
- **Memory wired into the agent** (`agent/memory.py`): relevant facts recalled into
  the system prompt before each turn, plus a `remember` tool the agent can call.
- **CLI** (`agent/cli.py`): interactive REPL (`/text`, `/code`, `/quit`) and a
  headless `coder "<task>"` mode.
- Packaging, dual license (AGPL-3.0-or-later + commercial), and documentation.

### Notes
- Tested end-to-end live against DeepSeek. The Claude path is built on the official
  Anthropic SDK but has not been separately load-tested.
- The default embedder is a light deterministic hash; install the `embed` extra and
  inject a real encoder for higher-quality recall.
