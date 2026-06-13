# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/) once it reaches 1.0.

## [Unreleased]

### Fixed (external-review follow-up, verified subset)
- **Embedder honesty + auto-pickup.** `AgentMemory` now uses the real
  sentence-transformers model automatically when the `embed` extra is installed;
  without it the hash fallback prints a one-line notice that recall matches
  exact keywords only. Docs updated to state the bag-of-words limit plainly.
- **Retry/backoff in `OpenAICompatProvider._post`.** Transient errors
  (429/5xx/network) retried with exponential backoff (Retry-After respected);
  client errors fail fast; non-JSON bodies and exhausted retries return a
  graceful `[LLM error: ...]` instead of crashing the session. (The Anthropic
  provider already had this via the official SDK's built-in retries.)
- **History budget in `Session`.** Conversation history is trimmed to a char
  budget (default 120k) by dropping the oldest turn pairs — long sessions no
  longer march into a context-window error. Durable facts belong in memory.
- **Council secretary override.** `Session.council(..., secretary=...)`
  passes through to `Council.consensus`; the default (seat A — a participant)
  is now documented as a potential bias, with a third provider recommended
  for contested questions.

### Added
- **Trilingual installer + Open WebUI acknowledgment.** `install.sh` now asks the language up front (English / Русский / Українська) and renders every user-facing line accordingly via a `t3` helper; `RAIDHO_LANG` overrides for non-interactive runs. README Acknowledgments credit the Open WebUI team — the web UI Raidho plugs into — as a perfect-fit interface.
- **Guided installer** (`install.sh`): one interactive script that explains every
  step bilingually (EN/RU), checks the system, creates a venv, walks through
  provider choice (DeepSeek / Anthropic / reasoning-execution split), shows
  sign-up URLs (QR if `qrencode` is present), verifies API keys with live calls,
  writes a `chmod 600` `.env`, runs a real end-to-end smoke question and prints
  a usage guide. Idempotent — re-running reuses existing keys and converges.
  Non-interactive via `RAIDHO_PROVIDER`/`RAIDHO_EMBED` envs (CI-able). Concept:
  [MavKa](https://github.com/MozgAI/MavKa) by Oles Lytvyn (MozgAI) — see README
  acknowledgments.
- **Automatic Open WebUI setup** in the guided installer + `scripts/owui_autowire.py`.
  The installer brings up Open WebUI (official Docker image, or `pip install
  open-webui` fallback into the same venv as Raidho), installs Raidho where Open
  WebUI can import it, then wires the plugin entirely through the Open WebUI REST
  API — admin signup, function create/update, Valves from `.env`, enable, and a
  live-answer verification through the web stack. No manual paste; idempotent
  (re-runs update in place and keep the function enabled). Verified end-to-end
  live against Open WebUI 0.9.x. The `code` model stays disabled (unsandboxed).
- **Context-first coding mode** (`agent/context.py`, `Session(context_first=True)`,
  `Session.code(..., context_first=...)`, env `CODER_CONTEXT_FIRST=1`, REPL `/ctx`):
  a deterministic collector packs the file tree + task-relevant sources (cheap
  keyword relevance, char budget, binaries/noise pruned) into the FIRST call, so
  the model does not burn loop iterations on discovery — the growing context is
  otherwise re-paid every iteration. Tools stay available for actions and for
  files omitted by budget; the context block is per-call evidence and is not
  stored in history. Measured motivation and live verification:
  `evidence/2026-06-11_opus_vs_raidho` (hybrid = quality ≥ pure loop at ×2.6
  less cost, ×3.1 fewer tokens); live run answered a code question in a single
  call with zero tool iterations.
- **Open WebUI integration** (`integrations/openwebui_raidho.py`): a Pipe Function
  exposing Raidho as selectable models — `chat`, `council`, and (opt-in) `code`.
  Providers/keys configured via Valves. The `code` model runs an unsandboxed shell
  on the host and is **disabled by default** (`enable_code`). See docs/OPENWEBUI.md.
- **Council mode** (`agent/council.py`): two providers debate a question (propose →
  critique/concede over N rounds, early stop on `AGREE`), then a neutral "secretary"
  pass distills points of agreement, residual disagreements, and a recommendation.
  Depersonalized and provider-pluggable (e.g. Claude vs DeepSeek). Exposed as
  `Council.consensus(...)`, `Session.council(...)`, and the REPL `/council <q>`.
  Verified live with two DeepSeek models.
- **Split reasoning / execution providers.** `Session` accepts an optional
  `reason_provider`; `text`/reasoning runs on it while `code`/execution runs on the
  main provider (e.g. plan on Claude, execute on DeepSeek). CLI wiring via
  `CODER_REASON_PROVIDER` / `CODER_REASON_MODEL`; keys resolved per provider.
  Backward-compatible: omit it and both modes use the single provider.

### Changed
- README rewritten benefit-first: leads with the reasoning/execution split and
  durable memory rather than the underlying VSA machinery.

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
