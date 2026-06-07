# Architecture

Raidho is two layers: a **VSA memory engine** (`vsa/`) and a **provider-pluggable
agent** (`agent/`). The agent is provider-agnostic; the memory engine depends only
on `numpy`.

```
┌──────────────────────────── agent/ ─────────────────────────────┐
│  cli.py     REPL + headless entrypoint, env config               │
│  loop.py    Session: two modes (chat / code), wires memory       │
│  providers  Provider abstraction → Anthropic | OpenAI-compatible │
│  tools.py   bash / read_file / write_file / list_dir             │
│  memory.py  AgentMemory: recall into prompt + `remember` tool     │
│  council.py two providers debate → consensus (Provider.chat only) │
└───────────────────────────────┬─────────────────────────────────┘
                                 │  uses
┌───────────────────────────────▼──── vsa/ ───────────────────────┐
│  core.py    MAP primitives: bind/bundle/permute/ground +         │
│             bit-packed similarity (pack/unpack/hamming_cosine)    │
│  memory.py  VSAMemory: facts, episodes, search, save/load        │
└──────────────────────────────────────────────────────────────────┘
```

## `vsa/` — memory engine

- **`core.py`** — Vector Symbolic Architecture (MAP model, bipolar ±1):
  `bind` (elementwise product, self-inverse), `bundle` (majority-sign superposition),
  `permute` (order encoding), `ground` (SimHash: embedding → bipolar hypervector).
  Plus **bit-packed similarity**: facts are stored as packed bits and compared with
  `popcount(XOR)` — `cos = (D − 2·popcount)/D`, bit-identical to the float dot
  product on ±1, at 1/32 the memory. See [MEMORY.md](MEMORY.md).
- **`memory.py`** — `VSAMemory`: stores facts (role-binding), episodes
  (permutation), and exposes `add_triple` / `query` / `search` / `add_episode` /
  `save` / `load`. The embedder is injected (`embed_fn`), so the package pulls no
  heavy ML dependencies.

## `agent/` — the agent

### Provider abstraction (`providers.py`)

Tools are defined once in a **canonical spec** (`name` / `description` / `parameters`
as JSON Schema) and translated per provider:

- **Anthropic** → `{name, description, input_schema}`; the tool loop drives on
  `stop_reason == "tool_use"` using the official SDK (`claude-opus-4-8`,
  adaptive thinking).
- **OpenAI-compatible** (DeepSeek, OpenAI, local gateways) → `{type: "function",
  function: {...}}`; the loop drives on `tool_calls` over `chat/completions`.

`Provider` exposes two methods: `chat` (text mode, no tools) and `agent_turn`
(tool loop). `get_provider(config)` is the factory.

### Session (`loop.py`)

`Session` holds the provider, a `Tools` instance bound to a working directory,
an optional `AgentMemory`, and a **neutral** conversation history
(`[{"role", "content"}]`). Provider-specific message shapes and intermediate
tool rounds stay *inside* the provider and never leak into the neutral history.

Two modes:

- **`chat(text)`** — reasoning, no tools. Uses `reason_provider`.
- **`code(task)`** — agentic tool loop. Uses `provider`.

`Session` takes an optional `reason_provider`; when set, reasoning and execution run
on different backends (e.g. plan on Claude, execute on DeepSeek). When omitted, both
use the single `provider`. Both inject relevant memory into the system prompt before
the call (recall).

### Tools (`tools.py`)

`bash`, `read_file`, `write_file`, `list_dir`, scoped to a working directory.
`bash` runs an unsandboxed shell — see [../SECURITY.md](../SECURITY.md).

### Memory wiring (`memory.py`)

`AgentMemory` wraps `VSAMemory`:

- **recall** — before each turn, `search(query)` returns relevant facts that are
  appended to the system prompt as a "Relevant memory" block;
- **remember** — a tool the agent can call to persist a `(subject, relation, object)`
  triple; available only when a memory is attached.

The default embedder is a light deterministic hash; inject a real embedder via
`embed_fn` for better recall (e.g. `sentence-transformers` from the `embed` extra).

### Council (`council.py`)

`Council` runs a two-provider debate plus a neutral synthesis pass: one seat
proposes a position, the other critiques on the merits or concedes (early stop on
`AGREE`) over a few rounds, then an impartial "secretary" pass distills points of
agreement, residual disagreements, and a recommendation. It uses only
`Provider.chat` — independent of tools, memory, and the working directory.
`Session.council(question)` seats `reason_provider` vs `provider`.

## Data flow (one `code` turn)

```
user task
  └─ Session.code(task)
       ├─ system = base prompt + AgentMemory.recall(task)
       ├─ provider.agent_turn(system, history, task, tools_spec, executor)
       │     loop: model → tool calls → execute (bash/read/write/list/remember) → repeat
       └─ final text appended to neutral history
```
