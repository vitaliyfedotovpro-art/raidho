# Providers & Authentication

Raidho is provider-agnostic. A canonical tool spec is translated into each
provider's native format, so the agent loop, tools, and memory are written once.

## Built-in providers

| `provider` | Backend | Default model | Transport |
|---|---|---|---|
| `anthropic` (default) | Claude | `claude-opus-4-8` | official `anthropic` SDK |
| `deepseek` | DeepSeek | `deepseek-chat` | `chat/completions` (httpx) |
| `openai` | OpenAI | `gpt-4o` | `chat/completions` (httpx) |
| `openai-compat` | any OpenAI-compatible endpoint | — | `chat/completions` (httpx) |

Install the backend you need:

```bash
pip install -e '.[anthropic]'      # Claude
pip install -e '.[openai-compat]'  # DeepSeek / OpenAI / local gateways
```

## Configuration

Via environment (see the table in the [README](../README.md#configuration)) or
directly:

```python
from agent.providers import get_provider

# Claude (default model claude-opus-4-8)
get_provider({"provider": "anthropic", "api_key": "sk-ant-..."})

# DeepSeek
get_provider({"provider": "deepseek", "api_key": "sk-...", "model": "deepseek-chat"})

# Any OpenAI-compatible endpoint
get_provider({
    "provider": "openai-compat",
    "base_url": "http://localhost:11434/v1/chat/completions",
    "api_key": "...",
    "model": "your-model",
})
```

## Split reasoning / execution across providers

`Session` accepts a separate `reason_provider`. The `text`/reasoning mode uses it;
the `code`/execution tool loop uses the main `provider`. This lets you plan with a
smart, expensive model and execute with a cheap, fast one:

```python
from agent.loop import Session
from agent.providers import get_provider

reason  = get_provider({"provider": "anthropic", "api_key": "sk-ant-..."})  # smart
execute = get_provider({"provider": "deepseek",  "api_key": "sk-..."})      # cheap
session = Session(execute, reason_provider=reason)
```

Or via environment (the CLI wires this for you):

```bash
export CODER_PROVIDER=deepseek          # execution
export CODER_REASON_PROVIDER=anthropic  # reasoning
# keys are resolved per provider: DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, ...
```

Omit `reason_provider` / `CODER_REASON_PROVIDER` and both modes use the one provider.

## Authentication — bring your own key

Raidho uses **your** API key. Provide it via env (`CODER_API_KEY`, or
`ANTHROPIC_API_KEY` / `DEEPSEEK_API_KEY` / `OPENAI_API_KEY`) or `config["api_key"]`.

`api_key` may also be a **callable** (a zero-argument function returning the key),
resolved at call time. This is the extension point for custom auth — e.g. fetching
a short-lived token from your own secret store:

```python
get_provider({"provider": "anthropic", "api_key": lambda: my_token_store.current()})
```

> **OAuth note.** Raidho does not implement an OAuth login flow. "Log in with your
> Claude account" is not a generally available mechanism for third-party tools, so
> the clean, ToS-compatible path is a user-supplied API key. The callable hook above
> is where you would plug your own token retrieval if you have one.

## Adding a provider

Subclass `Provider` and implement two async methods:

```python
from agent.providers import Provider

class MyProvider(Provider):
    name = "my-provider"

    async def chat(self, system: str, history: list, user_text: str) -> str:
        ...  # text mode, no tools

    async def agent_turn(self, system, history, user_text, tools_spec, tools,
                         max_iters=12, on_tool=None) -> str:
        ...  # tool loop: translate tools_spec, call your API, execute via `tools`
```

- `history` is the neutral format `[{"role": "user"|"assistant", "content": str}]`.
- `tools_spec` is the canonical list (`name` / `description` / `parameters`);
  translate it to your API's tool format.
- `tools(name, args)` is the async executor — call it for each tool the model
  requests, then feed results back in your API's shape.
- Return only the final assistant text; keep intermediate tool rounds internal.

Then either register it in `get_provider` or construct it directly and pass it to
`Session`.
