# Open WebUI integration

`integrations/openwebui_raidho.py` exposes Raidho inside [Open WebUI](https://openwebui.com)
as selectable models (a *Pipe* Function):

- **Raidho · chat** — reasoning, no tools (safe).
- **Raidho · council** — two-provider debate → consensus (safe).
- **Raidho · code ⚠️** — agentic tool loop. **Disabled by default** (see Security).

## Install

1. Install Raidho into the environment Open WebUI runs in:
   ```bash
   pip install -e '.[anthropic]'      # and/or .[openai-compat]
   ```
2. In Open WebUI: **Workspace → Functions → +**, paste the contents of
   `integrations/openwebui_raidho.py`, save.
3. Open the function's **Valves** and set your providers and keys (see below).
4. The Raidho models now appear in the model selector.

## Valves

| Valve | Meaning |
|---|---|
| `provider` / `model` / `api_key` | execution backend (code mode + the second council seat) |
| `base_url` | endpoint for `openai-compat` |
| `reason_provider` / `reason_model` / `reason_api_key` | reasoning backend (chat + first council seat); blank = same as execution |
| `council_rounds` | debate rounds for the council model |
| `enable_code` | ⚠️ expose the code model (unsandboxed shell) — default **off** |
| `workdir` | working directory for the code model |

Example: `provider = deepseek`, `reason_provider = anthropic` → council and chat reason
on Claude, the code loop runs on DeepSeek.

## ⚠️ Security

The **code** model runs Raidho's `bash` tool **unsandboxed on the Open WebUI host**,
with the privileges of the Open WebUI process, driven by the model. Anyone who can
select that model in your Open WebUI can make it run shell commands on that machine.

- It is **off by default** (`enable_code = false`). Only the `chat` and `council`
  models are exposed until you opt in.
- If you enable it, run Open WebUI (and Raidho) inside a container/VM with no access
  to anything you care about, and restrict who can use the function.
- `chat` and `council` use no tools and are safe to expose.

## Notes

- Responses are returned non-streamed (the final text). Token streaming may be added
  later.
- Each call is stateless per Open WebUI conversation: prior turns are replayed from
  the chat history; long-term VSA memory is not persisted across separate chats in
  this integration.
