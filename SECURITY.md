# Security

## ⚠️ The `bash` tool runs unsandboxed

Raidho's `bash` tool executes shell commands **with no sandbox**, in the working
directory, with the privileges of the user running it. In `code` mode the LLM
decides which commands to run. This is powerful and **dangerous**:

- Run Raidho only on code and tasks you trust.
- Prefer a container, VM, or a throwaway directory — not your home directory or a
  repo with secrets.
- Review what the agent does; tool calls are printed (`🔧 name(...)`).

There is no allowlist or confirmation gate in the current version. If you need one,
wrap `agent.tools.Tools` / the executor passed to `Session` with your own policy.

## API keys

- Keys are read from the environment (`CODER_API_KEY`, `ANTHROPIC_API_KEY`,
  `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`) or passed in `config`.
- **Never commit keys.** The shipped `.gitignore` excludes `.env`, `*.key`,
  `*.pem`, `credentials*.json`, and similar — keep it that way.
- `api_key` may be a callable so you can fetch short-lived tokens from your own
  secret store instead of holding a static key (see [docs/PROVIDERS.md](docs/PROVIDERS.md)).

## Network & data

- The only network calls are to the LLM endpoint you configure (Anthropic, DeepSeek,
  OpenAI, or your `CODER_BASE_URL`). There is no telemetry.
- Your prompts, file contents the agent reads, and tool outputs are sent to that
  endpoint as part of normal operation — use a provider you trust for sensitive code.

## Supported versions

Raidho is **alpha**. Security fixes land on the latest `main`. There are no
long-term support branches yet.

## Reporting a vulnerability

Please do **not** open a public issue for security problems. Email
**astrumproject@astrumverum.com** with details and reproduction steps. We'll
acknowledge and work on a fix before any public disclosure.
