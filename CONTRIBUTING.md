# Contributing to Raidho

Thanks for your interest. Raidho is small and dependency-light by design — please
keep changes in that spirit.

## Layout

- `vsa/` — the memory engine (depends only on `numpy`). Keep it self-contained and
  free of agent/LLM concerns.
- `agent/` — the provider-pluggable agent (providers, tools, loop, memory, cli).
- `tests/` — regression tests.
- `docs/` — architecture and memory model.

## Dev setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'            # pytest + core
pip install -e '.[anthropic]'      # if working on the Claude path
pip install -e '.[openai-compat]'  # if working on the OpenAI/DeepSeek path
```

Python ≥ 3.11.

## Tests

```bash
python tests/test_bitpack.py   # runnable script
pytest -q                      # or via pytest
```

The bit-pack tests assert the packed-similarity ranking is **bit-identical** to the
float version. Any change to `vsa/core.py` or `vsa/memory.py` must keep them green.
Add tests for new behavior.

## Style

- Match the surrounding code: naming, structure, and comment density.
- Prefer the smallest change that does the job; avoid new dependencies — the core's
  only runtime dependency is `numpy`, LLM SDKs are optional extras.
- Keep providers symmetric: a new provider implements `chat` and `agent_turn`, and
  translates the canonical tool spec (see [docs/PROVIDERS.md](docs/PROVIDERS.md)).
- Keep the neutral history neutral — provider-specific message shapes stay inside
  the provider.

## Pull requests

1. Open an issue first for anything non-trivial, so we can agree on scope.
2. One logical change per PR; keep the diff focused.
3. Include tests and update docs when behavior changes.
4. Make sure `pytest` passes locally.

By contributing you agree your contributions are licensed under the project's
[AGPL-3.0-or-later](LICENSE) (the dual-licensing of the project is handled
separately — see [COMMERCIAL.md](COMMERCIAL.md)).
