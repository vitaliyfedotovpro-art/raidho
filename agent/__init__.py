"""Agent layer: provider-pluggable LLM backend + the coder tool-loop.

The canonical tool-spec (see tools.py) is translated into a concrete provider's
format (Anthropic: input_schema; OpenAI-compatible: function) inside
providers.py — the rest of the code is provider-agnostic.
"""
