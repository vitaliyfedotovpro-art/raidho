"""Агентный слой: provider-pluggable LLM-бэкенд + tool-loop кодера.

Канонический tool-spec (см. tools.py) транслируется в формат конкретного
провайдера (Anthropic: input_schema; OpenAI-совместимые: function) внутри
providers.py — остальной код провайдер-агностичен.
"""
