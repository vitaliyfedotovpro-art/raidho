"""VSA memory for the agent: recall of relevant facts into the system prompt +
a `remember` tool (the agent itself decides what to store as a structural fact).

The default embedder is a light deterministic hash (no heavy deps). For quality,
inject sentence-transformers or provider embeddings via embed_fn.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

from vsa import VSAMemory

# Word tokenizer for the hash embedder. The character class intentionally covers
# Latin + Cyrillic (RU/UK) so non-Latin words are tokenized too — multilingual by
# design, not a leftover. Extend the class for other scripts as needed.
_WORD = re.compile(r"[a-zа-яёіїєґ0-9]+", re.IGNORECASE)


def hash_embed(text: str, dim: int = 256) -> np.ndarray:
    """Deterministic bag-of-words hash embedding (no external models)."""
    v = np.zeros(dim, dtype=np.float32)
    for tok in _WORD.findall(text.lower()):
        h = int(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).hexdigest(), 16)
        v[h % dim] += 1.0
        v[(h // dim) % dim] -= 1.0
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


# Canonical tool-spec: the agent decides what to remember (a durable structural fact).
REMEMBER_SPEC = {
    "name": "remember",
    "description": "Store a fact in long-term memory as a triple "
                   "(subject, relation, object). For durable things worth coming "
                   "back to later: decisions, preferences, names, deadlines.",
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {"type": "string"},
            "relation": {"type": "string"},
            "object": {"type": "string"},
        },
        "required": ["subject", "relation", "object"],
    },
}


class AgentMemory:
    """Wrapper over VSAMemory: remember (store a fact) + recall (search into the prompt)."""

    def __init__(self, mem: VSAMemory | None = None, embed_fn=None, D: int = 10_000):
        self.embed_fn = embed_fn or (lambda t: hash_embed(t))
        self.mem = mem or VSAMemory(D=D, seed=0, embed_fn=self.embed_fn)

    def remember(self, subject: str, relation: str, obj: str) -> str:
        self.mem.add_triple(subject, relation, obj)
        return f"remembered: ({subject}) —{relation}→ ({obj})"

    def recall(self, query: str, top_k: int = 5, min_score: float = 0.2) -> str:
        """Relevant facts as a text block for the system prompt (or '')."""
        try:
            hits = self.mem.search(query, top_k=top_k)
        except Exception:
            return ""
        hits = [h for h in hits if h["score"] >= min_score]
        if not hits:
            return ""
        lines = [f"- {s} —{r}→ {o}" for (s, r, o) in (h["triple"] for h in hits)]
        return "## Relevant memory\n" + "\n".join(lines)

    @property
    def n_facts(self) -> int:
        return self.mem.n_facts
