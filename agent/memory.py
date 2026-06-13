"""VSA memory for the agent: recall of relevant facts into the system prompt +
a `remember` tool (the agent itself decides what to store as a structural fact).

The default embedder is a light deterministic hash (no heavy deps). For quality,
inject sentence-transformers or provider embeddings via embed_fn.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

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


def _semantic_embedder_available() -> bool:
    """True when sentence-transformers is importable (the `embed` extra)."""
    import importlib.util
    return importlib.util.find_spec("sentence_transformers") is not None


class AgentMemory:
    """Wrapper over VSAMemory: remember (store a fact) + recall (search into the prompt).

    Embedder resolution (most semantic wins): an explicitly injected embed_fn;
    otherwise the real sentence-transformers model IF the `embed` extra is
    installed (VSAMemory lazy-loads it); otherwise the hash embedder — recall
    then matches on exact keywords only, and a one-line notice says so."""

    def __init__(self, mem: VSAMemory | None = None, embed_fn=None, D: int = 10_000,
                 path: str | None = None):
        # Persistence: when `path` is set, memory is loaded from disk if present and
        # save() writes it back — facts survive across runs. Per-project by default
        # (the CLI points it at <workdir>/.raidho/memory): a coder agent's facts
        # belong to that codebase.
        self.path = str(path) if path else None
        if mem is None and embed_fn is None:
            if _semantic_embedder_available():
                embed_fn = None        # VSAMemory's default lazy-loads the real model
            else:
                embed_fn = hash_embed
                print("  ⓘ memory: hash embedder — recall matches exact keywords "
                      "only. For semantic recall: pip install 'raidho[embed]'")
        self.embed_fn = embed_fn
        if mem is None and self.path and Path(self.path + ".json").exists():
            try:
                mem = VSAMemory.load(self.path, embed_fn=embed_fn)
                print(f"  ⓘ memory: loaded {mem.n_facts} fact(s) from {self.path}")
            except Exception as e:  # corrupt/old format — start fresh, keep the agent up
                print(f"  ⚠ memory: could not load {self.path} ({e}); starting empty")
                mem = None
        self.mem = mem or VSAMemory(D=D, seed=0, embed_fn=embed_fn)

    def save(self) -> bool:
        """Persist to self.path (no-op if path unset). Returns True if written."""
        if not self.path:
            return False
        try:
            self.mem.save(self.path)
            return True
        except Exception as e:  # never let a save failure crash a turn
            print(f"  ⚠ memory: save to {self.path} failed ({e})")
            return False

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

    def match_procedure(self, context: str, threshold: float = 0.45,
                       top_k: int = 3, use_fitness: bool = True,
                       mode_boosts: dict[str, float] | None = None) -> list[dict]:
        """Which procedures fit the context. use_fitness=True by default —
        in the agent facade, homeostasis is always on (proven-successful
        procedures rise, failed ones sink)."""
        return self.mem.match_trigger(context, threshold=threshold,
                                      top_k=top_k, use_fitness=use_fitness,
                                      mode_boosts=mode_boosts)

    @property
    def n_facts(self) -> int:
        return self.mem.n_facts
