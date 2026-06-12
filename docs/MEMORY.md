# The VSA Memory Model

Raidho's memory is a **Vector Symbolic Architecture** (VSA), in the MAP family
(Multiply-Add-Permute), over bipolar ±1 hypervectors of dimension `D` (default
10,000). It is *structural* memory, not RAG: relations and order are encoded
algebraically, and recall is exact for structure and approximate for similarity.

## Primitives (`vsa/core.py`)

| Op | Definition | Use |
|---|---|---|
| `bind(a, b)` | elementwise product `a ⊙ b` (self-inverse on ±1) | attach a role to a value |
| `bundle(V)` | `sign(Σ V)` (majority vote) | superpose several bindings |
| `permute(v, k)` | cyclic shift by `k` | encode position / order |
| `ground(e, P)` | `sign(e · P)` (SimHash) | embedding → bipolar hypervector |

## Entity types (`vsa/memory.py`)

- **Facts** — a triple `(subject, relation, object)` is stored as
  `bundle(bind(subj_role, S), bind(rel_role, R), bind(obj_role, O))`.
  `query(known, target_role)` unbinds the missing role and cleans up to the
  nearest concept. Direction is preserved: `(X, r, Y)` ≠ `(Y, r, X)`.
- **Episodes** — ordered sequences via permutation; `recall_at`, `successor`,
  `episode_order`.

Entity **identity** is decided by string normalization (casefold, Nordic letters,
NFKD, diacritics) plus an alias table — *not* by embedding cosine. Embeddings are
used only for `search` (free-text recall) and grounding.

## Bit-packed similarity (×32 RAM)

Two ±1 vectors satisfy `dot = D − 2·hamming`, where `hamming = popcount(a XOR b)`.
So storing bits and comparing with popcount gives:

```
cos = dot / D = (D − 2·popcount(XOR)) / D
```

This is **bit-identical** to `(codebook @ vec) / D` on ±1, so the ranking
(`argmax` / `argsort`) is exactly the float result — at **1/32** the memory
(`uint8` bits vs `float32`). Facts are stored packed; the float form is
reconstructed on demand only for the top-K unbind (≤5 per query).

`popcount` uses NumPy's native `np.bitwise_count` (NumPy ≥ 2.0) or a byte LUT
fallback. Verified by `tests/test_bitpack.py` (identity, recall before/after
save-load, storage size, episodes).

## The embedder (`embed_fn`)

`VSAMemory` takes an injected `embed_fn: str -> np.ndarray`, so the core needs only
`numpy`. Options:

- **auto (agent default)** — `AgentMemory` picks the real sentence-transformers
  model automatically when the `embed` extra is installed
  (`pip install 'raidho[embed]'`); semantic recall works out of the box.
- **fallback** — without the extra, a light deterministic hash embedder
  (`agent/memory.py:hash_embed`) is used and a one-line notice is printed.
  Know its limit: it is bag-of-words — recall matches exact keywords only
  (no synonyms, no paraphrase). Fine for wiring and small setups; do not
  expect "durable semantic memory" from it.
- **custom** — pass any encoder as `embed_fn` (it always wins over auto-pickup).

## Using it directly

```python
from vsa import VSAMemory
import numpy as np

mem = VSAMemory(D=10_000, seed=0, embed_fn=lambda t: np.random.default_rng(abs(hash(t)) % 2**32).standard_normal(64).astype("float32"))
mem.add_triple("Paris", "capital_of", "France")
mem.query({"subject": "Paris", "relation": "capital_of"}, "object")["answer"]  # "France"
mem.search("which city is the capital of France?")  # [{'triple', 'score', ...}]
mem.save("mem"); VSAMemory.load("mem", embed_fn=...)
```

## In the agent (`agent/memory.py`)

`AgentMemory` adds two things on top of `VSAMemory`:

- `recall(query)` — formats relevant facts (above a score threshold) into a
  "Relevant memory" block for the system prompt;
- `remember(subject, relation, object)` — exposed as a tool so the agent itself
  decides what is worth persisting.
