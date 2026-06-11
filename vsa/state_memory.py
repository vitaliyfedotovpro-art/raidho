"""VSA memory for SYMBOLIC procedure states (variable binding).

A procedure execution state is a set of register→value bindings (plus
step→outcome categories). VSA packs them into ONE fixed-width hypervector:
  - recall_value: reliably extracts what a register held, even with many
    bindings in superposition (capacity measured: ~128 bindings at D=2048,
    ~512 at D=10000 before accuracy drops below 0.9);
  - remember/match: episodic bank of seen states with outcomes — "I was in
    a similar situation, this is what worked" (exact-source retrieval holds
    at 1.0 even with 50% of bindings replaced, 20k episodes).

Scope boundary: SYMBOLIC values only — statuses, flags, branch keys, step
names, outcome categories. NOT raw text/meaning (embeddings win there; see
VSAMemory for semantic recall).

When an explicit symbol table is available at query time, a plain dict or
sparse pairs beat this on speed and exactness — use StateMemory when the
state must live INSIDE a fixed-width vector: snapshots of bounded size,
unbind-queries without unpacking, vector input to a model.

Episodes are stored bit-packed (D/8 bytes each, ×32 vs float) and the match
bank is cached between calls; ranking is bit-for-bit identical to the float
path (hamming_cosine ≡ dot/D on ±1 vectors, see core.py).
"""
from __future__ import annotations

import numpy as np

from . import core


class StateMemory:
    """VSA encoding of symbolic states (register→value bindings).

        sm = StateMemory()
        hv = sm.encode({"status": "ok", "branch": "pip", "step": "install"})
        sm.recall_value(hv, "branch")          # -> "pip"
        sm.remember({"status": "ok", ...}, outcome="success")
        sm.match({"status": "ok", ...})        # -> [{"outcome","score","state"}]
    """

    def __init__(self, D: int = core.DEFAULT_D, seed: int = 0) -> None:
        self.D = D
        self._rng = np.random.default_rng(seed)
        self._atoms: dict[str, np.ndarray] = {}     # symbol -> ±1 atom (lazy codebook)
        self._val_names: list[str] = []             # value names (for cleanup)
        self._episodes: list[dict] = []             # remembered states + outcomes
        self._bank: np.ndarray | None = None        # cached packed bank for match
        self._bank_dirty = False

    # ---- symbol codebook ----
    def _atom(self, key: str) -> np.ndarray:
        a = self._atoms.get(key)
        if a is None:
            a = core.random_atoms(1, self.D, self._rng)[0]
            self._atoms[key] = a
        return a

    def _value_atom(self, value: str) -> np.ndarray:
        key = f"val:{value}"
        if key not in self._atoms and value not in self._val_names:
            self._val_names.append(value)
        return self._atom(key)

    # ---- encoding / recall (variable binding) ----
    def encode(self, state: dict[str, str]) -> np.ndarray:
        """register→value bindings → one hypervector: bundle(bind(role_k, val_v))."""
        if not state:
            return np.zeros(self.D, dtype=np.float32)
        terms = np.stack([core.bind(self._atom(f"reg:{k}"), self._value_atom(str(v)))
                          for k, v in state.items()])
        return core.bundle(terms)    # rng=None → deterministic (one state = one hv)

    def recall_value(self, state_hv: np.ndarray, register: str) -> str | None:
        """What a register held: unbind by the register role + cleanup over values.
        Systematic even with many bindings in superposition."""
        if not self._val_names:
            return None
        unbound = core.unbind(state_hv, self._atom(f"reg:{register}"))
        cb = np.stack([self._atom(f"val:{v}") for v in self._val_names])
        sims = core.cosine_to_codebook(unbound, cb)
        j = int(np.argmax(sims))
        return self._val_names[j] if sims[j] > 0 else None

    # ---- episodic memory of states (similar situations → outcomes) ----
    def remember(self, state: dict[str, str], outcome: str, meta: dict | None = None) -> int:
        """Remember a symbolic state with its OUTCOME (success/fail/step-name etc.).
        The hypervector is stored bit-packed (D/8 bytes); empty state raises."""
        if not state:
            raise ValueError("remember: refusing to store an empty state")
        hv = self.encode(state)
        self._episodes.append({"hv": core.pack_bipolar(hv), "state": dict(state),
                               "outcome": outcome, "meta": meta or {}})
        self._bank_dirty = True
        return len(self._episodes) - 1

    def match(self, state: dict[str, str] | np.ndarray, top_k: int = 3) -> list[dict]:
        """Similar seen states by cosine. Returns [{'outcome','score','state','meta'}]
        descending. Uses the cached packed bank; ranking and scores are identical
        to the float path on ±1; a non-bipolar ndarray query gets binarized."""
        if not self._episodes:
            return []
        if isinstance(state, np.ndarray):
            q = state
        else:
            if not state:
                return []
            q = self.encode(state)
        if self._bank is None or self._bank_dirty:
            self._bank = np.stack([e["hv"] for e in self._episodes])
            self._bank_dirty = False
        sims = core.hamming_cosine(self._bank, core.pack_bipolar(q), self.D)
        order = np.argsort(-sims)[:top_k]
        return [{"outcome": self._episodes[i]["outcome"], "score": float(sims[i]),
                 "state": self._episodes[i]["state"], "meta": self._episodes[i]["meta"]}
                for i in order]

    @property
    def n_states(self) -> int:
        return len(self._episodes)

    def __repr__(self) -> str:
        return (f"StateMemory(D={self.D}, values={len(self._val_names)}, "
                f"states={len(self._episodes)})")
