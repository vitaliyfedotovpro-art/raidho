"""
VSAMemory — full compositional-episodic cognitive memory (Phase 3).

Not a demo: facts (role-binding) + episodes (order via permutation) + entity
normalization (dirty variants → one canonical form) + persistence (survives
sessions). Validated in Phase 0/1/2 (the algebra holds, grounding survived, the
discriminator beats raw cosine on real retrieval).

Geometry (MAP, bipolar, D=10k):
    fact      = bundle( R_subj⊗a(s), R_rel⊗a(r), R_obj⊗a(o) )
    episode   = bundle( ρ⁰a(e₀), ρ¹a(e₁), …, ρⁿa(eₙ) )   (ρ = cyclic shift)
    a(concept)= ground(embedding) — SimHash; close concepts → close atoms.

The embedder is injectable (`embed_fn`) — for deterministic, model-free tests;
defaults to sentence-transformers (lazy).
"""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from pathlib import Path
from typing import Callable

import numpy as np

from . import core

# Special letters that NFKD does not decompose (separate letters, not base+mark).
_SCAND = {"ð": "d", "þ": "th", "æ": "ae", "œ": "oe", "ø": "o", "đ": "d", "ł": "l"}

_log = logging.getLogger("raidho.vsa.memory")


def _normalize_surface(s: str) -> str:
    """Concept identity key by STRING (not by embedding).

    casefold → Scandinavian special letters → NFKD + diacritic removal → collapse
    whitespace. "Zürich"="ZÜRICH "="zurich" → one key. Different alphabets are not
    transliterated (Cyrillic stays Cyrillic) — cross-alphabet identity is defined
    by the alias table, not by guessing."""
    s = s.strip().casefold()
    s = "".join(_SCAND.get(ch, ch) for ch in s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(s.split())


class VSAMemory:
    ROLE_NAMES = ("subject", "relation", "object")

    def __init__(
        self,
        D: int = core.DEFAULT_D,
        embedder_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        seed: int = 0,
        normalize_threshold: float = 0.82,
        embed_fn: Callable[[str], np.ndarray] | None = None,
        identity_mode: str = "string",
        aliases_table: dict[str, str] | None = None,
    ) -> None:
        self.D = D
        self._embedder_model = embedder_model
        self._normalize_threshold = float(normalize_threshold)
        # How entity identity is decided: "string" (normalization + aliases, safe)
        # or "embedding" (legacy: cosine ≥ threshold — breeds false merges).
        self._identity_mode = identity_mode
        self._alias_map = {
            _normalize_surface(k): _normalize_surface(v)
            for k, v in (aliases_table or {}).items()
        }
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._embed_fn = embed_fn
        self._model = None
        self._emb_dim: int | None = None
        self._proj: np.ndarray | None = None

        self._roles = {n: core.random_atoms(1, D, self._rng)[0] for n in self.ROLE_NAMES}

        # Canonical concept codebook. _atoms — float ±1 (needed for bind/bundle/
        # permute); _atom_bits — their bit-pack (for popcount cleanup).
        self._names: list[str] = []
        self._kinds: list[str] = []          # "entity" | "relation" | "event"
        self._atoms: list[np.ndarray] = []
        self._atom_bits: list[np.ndarray] = []
        self._embs: list[np.ndarray] = []
        self._index: dict[str, int] = {}     # surface (lower) → canonical idx
        self._aliases: dict[int, list[str]] = {}

        # Facts and episodes. Facts are stored ONLY bit-packed (_fact_bits) — this
        # is the RAM-dominant layer (×32 savings); float ±1 is reconstructed on
        # demand for top-K unbind (≤5/query). Ranking is identical to the float version.
        self._fact_idx: list[tuple[int, int, int]] = []
        self._fact_bits: list[np.ndarray] = []
        self._fact_meta: list[dict] = []
        self._episodes: dict[str, dict] = {}
        self._ep_counter = 0

        # Procedures (procedural memory). VSA stores the trigger + body and matches
        # the trigger; the body is EXECUTED by an external interpreter (procedure_runner.py).
        self._procedures: dict[str, dict] = {}

        # Constraints. The second kind: NOT executed — the rule/tone is woven into
        # the system prompt when the trigger is active. Trigger:
        # always | predicate | semantic (as for procedures).
        self._constraints: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Embedding / grounding
    # ------------------------------------------------------------------
    def _ensure_proj(self, emb_dim: int) -> None:
        if self._proj is None:
            self._emb_dim = emb_dim
            self._proj = core.make_projection(emb_dim, self.D, self._rng)

    def _embed(self, text: str) -> np.ndarray:
        if self._embed_fn is not None:
            e = np.asarray(self._embed_fn(text), dtype=np.float32)
            n = np.linalg.norm(e)
            e = e / n if n > 0 else e
        else:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._embedder_model)
            e = self._model.encode(text, normalize_embeddings=True).astype(np.float32)
        self._ensure_proj(e.shape[0])
        return e

    # ------------------------------------------------------------------
    # Codebook with entity normalization
    # ------------------------------------------------------------------
    def _concept_index(self, concept: str, kind: str) -> int:
        """Canonical index of a concept.

        Entity identity is decided by STRING (normalization + alias table), not by
        embedding: cosine encodes semantic closeness, not referent identity
        (e.g. "Paris"↔"France"=0.84 > "Paris"↔"Texas"=0.64 — a threshold is
        helpless). The embedding is kept only for recall (search/grounding).
        Old behavior — identity_mode='embedding'. Bias: when unsure → a NEW concept
        (a duplicate is harmless, a false merge silently corrupts facts)."""
        key = _normalize_surface(concept)
        key = self._alias_map.get(key, key)          # declarative aliases
        if key in self._index:
            return self._index[key]
        emb = self._embed(concept)

        # Legacy: identity by embedding. Off by default; NEVER for events
        # (utterance-episodes are not concepts to dedup).
        if self._identity_mode == "embedding" and kind != "event":
            same = [i for i, k in enumerate(self._kinds) if k == kind]
            if same:
                E = np.stack([self._embs[i] for i in same])
                sims = E @ emb
                j = int(np.argmax(sims))
                if float(sims[j]) >= self._normalize_threshold:
                    canon = same[j]
                    self._index[key] = canon
                    if key != _normalize_surface(self._names[canon]):
                        self._aliases.setdefault(canon, []).append(concept)
                    return canon

        idx = len(self._names)
        self._names.append(concept)
        self._kinds.append(kind)
        atom = core.ground(emb, self._proj)
        self._atoms.append(atom)
        self._atom_bits.append(core.pack_bipolar(atom))
        self._embs.append(emb)
        self._index[key] = idx
        return idx

    def _cleanup(self, vec: np.ndarray, kind: str) -> tuple[str, float, int]:
        idxs = [i for i, k in enumerate(self._kinds) if k == kind]
        if not idxs:
            return "", 0.0, -1
        cb = np.stack([self._atom_bits[i] for i in idxs])
        sims = core.hamming_cosine(cb, core.pack_bipolar(vec), self.D)
        b = int(np.argmax(sims))
        return self._names[idxs[b]], float(sims[b]), idxs[b]

    # ------------------------------------------------------------------
    # Facts
    # ------------------------------------------------------------------
    def add_triple(self, subject: str, relation: str, obj: str, meta: dict | None = None) -> int:
        si = self._concept_index(subject, "entity")
        ri = self._concept_index(relation, "relation")
        oi = self._concept_index(obj, "entity")
        if (si, ri, oi) in self._fact_idx:           # dedup: don't duplicate the same triple
            return self._fact_idx.index((si, ri, oi))
        fact = core.bundle(np.stack([
            core.bind(self._roles["subject"], self._atoms[si]),
            core.bind(self._roles["relation"], self._atoms[ri]),
            core.bind(self._roles["object"], self._atoms[oi]),
        ]), self._rng)
        self._fact_idx.append((si, ri, oi))
        self._fact_bits.append(core.pack_bipolar(fact))   # store only the packed form
        self._fact_meta.append(meta or {})
        return len(self._fact_bits) - 1

    def query(self, known: dict[str, str], target_role: str) -> dict:
        """known: {role: concept}; returns the reconstructed concept for target_role
        + the retrieved triple. Distinguishes (X,r,Y) from (Y,r,X) by ROLES."""
        if not self._fact_bits:
            return {"answer": None, "score": 0.0, "triple": None, "fact_idx": -1}
        terms = []
        for role, concept in known.items():
            kind = "relation" if role == "relation" else "entity"
            ci = self._concept_index(concept, kind)
            terms.append(core.bind(self._roles[role], self._atoms[ci]))
        probe = core.bundle(np.stack(terms), self._rng)

        # Backtracking: check the top-K nearest facts. Similarity — popcount over
        # the packed facts (= (mem @ probe)/D on ±1, ranking identical).
        sims_facts = core.hamming_cosine(
            np.stack(self._fact_bits), core.pack_bipolar(probe), self.D)
        # Quarantined facts do not take part in structural recall (mask them out).
        for i in range(len(sims_facts)):
            if self._is_quarantined(i):
                sims_facts[i] = -np.inf
        top_k = min(5, len(self._fact_bits))
        best_fidxs = np.argsort(-sims_facts)[:top_k]

        kind = "relation" if target_role == "relation" else "entity"

        best_result = {"answer": None, "score": -1.0, "triple": None, "fact_idx": -1, "_mq": -1.0}

        for fidx in best_fidxs:
            fidx = int(fidx)
            if not np.isfinite(sims_facts[fidx]):   # quarantined — skip
                continue
            fact_vec = core.unpack_bipolar(self._fact_bits[fidx], self.D)  # ±1 from packing
            unbound = core.unbind(fact_vec, self._roles[target_role])
            name, score, _ = self._cleanup(unbound, kind)

            # Account for both fact-to-query relevance and extraction cleanliness.
            # Clamp negatives: an anti-correlated fact (sims<0) or a dirty
            # extraction (score<0) is not a "success". Without the clamp neg×neg
            # would yield a falsely high match_quality and could beat an honest match.
            match_quality = max(0.0, float(sims_facts[fidx])) * max(0.0, score)

            if match_quality > best_result["_mq"]:
                si, ri, oi = self._fact_idx[fidx]
                best_result = {
                    "answer": name,
                    "score": score,
                    "triple": (self._names[si], self._names[ri], self._names[oi]),
                    "fact_idx": fidx,
                    "_mq": match_quality
                }

            # For a 3-term bundle the expected score is ≈ 0.5. If we extracted with
            # score > 0.35 from a relevant fact — that's a success, stop searching.
            if score > 0.35 and sims_facts[fidx] > 0.2:
                break

        best_result.pop("_mq", None)
        return best_result

    def search(self, query: str, top_k: int = 8, include_quarantined: bool = False) -> list[dict]:
        """Similarity recall over facts (free-form query): query embedding vs
        fact embedding (= normalized sum of its concept embeddings).
        Returns [{'triple': (s,r,o), 'score': cos, 'fact_idx': i, 'quarantined': bool}],
        sorted descending. This is the "ANN layer underneath" — it complements the
        structural query().

        Quarantined facts ("forget about this") do NOT surface by default;
        include_quarantined=True is only needed by the restore/listing command."""
        if not self._fact_idx:
            return []
        active = [
            i for i in range(len(self._fact_idx))
            if include_quarantined or not self._is_quarantined(i)
        ]
        if not active:
            return []
        qe = self._embed(query)
        rows = np.stack([
            self._embs[si] + self._embs[ri] + self._embs[oi]
            for (si, ri, oi) in (self._fact_idx[i] for i in active)
        ])
        rows /= np.linalg.norm(rows, axis=1, keepdims=True)
        sims = rows @ qe
        order = np.argsort(-sims)[:top_k]
        return [
            {
                "triple": (
                    self._names[self._fact_idx[active[o]][0]],
                    self._names[self._fact_idx[active[o]][1]],
                    self._names[self._fact_idx[active[o]][2]],
                ),
                "score": float(sims[o]),
                "fact_idx": active[o],
                "quarantined": self._is_quarantined(active[o]),
            }
            for o in order
        ]

    # ------------------------------------------------------------------
    # Fact quarantine ("forget about this" — soft forgetting without deletion)
    # ------------------------------------------------------------------
    def _is_quarantined(self, i: int) -> bool:
        m = self._fact_meta[i] if 0 <= i < len(self._fact_meta) else None
        return bool(m) and bool(m.get("quarantined"))

    def quarantine(self, fact_indices, reason: str = "") -> int:
        """Mark facts as quarantined (a flag in _fact_meta, survives saving).
        The fact stays in memory but does not surface in recall. Returns the count of new ones."""
        n = 0
        for i in fact_indices:
            if 0 <= i < len(self._fact_meta):
                meta = dict(self._fact_meta[i] or {})
                if not meta.get("quarantined"):
                    n += 1
                meta.update(quarantined=True, quarantine_reason=reason,
                            quarantine_ts=time.time())
                self._fact_meta[i] = meta
        return n

    def unquarantine(self, fact_indices) -> int:
        """Lift quarantine — the fact surfaces in recall again. Returns the count lifted."""
        n = 0
        for i in fact_indices:
            if 0 <= i < len(self._fact_meta) and (self._fact_meta[i] or {}).get("quarantined"):
                meta = dict(self._fact_meta[i])
                for k in ("quarantined", "quarantine_reason", "quarantine_ts"):
                    meta.pop(k, None)
                self._fact_meta[i] = meta
                n += 1
        return n

    def quarantined(self) -> list[dict]:
        """Everything currently quarantined: [{'fact_idx','triple','reason','ts'}]."""
        out = []
        for i, m in enumerate(self._fact_meta):
            if m and m.get("quarantined"):
                si, ri, oi = self._fact_idx[i]
                out.append({
                    "fact_idx": i,
                    "triple": (self._names[si], self._names[ri], self._names[oi]),
                    "reason": m.get("quarantine_reason", ""),
                    "ts": m.get("quarantine_ts"),
                })
        return out

    # ------------------------------------------------------------------
    # Episodes (order via permutation)
    # ------------------------------------------------------------------
    def add_episode(self, items: list[str], episode_id: str | None = None) -> str:
        idxs = [self._concept_index(it, "event") for it in items]
        vec = core.bundle(
            np.stack([core.permute(self._atoms[i], pos) for pos, i in enumerate(idxs)]),
            self._rng,
        )
        eid = episode_id or f"ep{self._ep_counter}"
        self._ep_counter += 1
        self._episodes[eid] = {"item_idx": idxs, "vec": vec}
        return eid

    def recall_at(self, episode_id: str, pos: int) -> str:
        ep = self._episodes[episode_id]
        if pos < 0 or pos >= len(ep["item_idx"]):
            return ""
        return self._cleanup(core.unpermute(ep["vec"], pos), "event")[0]

    def episode_order(self, episode_id: str) -> list[str]:
        ep = self._episodes[episode_id]
        return [self.recall_at(episode_id, p) for p in range(len(ep["item_idx"]))]

    def episode_items(self, episode_id: str) -> list[str]:
        """Exact list of episode items (from the codebook, no lossy recall)."""
        ep = self._episodes.get(episode_id)
        if not ep:
            return []
        return [self._names[i] for i in ep["item_idx"]]

    def successor(self, episode_id: str, item: str) -> str | None:
        ep = self._episodes[episode_id]
        n = len(ep["item_idx"])
        a = self._atoms[self._concept_index(item, "event")]
        sims = [(core.unpermute(ep["vec"], p) @ a) / self.D for p in range(n)]
        pos = int(np.argmax(sims))
        return self.recall_at(episode_id, pos + 1) if pos + 1 < n else None

    # ------------------------------------------------------------------
    # Procedures (procedural memory)
    #
    # The VSA role is SEARCH, not execution: it stores a procedure and matches
    # "which one fits the current situation". The body is a structured program
    # (opcodes/args/branches/registers) — it lives as a dict, NOT in the
    # hypervector (a branch and a runtime arg cannot be encoded by permutation-
    # bundle). The body is executed by a separate interpreter (procedure_runner.py).
    #
    # Two kinds of trigger:
    #   predicate — exact pattern (regex over the context text), no VSA;
    #   semantic  — fuzzy match against EXAMPLE ANCHORS: each anchor is placed in
    #               the codebook as a concept kind="trigger" (persisted), score =
    #               the MAXIMUM cosine of the context to any anchor.
    #
    # Why example anchors rather than one description: on live text a short remark
    # ("this class needs reworking") is an EXAMPLE of the situation, not a
    # paraphrase of an abstract description, and cosine to the description drops to
    # ~0. Several live anchors sharply raise recall (prototype matching). The
    # trigger gives either "examples": [...] (preferred) or "situation": <text>
    # (treated as a single anchor — backward compatibility).
    # ------------------------------------------------------------------
    @staticmethod
    def _trigger_anchors(trigger: dict) -> list[str]:
        ex = trigger.get("examples")
        if ex:
            return list(ex)
        sit = trigger.get("situation")
        return [sit] if sit else []

    def add_procedure(self, proc_id: str, trigger: dict, body: dict,
                      meta: dict | None = None) -> str:
        """Store a procedure. trigger = {"type":"predicate","pattern":<regex>}
        or {"type":"semantic","examples":[...]} (or legacy "situation":<text>).
        The body is stored as is. Returns proc_id."""
        ttype = trigger.get("type")
        trigger_idxs: list[int] = []
        if ttype == "semantic":
            anchors = self._trigger_anchors(trigger)
            if not anchors:
                raise ValueError("semantic trigger: needs 'examples' or 'situation'")
            trigger_idxs = [self._concept_index(a, "trigger") for a in anchors]
        elif ttype == "predicate":
            re.compile(trigger["pattern"])  # broken pattern — fail now, not at runtime
        else:
            raise ValueError(f"unknown trigger type: {ttype!r}")
        self._procedures[proc_id] = {
            "trigger": trigger,
            "trigger_idxs": trigger_idxs,
            "body": body,
            "meta": meta or {},
        }
        for w in self.lint_procedure(body):  # proofreading: warning, not rejection
            _log.warning("procedure %s: %s", proc_id, w)
        return proc_id

    def match_trigger(self, context: str, threshold: float = 0.45,
                      top_k: int = 3, use_fitness: bool = False,
                      mode_boosts: dict[str, float] | None = None) -> list[dict]:
        """Which procedures fit the context. Predicates — regex (score 1.0);
        semantic — MAX cosine of the context embedding to the trigger anchors
        (score), cut off by threshold. Returns [{'proc_id','score','type'}] by
        descending score.

        Quarantined procedures (meta['quarantined']) are NOT matched.

        use_fitness — homeostasis (negative feedback): the final score is
        multiplied by (0.5 + procedure_fitness) ∈ [0.5, 1.5]. A neutral fitness
        (no outcomes → 0.5) gives a factor of 1.0 — ordering is unchanged until
        VERIFIED outcomes accumulate (record_outcome). Thus proven-successful
        procedures rise, failed ones sink — without editing triggers.

        mode_boosts — allostery/epigenetics: {mode: boost}. If meta['modes'] of
        the procedure intersects with the active modes, score *= (1 + boost). A
        weak long-term context modifier (e.g. {'ai-dev': 0.3})."""
        hits: list[dict] = []
        sem = []
        for pid, p in self._procedures.items():
            if (p.get("meta") or {}).get("quarantined"):
                continue
            t = p["trigger"]
            if t.get("type") == "predicate":
                if re.search(t["pattern"], context):
                    hits.append({"proc_id": pid, "score": 1.0, "type": "predicate"})
            elif t.get("type") == "semantic":
                sem.append((pid, p))
        if sem:
            qe = self._embed(context)
            for pid, p in sem:
                # backward compatibility: old key trigger_idx (single) → list
                idxs = p.get("trigger_idxs")
                if idxs is None:
                    one = p.get("trigger_idx")
                    idxs = [one] if one is not None else []
                if not idxs:
                    continue
                score = max(float(self._embs[i] @ qe) for i in idxs)  # max over anchors
                if score >= threshold:
                    hits.append({"proc_id": pid, "score": score, "type": "semantic"})
        if use_fitness or mode_boosts:
            for h in hits:
                h["base_score"] = h["score"]
                meta = self._procedures[h["proc_id"]].get("meta") or {}
                if use_fitness:
                    h["fitness"] = self.procedure_fitness(h["proc_id"])
                    h["score"] *= 0.5 + h["fitness"]
                if mode_boosts:
                    for m in (meta.get("modes") or []):
                        if m in mode_boosts:
                            h["score"] *= 1.0 + mode_boosts[m]
        hits.sort(key=lambda h: -h["score"])
        return hits[:top_k]

    def get_procedure(self, proc_id: str) -> dict | None:
        """Procedure body + meta (for the interpreter). None if missing."""
        p = self._procedures.get(proc_id)
        if not p:
            return None
        return {"id": proc_id, "trigger": p["trigger"],
                "body": p["body"], "meta": p["meta"]}

    @property
    def procedures(self) -> list[str]:
        return list(self._procedures.keys())

    # ------------------------------------------------------------------
    # Procedure homeostasis: outcome → fitness → ranking
    #
    # Biological motivation — negative feedback: the system adjusts the
    # "activity" (selection probability) of a procedure based on its real
    # OUTCOMES. KEY (see process well): an outcome is only VERIFIED (test
    # passed / human confirmed / procedure objectively crashed), NOT model
    # self-assessment — otherwise self-reinforcing drift. Stored in the
    # procedure's meta (persists with it).
    # ------------------------------------------------------------------
    def record_outcome(self, proc_id: str, success: bool) -> dict:
        """Record a VERIFIED outcome of a procedure execution.
        success=True — confirmed success (/ok, test); False — objective
        failure (exception during execution). Accumulates counters in
        meta['fitness']. Returns {'success','failure','fitness'}."""
        p = self._procedures.get(proc_id)
        if p is None:
            raise KeyError(f"no procedure {proc_id!r}")
        meta = dict(p.get("meta") or {})
        f = dict(meta.get("fitness") or {"success": 0, "failure": 0})
        f["success" if success else "failure"] += 1
        meta["fitness"] = f
        p["meta"] = meta
        return {**f, "fitness": self.procedure_fitness(proc_id)}

    def procedure_fitness(self, proc_id: str) -> float:
        """Procedure fitness ∈ (0,1) based on verified outcomes.
        Beta-mean with Laplace smoothing: (s+1)/(s+f+2). No outcomes → 0.5
        (neutral, ranking multiplier 1.0). One failure does not zero out —
        it sinks gradually."""
        p = self._procedures.get(proc_id)
        if p is None:
            return 0.5
        f = (p.get("meta") or {}).get("fitness") or {}
        s, fl = int(f.get("success", 0)), int(f.get("failure", 0))
        return (s + 1) / (s + fl + 2)

    # ------------------------------------------------------------------
    # Procedure quarantine (ubiquitin-proteasome): weak → NOT executed.
    # Reversible (like fact quarantine) — not a delete, to preserve the recipe.
    # ------------------------------------------------------------------
    def quarantine_procedure(self, proc_id: str, reason: str = "") -> bool:
        """Mark a procedure as quarantined — it stops matching (match_trigger
        skips it). Returns True if the state changed."""
        p = self._procedures.get(proc_id)
        if p is None:
            return False
        meta = dict(p.get("meta") or {})
        was = bool(meta.get("quarantined"))
        meta.update(quarantined=True, quarantine_reason=reason, quarantine_ts=time.time())
        p["meta"] = meta
        return not was

    def unquarantine_procedure(self, proc_id: str) -> bool:
        """Lift quarantine — the procedure matches again."""
        p = self._procedures.get(proc_id)
        if p is None or not (p.get("meta") or {}).get("quarantined"):
            return False
        meta = dict(p["meta"])
        for k in ("quarantined", "quarantine_reason", "quarantine_ts"):
            meta.pop(k, None)
        p["meta"] = meta
        return True

    def prune_weak_procedures(self, min_fitness: float = 0.3,
                              min_samples: int = 3) -> list[dict]:
        """Quarantine procedures with LOW verified fitness. Only when
        sufficient statistics have accumulated (success+failure ≥ min_samples)
        — don't execute for a single failure. Does NOT delete (reversible).
        Called manually/periodically, NOT automatically at runtime (no
        autonomy without a human). Returns [{'proc_id','fitness','samples'}]
        for the quarantined ones."""
        pruned = []
        for pid, p in self._procedures.items():
            if (p.get("meta") or {}).get("quarantined"):
                continue
            f = (p.get("meta") or {}).get("fitness") or {}
            samples = int(f.get("success", 0)) + int(f.get("failure", 0))
            fit = self.procedure_fitness(pid)
            if samples >= min_samples and fit < min_fitness:
                self.quarantine_procedure(pid, reason=f"low fitness {fit:.2f} ({samples} outcomes)")
                pruned.append({"proc_id": pid, "fitness": fit, "samples": samples})
        return pruned

    @staticmethod
    def lint_procedure(body: dict) -> list[str]:
        """Static proofreading of a procedure body (no reference needed —
        relies on STRUCTURE, not on cosine-drift of output). Currently one
        rule: a generative step (transform/execute mode=generative with an
        out-register) whose result is never consumed downstream by a validate
        step — a candidate for an unverified hallucination. Returns a list of
        warnings (NOT errors): the procedure is valid, but its reliability is
        questionable."""
        steps = body.get("steps", [])
        validated_regs: set[str] = set()
        for s in steps:
            args = s.get("args") or {}
            for v in args.values():
                if isinstance(v, str) and v.startswith("$"):
                    validated_regs.add(v[1:])
                elif isinstance(v, list):
                    validated_regs.update(x[1:] for x in v if isinstance(x, str) and x.startswith("$"))
        warns = []
        for s in steps:
            if s.get("mode") == "generative" and s.get("out"):
                # is there a downstream validate consuming this out?
                consumed_by_validate = any(
                    o.get("op") == "validate" and f"${s['out']}" in str(o.get("args", {}))
                    for o in steps
                )
                if not consumed_by_validate:
                    warns.append(
                        f"step {s.get('id')}: generative→{s['out']} without downstream validate "
                        f"(model output is not verified)")
        return warns

    # ------------------------------------------------------------------
    # Constraints — the second kind of procedural memory
    #
    # A Procedure is EXECUTED (steps). A Constraint is NOT executed — its rule is
    # woven into the system prompt when the trigger is active. It is "how to
    # behave", not "what to do": tone (no-sycophancy), caution
    # (verify-before-claim), protocol (say-back). Trigger:
    #   always    — the rule applies on every turn;
    #   predicate — regex over the context;
    #   semantic  — max cosine to anchors (as for procedures).
    # ------------------------------------------------------------------
    def add_constraint(self, cid: str, trigger: dict, rule: str,
                       meta: dict | None = None) -> str:
        """Store a constraint. trigger = {"type":"always"} |
        {"type":"predicate","pattern":..} | {"type":"semantic","examples":[..]}.
        rule — the text woven into the system prompt. Returns cid."""
        ttype = trigger.get("type")
        trigger_idxs: list[int] = []
        if ttype == "always":
            pass
        elif ttype == "semantic":
            anchors = self._trigger_anchors(trigger)
            if not anchors:
                raise ValueError("semantic constraint: needs 'examples' or 'situation'")
            trigger_idxs = [self._concept_index(a, "trigger") for a in anchors]
        elif ttype == "predicate":
            re.compile(trigger["pattern"])
        else:
            raise ValueError(f"unknown constraint trigger type: {ttype!r}")
        self._constraints[cid] = {
            "trigger": trigger,
            "trigger_idxs": trigger_idxs,
            "rule": rule,
            "meta": meta or {},
        }
        return cid

    def match_constraints(self, context: str, threshold: float = 0.45,
                          mode_boosts: dict[str, float] | None = None) -> list[dict]:
        """Constraints active for the current context: always (score 1.0) +
        predicate (regex) + semantic (max cosine ≥ threshold).
        Returns [{'id','rule','score','type'}] by descending score.

        mode_boosts — allostery: {mode: boost}. A constraint with meta['modes']
        intersecting the active modes gets its score raised (*= 1+boost). Thus in
        an "irritated" mode no-sycophancy/say-back rise above others and reliably
        enter the prompt (ordering and compliance threshold)."""
        out: list[dict] = []
        sem = []
        for cid, c in self._constraints.items():
            t = c["trigger"]
            tt = t.get("type")
            if tt == "always":
                out.append({"id": cid, "rule": c["rule"], "score": 1.0, "type": "always"})
            elif tt == "predicate":
                if re.search(t["pattern"], context):
                    out.append({"id": cid, "rule": c["rule"], "score": 1.0, "type": "predicate"})
            elif tt == "semantic":
                sem.append((cid, c))
        if sem:
            qe = self._embed(context)
            for cid, c in sem:
                idxs = c.get("trigger_idxs") or []
                if not idxs:
                    continue
                score = max(float(self._embs[i] @ qe) for i in idxs)
                if score >= threshold:
                    out.append({"id": cid, "rule": c["rule"], "score": score, "type": "semantic"})
        if mode_boosts:
            for h in out:
                for m in ((self._constraints[h["id"]].get("meta") or {}).get("modes") or []):
                    if m in mode_boosts:
                        h["score"] *= 1.0 + mode_boosts[m]
        out.sort(key=lambda h: -h["score"])
        return out

    @property
    def constraints(self) -> list[str]:
        return list(self._constraints.keys())

    # ------------------------------------------------------------------
    # Persistence (memory survives sessions)
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        # proj — a fixed emb_dim×D matrix (~15 MB, gaussian → compresses poorly).
        # It used to be written into the .npz on EVERY save (i.e. every bot reply) —
        # heavy I/O for nothing, since proj is immutable after the first embedding.
        # Now it is written ONCE to a sidecar .proj.npy; the main .npz holds only the
        # light roles/embs (~hundreds of KB). Loading stays backward-compatible:
        # an old .npz with the proj key inside still reads as before (see load).
        if self._proj is not None and self._proj.size > 0:
            proj_path = base.with_suffix(".proj.npy")
            need_write = True
            if proj_path.exists():
                try:
                    if tuple(np.load(proj_path, mmap_mode="r").shape) == tuple(self._proj.shape):
                        need_write = False
                except Exception:
                    need_write = True
            if need_write:
                np.save(proj_path, self._proj)
        np.savez_compressed(
            base.with_suffix(".npz"),
            roles=np.stack([self._roles[n] for n in self.ROLE_NAMES]).astype(np.int8),
            embs=(np.stack(self._embs) if self._embs
                  else np.zeros((0, self._emb_dim or 1), np.float32)),
        )
        meta = {
            "D": self.D, "seed": self._seed, "emb_dim": self._emb_dim,
            "embedder_model": self._embedder_model,
            "normalize_threshold": self._normalize_threshold,
            "identity_mode": self._identity_mode,
            "alias_map": self._alias_map,
            "role_names": list(self.ROLE_NAMES),
            "names": self._names, "kinds": self._kinds,
            "index": self._index,
            "aliases": {str(k): v for k, v in self._aliases.items()},
            "fact_idx": self._fact_idx, "fact_meta": self._fact_meta,
            "episodes": {k: v["item_idx"] for k, v in self._episodes.items()},
            "ep_counter": self._ep_counter,
            # Procedures are fully JSON-safe (trigger/trigger_idxs/body/meta) — the
            # body is not in the hypervector, so it saves and loads as is.
            "procedures": self._procedures,
            "constraints": self._constraints,  # constraints are also fully JSON-safe
        }
        base.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path, embed_fn: Callable[[str], np.ndarray] | None = None) -> "VSAMemory":
        base = Path(path)
        arr = np.load(base.with_suffix(".npz"))
        meta = json.loads(base.with_suffix(".json").read_text())

        m = cls(D=meta["D"], embedder_model=meta.get("embedder_model", "paraphrase-multilingual-MiniLM-L12-v2"),
                seed=meta["seed"], normalize_threshold=meta["normalize_threshold"],
                embed_fn=embed_fn,
                identity_mode=meta.get("identity_mode", "string"))
        m._alias_map = dict(meta.get("alias_map", {}))
        m._emb_dim = meta["emb_dim"]
        # proj: new format — sidecar .proj.npy; old — proj key inside the .npz.
        proj_path = base.with_suffix(".proj.npy")
        if proj_path.exists():
            m._proj = np.load(proj_path)
        elif "proj" in arr.files:
            m._proj = arr["proj"]
        else:
            m._proj = None
        if m._proj is not None and m._proj.size == 0:
            m._proj = None  # empty old-format placeholder → None (as on init)
        roles = arr["roles"].astype(np.float32)
        m._roles = {n: roles[i] for i, n in enumerate(meta["role_names"])}
        m._names = list(meta["names"])
        m._kinds = list(meta["kinds"])
        m._index = {k: int(v) for k, v in meta["index"].items()}
        m._aliases = {int(k): v for k, v in meta["aliases"].items()}
        embs = arr["embs"]
        m._embs = [embs[i] for i in range(len(m._names))]
        m._atoms = [core.ground(e, m._proj) for e in m._embs]  # deterministic
        m._atom_bits = [core.pack_bipolar(a) for a in m._atoms]

        m._fact_idx = [tuple(t) for t in meta["fact_idx"]]
        m._fact_meta = meta["fact_meta"]
        m._fact_bits = [
            core.pack_bipolar(core.bundle(np.stack([
                core.bind(m._roles["subject"], m._atoms[si]),
                core.bind(m._roles["relation"], m._atoms[ri]),
                core.bind(m._roles["object"], m._atoms[oi]),
            ]), m._rng))
            for (si, ri, oi) in m._fact_idx
        ]
        m._episodes = {}
        for eid, idxs in meta["episodes"].items():
            idxs = list(idxs)
            vec = core.bundle(
                np.stack([core.permute(m._atoms[i], pos) for pos, i in enumerate(idxs)]),
                m._rng,
            )
            m._episodes[eid] = {"item_idx": idxs, "vec": vec}
        m._ep_counter = meta["ep_counter"]
        # Procedures: .get for backward compatibility with old snapshots lacking the key.
        m._procedures = meta.get("procedures", {})
        m._constraints = meta.get("constraints", {})  # .get for backward compatibility
        return m

    # ------------------------------------------------------------------
    @property
    def n_facts(self) -> int:
        return len(self._fact_bits)

    @property
    def n_concepts(self) -> int:
        return len(self._names)

    @property
    def triples(self) -> list[tuple[str, str, str]]:
        return [(self._names[s], self._names[r], self._names[o]) for s, r, o in self._fact_idx]

    def aliases_of(self, concept: str) -> list[str]:
        key = concept.strip().lower()
        idx = self._index.get(key)
        return self._aliases.get(idx, []) if idx is not None else []

    def __repr__(self) -> str:
        return (f"VSAMemory(D={self.D}, facts={self.n_facts}, "
                f"concepts={self.n_concepts}, episodes={len(self._episodes)})")
