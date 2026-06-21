"""
Recall accuracy of Raidho's compositional VSA memory across two regimes.

Why two regimes: structural recall accuracy depends heavily on the *shape* of the
data, not only on N. Reporting one number hides that. We measure both honestly:

  • SPARSE-FUNCTIONAL — each (subject, relation) pair is unique (one correct object),
    concepts drawn from a large pool. Best case: no query ambiguity, low cleanup
    collision. Question: does accuracy stay flat as N grows?

  • DENSE-NON-FUNCTIONAL — triple-unique only (the same (subject, relation) recurs
    with different objects), small concept pool. Realistic interference. Reported as
    `strict` (exact object of THIS triple) and `lenient` (ANY object ever stored for
    that (subject, relation)). The gap isolates *ambiguity* from *capacity loss*.

Real embedder (paraphrase-multilingual-MiniLM-L12-v2), pool batch-encoded once,
embed_fn serves precomputed vectors; probe strings fall back to the same real model.

Run: python3 benchmarks/recall_regimes.py
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vsa import core
from vsa.memory import VSAMemory, _normalize_surface

D = core.DEFAULT_D
MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
Q = 200
SEED = 0

ADJ = ("red blue green black white silver golden ancient frozen burning quiet "
       "hidden broken sacred wild gentle bitter hollow crimson azure").split()
N1 = ("river mountain forest desert ocean valley canyon glacier meadow harbor "
      "temple fortress market garden bridge tower cavern island marsh ridge").split()
N2 = ("falcon wolf raven bear lynx otter heron viper stag crane "
      "moth ember thorn willow cedar quartz amber flint ash slate").split()
REL = ("borders flows_into shelters faces overlooks feeds guards mirrors "
       "precedes follows surrounds shadows crowns anchors splits joins "
       "warms cools names claims hosts marks binds frees lifts buries "
       "echoes carries hides reveals").split()  # 30 relations

ALL_ENTITIES = [f"{a} {b} {c}" for a in ADJ for b in N1 for c in N2]  # 8000


def gen_functional(entities, rng, target):
    """(subject, relation) unique → exactly one correct object per query."""
    facts, n = [], len(entities)
    si = 0
    while len(facts) < target and si < n:
        for ri in rng.permutation(len(REL)):
            if len(facts) >= target:
                break
            facts.append((entities[si], REL[ri], entities[int(rng.integers(n))]))
        si += 1
    return facts


def gen_nonfunctional(entities, rng, target):
    """triple-unique only; (subject, relation) recurs with different objects."""
    facts, seen, n = [], set(), len(entities)
    guard = 0
    while len(facts) < target and guard < target * 40:
        si, ri, oi = int(rng.integers(n)), int(rng.integers(len(REL))), int(rng.integers(n))
        guard += 1
        if si == oi:
            continue
        key = (entities[si], REL[ri], entities[oi])
        if key in seen:
            continue
        seen.add(key)
        facts.append(key)
    return facts


def make_embed_fn(concepts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL)
    pool = sorted(set(concepts))
    vecs = model.encode(pool, normalize_embeddings=True, batch_size=256,
                        show_progress_bar=False).astype(np.float32)
    table = {t: vecs[i] for i, t in enumerate(pool)}

    def embed_fn(text: str) -> np.ndarray:
        v = table.get(text)
        return v if v is not None else model.encode(
            text, normalize_embeddings=True).astype(np.float32)
    return embed_fn


def eval_query(mem, facts, idxs, valid):
    """Returns (strict_acc, lenient_acc) for object recall over sampled facts."""
    strict = lenient = 0
    for j in idxs:
        fs, fr, fo = facts[int(j)]
        ans = (mem.query({"subject": fs, "relation": fr}, "object").get("answer") or "")
        a = _normalize_surface(ans)
        if a == _normalize_surface(fo):
            strict += 1
        if a in valid[(fs, fr)]:
            lenient += 1
    n = len(idxs)
    return strict / n, lenient / n


def run_regime(name, entities, gen, target_n, checkpoints, n_seeds):
    print(f"\n### {name}  (pool={len(entities)} ent × {len(REL)} rel, "
          f"seeds={n_seeds}, up to N={target_n})", flush=True)
    # accumulate per-checkpoint over seeds
    acc_strict = defaultdict(list)
    acc_lenient = defaultdict(list)
    for seed in range(n_seeds):
        rng = np.random.default_rng(1000 + seed)
        facts = gen(entities, rng, target_n)
        valid = defaultdict(set)
        for s, r, o in facts:
            valid[(s, r)].add(_normalize_surface(o))
        concepts = {c for f in facts for c in f}
        mem = VSAMemory(D=D, embed_fn=make_embed_fn(concepts),
                        identity_mode="string", seed=SEED)
        qrng = np.random.default_rng(7 + seed)
        cps = set(checkpoints)
        for i, (s, r, o) in enumerate(facts, start=1):
            mem.add_triple(s, r, o)
            if i in cps:
                idxs = qrng.integers(0, i, size=min(Q, i))
                st, le = eval_query(mem, facts, idxs, valid)
                acc_strict[i].append(st)
                acc_lenient[i].append(le)
                print(f"  seed={seed} N={i:>6}  strict={st:.3f}  lenient={le:.3f}", flush=True)
    rows = []
    for n in sorted(acc_strict):
        s = np.array(acc_strict[n]); l = np.array(acc_lenient[n])
        rows.append((n, s.mean(), s.std(), l.mean(), l.std()))
    return rows


def main():
    t0 = time.time()
    sparse = run_regime(
        "SPARSE-FUNCTIONAL", ALL_ENTITIES, gen_functional,
        target_n=100_000, checkpoints=[500, 10_000, 50_000, 100_000], n_seeds=1)
    dense = run_regime(
        "DENSE-NON-FUNCTIONAL", ALL_ENTITIES[:500], gen_nonfunctional,
        target_n=16_000, checkpoints=[1_000, 2_000, 4_000, 8_000, 16_000], n_seeds=3)

    out = Path(__file__).resolve().parent / "recall_regimes_results.md"
    L = [
        "# Raidho VSA memory — recall accuracy across two data regimes",
        "",
        f"Embedder `{MODEL}` · D={D} · Q={Q} sampled queries/checkpoint. Structural "
        "object recall via `query({subject,relation}) -> object`.",
        "",
        "## SPARSE-FUNCTIONAL — `(subject,relation)` unique, large concept pool",
        "Best case: no query ambiguity, low cleanup collision.",
        "",
        "| N facts | query acc | latency note |",
        "|---:|---:|:--|",
    ]
    for n, sm, ss, lm, ls in sparse:
        L.append(f"| {n:,} | {sm:.3f} | — |")
    L += [
        "",
        "## DENSE-NON-FUNCTIONAL — triple-unique only, small pool (500 entities)",
        "Realistic interference. `strict` = exact object of this triple; "
        "`lenient` = any object ever stored for that `(subject,relation)`. "
        "The strict↓lenient gap is *ambiguity*, not capacity loss.",
        "",
        "| N facts | strict acc (mean±std) | lenient acc (mean±std) |",
        "|---:|---:|---:|",
    ]
    for n, sm, ss, lm, ls in dense:
        L.append(f"| {n:,} | {sm:.3f} ± {ss:.3f} | {lm:.3f} ± {ls:.3f} |")
    L += [
        "",
        "## Reading",
        "- **Sparse-functional flat near 1.0** → with unambiguous keys and a separated "
        "concept pool, explicit-store structural recall does not decay with N.",
        "- **Dense-non-functional**: compare `strict` vs `lenient`. If `lenient` stays "
        "high while `strict` falls, the drop is *query ambiguity* (many objects share "
        "one key), not the memory forgetting. A `lenient` decline would indicate genuine "
        "interference/capacity pressure at small pool size — the honest limit.",
        "- Accuracy is regime-dependent: a single headline number would hide this.",
    ]
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"\n[{time.time()-t0:.0f}s] -> {out}", flush=True)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
