"""Regression test for bit-packed similarity: ranking identical to float, recall holds.

Run: `python tests/test_bitpack.py` or `pytest tests/test_bitpack.py`.
"""
import os
import tempfile

import numpy as np

from vsa import core
from vsa.memory import VSAMemory

D = 10_000
EMB = 64


def _emb_fn():
    rng = np.random.default_rng(1)
    cache = {}

    def emb(t):
        if t not in cache:
            cache[t] = rng.standard_normal(EMB).astype(np.float32)
        return cache[t]

    return emb


def test_identity_popcount_equals_float():
    """popcount-cos == (cb @ vec)/D on ±1, bit-for-bit (same ranking)."""
    rng = np.random.default_rng(7)
    M = np.where(rng.integers(0, 2, (2000, D)) > 0, 1.0, -1.0).astype(np.float32)
    v = np.where(rng.integers(0, 2, D) > 0, 1.0, -1.0).astype(np.float32)
    float_cos = (M @ v) / D
    pk = np.stack([core.pack_bipolar(r) for r in M])
    ham_cos = core.hamming_cosine(pk, core.pack_bipolar(v), D)
    assert np.allclose(float_cos, ham_cos, atol=1e-6)
    assert int(np.argmax(float_cos)) == int(np.argmax(ham_cos))


def test_round_trip_recall():
    m = VSAMemory(D=D, seed=0, embed_fn=_emb_fn())
    triples = [(f"s{i}", f"r{i}", f"o{i}") for i in range(300)]
    for s, r, o in triples:
        m.add_triple(s, r, o)
    ok = sum(m.query({"subject": s, "relation": r}, "object")["answer"] == o
             for s, r, o in triples)
    assert ok == len(triples)


def test_save_load_preserves_recall():
    m = VSAMemory(D=D, seed=0, embed_fn=_emb_fn())
    triples = [(f"s{i}", f"r{i}", f"o{i}") for i in range(300)]
    for s, r, o in triples:
        m.add_triple(s, r, o)
    p = os.path.join(tempfile.mkdtemp(), "mem")
    m.save(p)
    m2 = VSAMemory.load(p, embed_fn=_emb_fn())
    ok = sum(m2.query({"subject": s, "relation": r}, "object")["answer"] == o
             for s, r, o in triples)
    assert ok == len(triples)


def test_storage_is_bitpacked():
    """A fact is stored as uint8 over D/8 bytes (×32 vs float32)."""
    m = VSAMemory(D=D, seed=0, embed_fn=_emb_fn())
    m.add_triple("a", "r", "b")
    fb = m._fact_bits[0]
    assert fb.dtype == np.uint8
    assert fb.nbytes == D // 8


def test_episodes_intact():
    m = VSAMemory(D=D, seed=0, embed_fn=_emb_fn())
    eid = m.add_episode(["a", "b", "c", "d"])
    assert m.recall_at(eid, 2) == "c"
    assert m.successor(eid, "b") == "c"


if __name__ == "__main__":
    test_identity_popcount_equals_float()
    print("1) identity popcount==float: OK")
    test_round_trip_recall()
    print("2) round-trip recall: OK")
    test_save_load_preserves_recall()
    print("3) save/load recall: OK")
    test_storage_is_bitpacked()
    print("4) storage bit-packed ×32: OK")
    test_episodes_intact()
    print("5) episodes: OK")
    print("\nALL CHECKS PASSED")
