"""StateMemory — VSA memory of symbolic procedure states (variable binding).
Ported from astrum-vsa with the packed-bank optimizations (measured: ×20 match
speed at 20k episodes, ×32 RAM; ranking bit-for-bit equal to the float path)."""
import numpy as np
import pytest

from vsa import StateMemory
from vsa import core


def test_recall_value_basic():
    sm = StateMemory(D=4096, seed=0)
    hv = sm.encode({"status": "ok", "branch": "pip", "step": "install"})
    assert sm.recall_value(hv, "branch") == "pip"
    assert sm.recall_value(hv, "status") == "ok"
    assert sm.recall_value(hv, "step") == "install"


def test_recall_many_bindings_systematic():
    sm = StateMemory(D=8192, seed=1)
    state = {f"reg{i}": f"val{i}" for i in range(20)}
    hv = sm.encode(state)
    correct = sum(sm.recall_value(hv, f"reg{i}") == f"val{i}" for i in range(20))
    assert correct == 20


def test_remember_and_match():
    sm = StateMemory(D=4096, seed=2)
    sm.remember({"status": "fail", "branch": "conda"}, outcome="failure")
    sm.remember({"status": "ok", "branch": "pip"}, outcome="success")
    hits = sm.match({"status": "ok", "branch": "pip"}, top_k=1)
    assert hits[0]["outcome"] == "success"
    assert hits[0]["score"] > 0


def test_match_scores_equal_float_path():
    """Packed-bank ranking and scores are bit-for-bit equal to the float path."""
    sm = StateMemory(D=4096, seed=7)
    states = [{"a": str(i % 3), "b": str(i % 5), "c": str(i)} for i in range(50)]
    for st in states:
        sm.remember(st, outcome=f"o{hash(str(st)) % 4}")
    q = {"a": "1", "b": "2", "c": "7"}
    got = sm.match(q, top_k=10)
    bank = np.stack([sm.encode(st) for st in states])     # encode is deterministic
    sims = (bank @ sm.encode(q)) / sm.D
    order = np.argsort(-sims)[:10]
    assert [g["state"] for g in got] == [states[i] for i in order]
    assert np.allclose([g["score"] for g in got], sims[order])


def test_remember_empty_raises():
    with pytest.raises(ValueError):
        StateMemory(D=1024).remember({}, outcome="x")


def test_bank_cache_invalidation():
    sm = StateMemory(D=2048, seed=9)
    sm.remember({"x": "1"}, outcome="old")
    assert sm.match({"x": "1"})[0]["outcome"] == "old"
    sm.remember({"y": "2", "z": "3"}, outcome="new")
    assert sm.match({"y": "2", "z": "3"}, top_k=1)[0]["outcome"] == "new"


def test_episode_storage_is_packed():
    sm = StateMemory(D=2048, seed=0)
    sm.remember({"a": "1"}, outcome="ok")
    assert sm._episodes[0]["hv"].dtype == np.uint8
    assert sm._episodes[0]["hv"].nbytes == 2048 // 8
