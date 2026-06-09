"""Regression tests for procedure homeostasis: fitness, quarantine, prune,
lint, and mode_boosts.

Run: `python tests/test_feedback.py` or `pytest tests/test_feedback.py`.
"""
import numpy as np

from vsa.memory import VSAMemory

D = 1000  # small for fast tests
EMB = 32


def _emb_fn():
    """Deterministic embedding for repeatable tests."""
    rng = np.random.default_rng(42)
    cache = {}

    def emb(t):
        if t not in cache:
            v = rng.standard_normal(EMB).astype(np.float32)
            cache[t] = v / (np.linalg.norm(v) + 1e-12)
        return cache[t]

    return emb


def _make_mem():
    return VSAMemory(D=D, seed=42, embed_fn=_emb_fn())


# ------------------------------------------------------------------
# 1. Fitness starts at 0.5 / Beta-mean
# ------------------------------------------------------------------
def test_fitness_starts_neutral():
    """Without outcomes, fitness = 0.5 (neutral, multiplier 1.0)."""
    m = _make_mem()
    m.add_procedure("p1", {"type": "predicate", "pattern": "hello"}, {"steps": []})
    assert m.procedure_fitness("p1") == 0.5


def test_fitness_beta_mean_after_outcomes():
    """Record outcomes → fitness follows Beta-mean with Laplace smoothing."""
    m = _make_mem()
    m.add_procedure("p1", {"type": "predicate", "pattern": "hello"}, {"steps": []})

    # One success: (1+1)/(1+0+2) = 2/3 ≈ 0.666...
    m.record_outcome("p1", True)
    assert abs(m.procedure_fitness("p1") - 2.0 / 3.0) < 1e-9

    # One failure: (1+1)/(1+1+2) = 2/4 = 0.5
    m.record_outcome("p1", False)
    assert abs(m.procedure_fitness("p1") - 0.5) < 1e-9

    # Two more successes: (3+1)/(3+1+2) = 4/6 ≈ 0.666...
    m.record_outcome("p1", True)
    m.record_outcome("p1", True)
    assert abs(m.procedure_fitness("p1") - 4.0 / 6.0) < 1e-9


def test_fitness_stored_in_meta():
    """Fitness counters live in meta['fitness'], not in a separate class attribute."""
    m = _make_mem()
    m.add_procedure("p1", {"type": "predicate", "pattern": "hello"}, {"steps": []})
    m.record_outcome("p1", True)
    m.record_outcome("p1", False)
    f = m._procedures["p1"]["meta"]["fitness"]
    assert f == {"success": 1, "failure": 1}


# ------------------------------------------------------------------
# 2. Quarantine hides from match_trigger
# ------------------------------------------------------------------
def test_quarantine_hides_from_match():
    """A quarantined procedure is skipped by match_trigger."""
    m = _make_mem()
    m.add_procedure("p_vis", {"type": "predicate", "pattern": "test"}, {"steps": []})
    m.add_procedure("p_hid", {"type": "predicate", "pattern": "test"}, {"steps": []})

    # Before quarantine: both match
    hits = m.match_trigger("test context")
    ids = {h["proc_id"] for h in hits}
    assert "p_vis" in ids
    assert "p_hid" in ids

    # Quarantine p_hid
    m.quarantine_procedure("p_hid", reason="testing")
    hits = m.match_trigger("test context")
    ids = {h["proc_id"] for h in hits}
    assert "p_vis" in ids
    assert "p_hid" not in ids


def test_unquarantine_restores_match():
    """unquarantine_procedure makes the procedure match again."""
    m = _make_mem()
    m.add_procedure("p1", {"type": "predicate", "pattern": "test"}, {"steps": []})
    m.quarantine_procedure("p1")
    assert len(m.match_trigger("test")) == 0
    m.unquarantine_procedure("p1")
    assert len(m.match_trigger("test")) == 1


# ------------------------------------------------------------------
# 3. Prune quarantines weak, does NOT delete
# ------------------------------------------------------------------
def test_prune_quarantines_weak_does_not_delete():
    """prune_weak_procedures quarantines, does not delete the procedure."""
    m = _make_mem()
    m.add_procedure("weak", {"type": "predicate", "pattern": "x"}, {"steps": []})
    # 3 failures → fitness = (0+1)/(3+2) = 1/5 = 0.2 < 0.3
    for _ in range(3):
        m.record_outcome("weak", False)

    pruned = m.prune_weak_procedures(min_fitness=0.3, min_samples=3)
    assert len(pruned) == 1
    assert pruned[0]["proc_id"] == "weak"
    assert pruned[0]["fitness"] < 0.3
    assert pruned[0]["samples"] == 3

    # Procedure still exists
    assert m.get_procedure("weak") is not None
    # But is quarantined
    assert m._procedures["weak"]["meta"]["quarantined"] is True


def test_prune_skips_insufficient_samples():
    """Don't quarantine without enough data (min_samples not met)."""
    m = _make_mem()
    m.add_procedure("fresh", {"type": "predicate", "pattern": "x"}, {"steps": []})
    m.record_outcome("fresh", False)  # 1 failure, fitness 1/3 ≈ 0.33 < 0.3? No, 0.33 > 0.3

    # Even if fitness were low, 1 sample < 3
    pruned = m.prune_weak_procedures(min_fitness=0.3, min_samples=3)
    assert len(pruned) == 0


# ------------------------------------------------------------------
# 4. Lint catches generative without validate
# ------------------------------------------------------------------
def test_lint_catches_generative_without_validate():
    """A generative step whose output is never validated → warning."""
    body = {
        "steps": [
            {"id": "s1", "mode": "generative", "out": "answer",
             "args": {"prompt": "..."}},
        ]
    }
    warns = VSAMemory.lint_procedure(body)
    assert len(warns) == 1
    assert "generative" in warns[0]
    assert "validate" in warns[0].lower()


def test_lint_passes_when_validate_present():
    """A generative step followed by a validate step using its output → clean."""
    body = {
        "steps": [
            {"id": "s1", "mode": "generative", "out": "answer",
             "args": {"prompt": "..."}},
            {"id": "s2", "op": "validate",
             "args": {"input": "$answer", "criterion": "..."}},
        ]
    }
    warns = VSAMemory.lint_procedure(body)
    assert len(warns) == 0


def test_lint_validate_consumes_register():
    """A validate step that references the register indirectly (via list) is detected."""
    body = {
        "steps": [
            {"id": "s1", "mode": "generative", "out": "answer",
             "args": {"prompt": "..."}},
            {"id": "s2", "op": "validate",
             "args": {"inputs": ["$answer", "$context"]}},
        ]
    }
    warns = VSAMemory.lint_procedure(body)
    assert len(warns) == 0  # consumed via list → no warning


# ------------------------------------------------------------------
# 5. mode_boosts raises score
# ------------------------------------------------------------------
def test_mode_boosts_raises_score():
    """mode_boosts multiplies score by (1+boost) for matching modes."""
    m = _make_mem()
    m.add_procedure("p_mode", {"type": "predicate", "pattern": "hello"},
                    {"steps": []}, meta={"modes": ["ai-dev"]})

    # Without mode_boosts: score = 1.0
    hits = m.match_trigger("hello world")
    assert hits[0]["score"] == 1.0

    # With mode_boosts: score = 1.0 * (1 + 0.3) = 1.3
    hits = m.match_trigger("hello world", mode_boosts={"ai-dev": 0.3})
    assert hits[0]["score"] == 1.0 * 1.3


def test_mode_boosts_no_match_no_effect():
    """A mode that doesn't match any procedure's modes → no score change."""
    m = _make_mem()
    m.add_procedure("p1", {"type": "predicate", "pattern": "hello"},
                    {"steps": []}, meta={"modes": ["chat"]})

    hits = m.match_trigger("hello", mode_boosts={"ai-dev": 0.5})
    assert hits[0]["score"] == 1.0  # "ai-dev" doesn't match "chat"


def test_use_fitness_adjusts_score():
    """use_fitness=True multiplies base_score by (0.5 + fitness)."""
    m = _make_mem()
    m.add_procedure("p_success", {"type": "predicate", "pattern": "hello"},
                    {"steps": []})

    # Neutral fitness (0.5) → score unchanged
    hits = m.match_trigger("hello", use_fitness=True)
    assert abs(hits[0]["score"] - 1.0) < 1e-9
    assert "fitness" in hits[0]
    assert hits[0]["fitness"] == 0.5

    # After 3 successes: fitness = (3+1)/(3+0+2) = 4/5 = 0.8
    for _ in range(3):
        m.record_outcome("p_success", True)
    hits = m.match_trigger("hello", use_fitness=True)
    expected = 1.0 * (0.5 + 0.8)  # 1.3
    assert abs(hits[0]["score"] - expected) < 1e-9


def test_record_outcome_missing_raises():
    """record_outcome on a nonexistent procedure raises KeyError."""
    m = _make_mem()
    try:
        m.record_outcome("no_such", True)
        assert False, "should have raised"
    except KeyError:
        pass


if __name__ == "__main__":
    # 1. Fitness
    test_fitness_starts_neutral()
    print("1a) fitness starts at 0.5: OK")
    test_fitness_beta_mean_after_outcomes()
    print("1b) Beta-mean after outcomes: OK")
    test_fitness_stored_in_meta()
    print("1c) fitness stored in meta: OK")

    # 2. Quarantine
    test_quarantine_hides_from_match()
    print("2a) quarantine hides from match_trigger: OK")
    test_unquarantine_restores_match()
    print("2b) unquarantine restores match: OK")

    # 3. Prune
    test_prune_quarantines_weak_does_not_delete()
    print("3a) prune quarantines weak, does not delete: OK")
    test_prune_skips_insufficient_samples()
    print("3b) prune skips insufficient samples: OK")

    # 4. Lint
    test_lint_catches_generative_without_validate()
    print("4a) lint catches generative without validate: OK")
    test_lint_passes_when_validate_present()
    print("4b) lint passes when validate present: OK")
    test_lint_validate_consumes_register()
    print("4c) lint detects validate via list args: OK")

    # 5. mode_boosts
    test_mode_boosts_raises_score()
    print("5a) mode_boosts raises score: OK")
    test_mode_boosts_no_match_no_effect()
    print("5b) mode_boosts no match → no effect: OK")
    test_use_fitness_adjusts_score()
    print("5c) use_fitness adjusts score: OK")

    test_record_outcome_missing_raises()
    print("6) record_outcome on missing raises: OK")

    print("\nALL FEEDBACK CHECKS PASSED")
