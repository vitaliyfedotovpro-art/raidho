"""
VSA core — Vector Symbolic Architecture primitives (MAP model, bipolar).

Promoted from the killer experiments of Phase 0/1 (both PASS): the binding
algebra holds 100+ pairs at D=10k and survives grounding from real embeddings
(SimHash, corr 0.988). This is the reusable core.

  bind(a, b)   = a ⊙ b            (elementwise product; self-inverse)
  unbind(x, r) = x ⊙ r            (same operation — MAP is bipolar)
  bundle(V)    = sign(Σ V)        (superposition with majority sign)
  ground(e, P) = sign(e · P)      (SimHash: embedding → bipolar hypervector)
"""

from __future__ import annotations

import numpy as np

DEFAULT_D = 10_000


def random_atoms(n: int, D: int, rng: np.random.Generator) -> np.ndarray:
    """n random bipolar hypervectors {-1,+1}^D (quasi-orthogonal)."""
    return rng.integers(0, 2, size=(n, D), dtype=np.int8).astype(np.float32) * 2.0 - 1.0


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b


def unbind(x: np.ndarray, role: np.ndarray) -> np.ndarray:
    return x * role  # MAP: binding is self-inverse


def permute(x: np.ndarray, k: int = 1) -> np.ndarray:
    """Cyclic shift ρ^k — encodes position in a sequence (episodes)."""
    return np.roll(x, k)


def unpermute(x: np.ndarray, k: int = 1) -> np.ndarray:
    """Inverse permutation ρ^{-k}."""
    return np.roll(x, -k)


def bundle(vectors: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Superposition with majority sign; ties (sum==0) are broken randomly."""
    s = vectors.sum(axis=0)
    out = np.sign(s)
    ties = out == 0
    if ties.any():
        r = rng if rng is not None else np.random.default_rng(0)
        out[ties] = r.integers(0, 2, size=int(ties.sum())) * 2.0 - 1.0
    return out.astype(np.float32)


def make_projection(emb_dim: int, D: int, rng: np.random.Generator) -> np.ndarray:
    """Fixed random projection for SimHash grounding (emb_dim → D)."""
    return rng.standard_normal((emb_dim, D)).astype(np.float32)


def ground(embedding: np.ndarray, projection: np.ndarray) -> np.ndarray:
    """embedding → bipolar hypervector via SimHash (sign of a random projection).
    Preserves angular closeness: cos(atoms) ≈ 1 − 2·θ/π."""
    atom = np.sign(embedding @ projection).astype(np.float32)
    atom[atom == 0] = 1.0
    return atom


def cosine_to_codebook(vec: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Cosine of bipolar vec to every codebook row (= dot/D on ±1)."""
    return codebook @ vec


# ----------------------------------------------------------------------
# Bit-packed similarity (recall optimization — ×32 RAM, ~×3 speed)
#
# Bipolar ±1 values are stored as bits (1 ⟺ component > 0). For two ±1 vectors
# dot = #matches − #mismatches = D − 2·hamming, where hamming = popcount(XOR of
# the bit masks). So cos = dot/D = (D − 2·popcount(XOR))/D — BIT-FOR-BIT the same
# as (cb @ vec)/D on ±1, hence the ranking (argmax/argsort) is identical to float.
# ----------------------------------------------------------------------
if hasattr(np, "bitwise_count"):          # numpy ≥ 2.0 — native vectorized popcount
    _popcount = np.bitwise_count
else:                                      # portable fallback (per-byte LUT)
    _POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def _popcount(a: np.ndarray) -> np.ndarray:
        return _POPCOUNT_LUT[a]


def pack_bipolar(v: np.ndarray) -> np.ndarray:
    """Bipolar ±1 → bit packing (uint8). bit=1 ⟺ component > 0.
    The last axis is packed; D is a multiple of 8 → no tail padding effects."""
    return np.packbits(v > 0, axis=-1)


def unpack_bipolar(packed: np.ndarray, D: int) -> np.ndarray:
    """Inverse: packed bits → bipolar ±1 (float32), of length D."""
    bits = np.unpackbits(packed, axis=-1)[..., :D]
    return bits.astype(np.float32) * 2.0 - 1.0


def hamming_cosine(packed_cb: np.ndarray, packed_vec: np.ndarray, D: int) -> np.ndarray:
    """Cosine over packed bits: (D − 2·popcount(cb XOR vec))/D. Identical to
    (cb @ vec)/D on ±1. packed_cb (N, B) uint8, packed_vec (B,) uint8 → (N,) float."""
    ham = _popcount(np.bitwise_xor(packed_cb, packed_vec)).sum(axis=-1)
    return (D - 2.0 * ham) / D
