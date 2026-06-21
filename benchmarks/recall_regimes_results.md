# Raidho VSA memory — recall accuracy across two data regimes

Embedder `paraphrase-multilingual-MiniLM-L12-v2` · D=10000 · Q=200 sampled queries/checkpoint. Structural object recall via `query({subject,relation}) -> object`.

## SPARSE-FUNCTIONAL — `(subject,relation)` unique, large concept pool
Best case: no query ambiguity, low cleanup collision.

| N facts | query acc | latency note |
|---:|---:|:--|
| 500 | 1.000 | — |
| 10,000 | 1.000 | — |
| 50,000 | 1.000 | — |
| 100,000 | 1.000 | — |

## DENSE-NON-FUNCTIONAL — triple-unique only, small pool (500 entities)
Realistic interference. `strict` = exact object of this triple; `lenient` = any object ever stored for that `(subject,relation)`. The strict↓lenient gap is *ambiguity*, not capacity loss.

| N facts | strict acc (mean±std) | lenient acc (mean±std) |
|---:|---:|---:|
| 1,000 | 0.955 ± 0.023 | 1.000 ± 0.000 |
| 2,000 | 0.933 ± 0.005 | 1.000 ± 0.000 |
| 4,000 | 0.875 ± 0.015 | 1.000 ± 0.000 |
| 8,000 | 0.792 ± 0.042 | 1.000 ± 0.000 |
| 16,000 | 0.615 ± 0.022 | 1.000 ± 0.000 |

## Reading

- **Sparse-functional: exact recall stays at 1.000 from N=500 to 100,000.** With
  unambiguous keys and a separated concept pool, explicit-store structural recall does
  not decay with scale.

- **Dense-non-functional: `lenient` is 1.000 at every N and every seed, while `strict`
  declines 0.96 → 0.61.** This is the important result. `lenient = 1.000` means the
  memory *always* returns an object that was genuinely stored for that
  `(subject, relation)` — it never forgets a fact and never returns garbage. The
  `strict` decline is therefore **query ambiguity, not capacity loss**: when one
  `(subject, relation)` key maps to many objects, the key under-determines the answer,
  so returning *a* valid object (rather than the one specific triple the test asked
  about) is correct behaviour, not a defect.

- **Consequence for interpretation:** a strict-match score on non-functional data
  conflates ambiguity with forgetting. Measured here, the apparent "capacity decline"
  is the former; the facts are retained and retrievable at every scale tested
  (`lenient = 1.000`). The genuine limit is the *probe*: disambiguating a non-unique
  key needs more than `(subject, relation)`, not more memory.

- **Accuracy is regime-dependent — a single headline number would hide this.**

### Scope / not measured here
Query **latency** grows linearly (O(N) similarity scan over the fact bank) — this is an
architectural property, addressable with an ANN index over the bank without changing the
memory model. It is not benchmarked in this file; this file isolates *accuracy* vs data
regime. Numbers are from this repo's vendored `vsa` core on Apple silicon; reproduce with
`python3 benchmarks/recall_regimes.py`.