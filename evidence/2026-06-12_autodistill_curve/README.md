# Evidence: where spend falls with repetition (auto-distillation)

Real-API measurement (live `deepseek-chat`), the same task ×5, twice (autodistill
off and on), TWO load profiles. Tokens from the API `usage` field. Script:
`benchmarks/autodistill_curve.py`. Pricing verified 2026-06-11 ($0.14/M in,
$0.28/M out).

## Result — levers against the loop

| profile | base | distill | context-first | combined |
|---|---|---|---|---|
| light (little data) | $0.00034 | **$0.00004 (×9.7)** | — | — |
| heavy (audit) | $0.00100 | $0.00046 (×2.2, variable) | $0.00059 (×1.7, stable) | **$0.00013 (×7.7)** |

**Levers and their niches:**
- **distill** — iteration overhead (many cheap steps over little data): light → ×9.7.
- **context-first** — data carried through the loop (audit / large files): one call
  with the whole workspace instead of ~4 iterations → stable ×1.7 on heavy.
- **combined (both)** — best result on heavy, **×7.7**. Synergy, not displacement:
  on the LEARNING run context-first hands the model all files → it uses compact
  commands (`grep -c`, not cat-everything) → distill captures the lean trajectory
  → replays = synthesis over little data (502 vs 5500 tokens). context-first shapes
  an economical trajectory, distill makes it permanent — which also tames distill's
  data-heavy "lottery".

⚠️ Single 5-run sample; the model is stochastic. Replays (2–5) are stable (the
procedure is deterministic); the variance is in WHAT gets learned on run 1.
Combined reduces it but not to zero. distill alone on data-heavy stays a lottery
(×1–×5).

## Main finding — it refuted the intuition

The hypothesis "the heavier the task, the bigger the saving" was **refuted by
measurement.** Distillation's saving scales **not with task size, but with the
share of iteration overhead in total cost:**

- **light:** 3 loop iterations (list_dir → read_file → answer) over near-empty
  data. Iterations are pure overhead (each re-pays system+tools+history). The
  procedure removes them → deterministic collection (0 LLM) + tiny synthesis → **×9.6**.
- **heavy:** cost is dominated by the DATA in context. Here **context-first** wins
  (data in one call instead of iterations) → stable ×1.7. distill helps only if it
  learned a compact procedure — variable (×1–×5).

## The honest rule

**Distillation cuts the repeated per-iteration context cost, not the data volume.**
Saving ≈ (iterations removed × context per iteration) / total cost.
- Many cheap iterations over small data → big win (light, ×9.6).
- Few iterations, cost in the data → almost no win (heavy, ~×1): the data is the
  floor the synthesis can't go below.

Compare with the 2026-06-11 experiment (Opus, context-first): there the loop ran
8 iterations re-paying 41k of context → the hybrid gave ×3. The difference from
this audit: there the loop was LONG and iteration-wasteful; here DeepSeek did the
audit in ~4 iterations, so there's little to cut.

## What it means for the user

- Repeated multi-step tasks over small data → **distill** (×9.7).
- Tasks with data carried through the loop (audits, large files) → **context-first**
  (stable ×1.7).
- Repeated data-heavy tasks → **enable BOTH**: context-first steers the model to a
  compact learning trajectory, distill makes it permanent → best result (×7.7 in
  this measurement). Safety unchanged (read-only, fitness rollback).

⚠️ Safety boundary unchanged: only read-only tasks are distilled (including
read-only pipelines like `grep … | wc`); writes always stay on the LLM path.
