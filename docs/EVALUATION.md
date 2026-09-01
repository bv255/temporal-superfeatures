# Evaluation: How a GA Run's Output Gets Scored

This document covers everything that happens **after** a fold's genetic search has produced a
winning expression — how that winner (and the baselines it's compared against) get scored, what
significance tests are applied, and how two arms of the temporal-operator ablation (§1 of
`METHODOLOGY.md`) get compared into the pipeline's headline verdict. It does not cover the search
itself (population, mutation, crossover, temporal operators — see `METHODOLOGY.md` §3) except
where the same scoring machinery the search uses internally is reused here unchanged.

There are two distinct layers, run at two different times:

1. **Per-run evaluation** — runs once per fold, inside `run_ga_for_fold()`, for every arm
   (temporal-ON and temporal-OFF) independently. Produces each arm's own `fold_result.json`,
   `final_test_summary.csv`, `pairwise_comparisons.csv`, `winner_composition.csv`.
2. **Cross-run evaluation** — runs once, after *both* arms have finished, via
   `research/compare_ga_runs.py`. Reads both arms' `fold_result.json` trees and produces the
   actual ON-vs-OFF verdict this whole pipeline exists to answer, under `comparison_outputs/` (or
   `comparison_outputs_fast/`).

## 1. Shared scoring mechanic: the GBT fit

Every evaluation below — the GA's own per-generation fitness, every baseline, and the winner's
final true-test score — goes through the identical model-fitting step:
`GBTRegressor(maxIter=10, seed=<deterministic>)`, otherwise PySpark's stock defaults (`maxDepth=5`,
`stepSize=0.1`, `subsamplingRate=1.0`, `lossType="squared"`). Only iteration count and seed are
overridden — nothing about depth, learning rate, or regularization has been tuned. The seed is
derived via `derive_seed(fold_seed, gen, key)` (SHA256-based, not Python's process-randomized
`hash()`), so every fit is both reproducible across reruns and safe to call concurrently
(`evaluate_population` runs these inside a `ThreadPoolExecutor`).

The model is always trained on exactly the columns `prev_month_return`, `prev_month_sector_return`,
and (except for baseline A) whatever candidate `feature` is being scored — winsorized to its
1st/99th training-split percentile first, bounds fit on training rows only. Scoring is RMSE between
predicted and actual next-month return. This is a small, deliberately cheap, fixed-recipe model —
its job is to be an identical yardstick across thousands of candidates, not to be individually
tuned per candidate.

## 2. Held-out protocol

Every fold search uses its **validation split** (the inner-validation window, for final-test
folds — see `METHODOLOGY.md` §2.6) for every fitness evaluation during the search. The **true
held-out test split** is touched exactly once, after the winner is already locked in: once
`ga.run()` finishes, `_evaluate_on_true_test()` builds a fresh `DataCache` from the true test frame
and scores the frozen winning expression against it — the same GBT fit mechanic as above, just on
data no generation of the search ever saw.

## 3. Metrics

- **RMSE** — on both the validation split (search-time, drives fitness) and the true held-out test
  split (reported).
- **Rank IC** — monthly Spearman rank correlation between predicted and realized returns, computed
  separately within each calendar month (never pooled across months), then averaged across months
  into `mean_ic`.
- **IC-IR** — `mean_ic` divided by its standard deviation across months. Measures consistency, not
  just average strength, of the ranking signal.

## 4. Baselines (final-test folds only; development folds get A/B only)

- **Baseline A** — the two fixed lag predictors alone (`prev_month_return`,
  `prev_month_sector_return`), no evolved feature at all. `VectorAssembler(inputCols=[...2 cols])`
  — the evolved-feature column is dropped entirely, not just zeroed. This is the true "no super-
  feature" floor every other score is measured against.
- **Baseline B** — the single best-scoring raw leaf feature at generation 0, before any
  crossover/mutation. Captured for free: `ga.evaluate_population(gen=0)` is called directly right
  after `initialize_population()`, so every gen-0 individual is already cached when `ga.run()`'s
  own first iteration hits the same population — no extra compute.
- **Baseline C** — a matched-compute-budget random search: the same number of candidate
  evaluations the real GA search actually ran (`generations_run`, i.e. wherever early termination
  stopped it — not the configured max), but with uniform random sampling instead of tournament
  selection/crossover. Isolates how much of the GA's edge, if any, comes from evolution itself
  rather than simply trying many candidates. Deliberately reuses the real run's already-warmed
  leaf caches (every call passes `gen=1`, forcing the cache-read path) — not a second full search's
  worth of compute. Optional; skippable via `--no-baseline-c`.

## 5. Secondary baseline comparisons (`analysis/significance.py`, one arm at a time)

**This section implements `evaluation_framework.md`'s "secondary baseline comparisons"**: within
each arm, the GA winner is compared against baselines A (fixed predictors), B (best raw feature),
and C (matched-compute random search, if it ran) — up to three comparisons per arm. Every one of
these is a **directional** claim ("the GA winner's Rank IC exceeds this baseline's"), not merely
"they differ," so the bootstrap below is one-sided throughout, matching the primary comparison's
own test shape (§8) rather than a symmetric two-sided convention.

For each comparison, every final-test fold's matched (winner IC − baseline IC) month-level deltas
are grouped by fold and chronologically ordered within each fold, then:

- **Block bootstrap within each fold** (not by fold identity, not by row): 3,000 resamples: for
  each fold independently, contiguous blocks of chronologically-ordered months (`block_length`
  months per block, a provisional default — see §8's note on calibration) are resampled with
  replacement from that fold's own series; the resampled per-fold series are then pooled across
  folds (every month weighted equally) before averaging. This preserves within-fold
  month-to-month dependence — the same walk-forward structure §8 relies on — while resampling at
  the finer month-block grain rather than whole fold identities.
- **One-sided, null-centered** ("basic"/reflected bootstrap): the raw bootstrap distribution is
  centered near the *observed* delta, not zero, so it's reflected around the observed delta to
  approximate the sampling distribution under the boundary null (delta = 0). The one-sided 95%
  lower bound is `2×observed_delta − the 95th percentile of the bootstrap replicates`; the
  p-value is the fraction of replicates `≥ 2×observed_delta`. See
  `analysis/significance.py`'s `block_bootstrap_ic_delta` docstring for the full derivation.
- **`min_folds = 5` guard**: below this many distinct final-test folds, the bootstrap is skipped
  entirely (`insufficient_folds=True`, lower bound/p-value come back `NaN`) rather than reporting
  a distribution built mostly from repeats of one or two folds.
- **Holm-Bonferroni, per arm**: applied once per arm across that arm's own 2–3 baseline
  comparisons together (comparisons flagged `insufficient_folds` are excluded from the
  correction, not just from significance). This stays a **separate family per arm** — temporal-ON's
  baseline checks are corrected independently from temporal-OFF's — rather than pooling both
  arms' baseline checks into one six-comparison family, the way an earlier version of this
  pipeline did.

## 6. Winner composition

Each final-test fold's winning expression (temporal arm only) is decomposed leaf by leaf: does it
use a temporal operator at all, and what fraction of its leaves are temporal. This is the primary
diagnostic for *why* the ON-vs-OFF comparison comes out the way it does — if temporal outperforms
but winners rarely use a temporal operator, the gap needs a different explanation. Reported
alongside leaf count (expression size) per fold and aggregated.

## 7. Per-arm aggregation

`final_test_summary.csv` — one row per final-test fold: winner vs. baseline A/B/C RMSE + IC,
spanning every held-out year. `pairwise_comparisons.csv` and `winner_composition.csv` as above.
Development folds run the same machinery but only feed hyperparameter tuning, never the reported
aggregation.

## 8. Cross-run evaluation: the primary hypothesis test (`compare_ga_runs.py`)

This is the actual headline result — everything above exists to make this one comparison
trustworthy. Run once both arms (`ga/` and `ga_no_temporal/`, or the `--fast` pair) have finished.
**This section implements `evaluation_framework.md`'s "Primary evaluation" and "Statistical
significance"** — the sole pre-specified primary hypothesis test in this project.

**Matching**: reads both arms' final-test `fold_result.json` trees, matches folds by name, and
warns if a matched fold's `eval_year` has diverged between arms (their walk-forward boundaries
would no longer line up, breaking the pairing).

**Estimand**: for each held-out month `t` in each matched final-test fold `f`, the paired Rank IC
difference is `d_{f,t} = IC_ON_{f,t} − IC_OFF_{f,t}`. The primary effect is
`δ̂ = mean over ALL matched fold-months of d_{f,t}` — every month contributes equally, not a
fold-weighted average.

> **Not yet implemented**: `evaluation_framework.md` specifies averaging `d_{f,s,t}` across **10
> pre-specified matched seeds** per arm *before* aggregating across fold-months
> (`d̄_{f,t} = mean over seeds of d_{f,s,t}`), so that GA randomness is treated as nuisance
> variation rather than folded into the effect estimate. This pipeline currently produces only one
> seed's worth of results per arm — `run_ga.py --seed N` exists for a seed sweep, but nothing yet
> aggregates a matched 10-seed set into `d̄_{f,t}`. `compare_ga_runs.py` runs on whatever single
> seed's results are on disk for each arm and flags this explicitly
> (`seed_averaging_applied=False` in `primary_comparison.csv`/`h1_verdict.json`) rather than
> silently treating a single-seed result as the seed-averaged estimand. Building the seed-matched
> aggregation is upcoming work.

**One-sided, null-centered block bootstrap, standalone (no Holm-Bonferroni correction)**: the
primary comparison uses the exact same bootstrap mechanic as §5's secondary baseline
comparisons — contiguous month-blocks resampled independently within each fold, pooled across
folds, then reflected around the observed `δ̂` to test against the boundary null `δ=0` (see §5's
description, or `analysis/significance.py`'s `block_bootstrap_ic_delta` docstring, for the
derivation). Per `evaluation_framework.md`: *"No Holm-Bonferroni correction is applied to this
test because it is the single pre-specified primary hypothesis."* This is a deliberate change
from an earlier version of this pipeline, which bundled the primary comparison together with
several baseline checks and a DiD test into one six-comparison Holm-Bonferroni family — those
baseline checks now live entirely in §5's per-arm family instead.

**H1 verdict — two conditions, both required for "H1 SUPPORTED"**:

| Condition | What it checks |
|---|---|
| Magnitude | observed `δ̂` (temporal − no_temporal) is positive |
| Statistical reliability | the one-sided 95% lower bound exceeds zero, equivalently p < 0.05 |

This replaces an earlier four-condition design (a Delta minimum-detectable-effect magnitude gate,
a fold-consistency majority gate, and a DiD-attribution gate stacked on top of statistical
reliability) — `evaluation_framework.md` has no minimum-detectable-effect threshold, and is
explicit that fold-by-fold consistency and the DiD-style diagnostic below are *"reported
separately as a robustness diagnostic"* / *"a mechanism diagnostic"* and *"not an additional
hypothesis gate."* Any condition whose inputs aren't available yet (e.g. too few matched folds for
the bootstrap's `min_folds` guard) leaves the verdict **"PENDING"** rather than silently skipping
it; otherwise it's **"H1 SUPPORTED"** or **"H1 NOT SUPPORTED — failed condition(s): [...]"**.

**Descriptive-only diagnostics (reported, do not gate the verdict)**:

- **Fold-by-fold consistency** — how many of the matched final-test folds temporal beats
  no_temporal on mean IC (`winner_composition_comparison.csv`, and a summary line in
  `h1_verdict.json`'s notes).
- **Difference-in-differences (DiD) attribution** — isolates whether an IC edge is attributable to
  the temporal operators specifically, or just to the temporal arm having a strictly larger search
  space to try. Per arm independently: `lift = winner's mean IC − that arm's own baseline C mean
  IC`; the DiD statistic is `lift(temporal) − lift(no_temporal)`. Both arms already have a
  matched-budget random-search baseline (§4); if temporal's edge over *its own* random search
  exceeds no_temporal's edge over *its own* random search, that gap can't be explained by
  search-space size alone. Bootstrapped with the same one-sided/null-centered machinery for
  transparency, but this number is purely informational — per `evaluation_framework.md`, it "may
  also be reported" as an optional mechanism diagnostic, not a formal gate.

**Outputs**, all under `comparison_outputs/` (or `comparison_outputs_fast/`):

| File | Contents |
|---|---|
| `primary_comparison.csv` | the single primary comparison — observed `δ̂`, one-sided lower bound, p-value, uncorrected |
| `attribution_table.csv` | per-fold lift(temporal)/lift(no_temporal)/DiD, plus the aggregate bootstrap numbers — descriptive only |
| `final_test_summary_comparison.csv` | combined per-fold temporal vs. no_temporal summary, both arms side by side |
| `h1_verdict.json` | the two conditions, notes (including the descriptive fold-consistency line), and the final verdict string |
