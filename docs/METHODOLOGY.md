# Methodology: Temporal Super-Feature Evolution Pipeline

This document describes the end-to-end methodology used to discover predictive "super-features"
for monthly stock returns: a causally-safe data preprocessing stage, a genetic-algorithm search
that composes raw fundamentals into candidate features using arithmetic and temporal operators,
and an evaluation protocol that scores each candidate against multiple baselines with statistical
significance testing on a strictly held-out test set. It is written for a reader who wants to
understand the approach and its reasoning, not the code that implements it.

## 1. Overview

**The question this pipeline is built to answer is whether the live temporal operators (stage 2)
actually improve predictive performance — not whether the search finds a good feature in
isolation.** Every other design decision (walk-forward folds, per-fold feature selection,
bootstrap significance testing) exists in service of making that one comparison trustworthy.

This is answered by an **ablation**: the identical pipeline is run twice per scale, with the
temporal-operator family switched on for one run and off for the other. The two runs share the
same preprocessed folds, the same feature selection, the same genetic-algorithm configuration,
and the same evaluation protocol — the single difference between them is whether the search is
allowed to construct and mutate temporal atoms (§3.2). The **no-temporal-operators run is not a
lesser or legacy variant** — it is the control condition, and the entire methodology below is
structured to isolate the effect of adding temporal operators, not to evaluate either run alone.

The pipeline has three stages:

1. **Data preprocessing** — raw fundamentals and price data are cleaned, aligned to the date each
   fact was actually knowable (not merely reported), and split into walk-forward folds with
   per-fold feature selection. Identical for both arms of the ablation.
2. **Temporal feature search** — a genetic algorithm evolves symbolic expressions over the
   selected features. Run twice per scale: once with the temporal-operator family available to
   the search, once with it disabled — otherwise identical configuration.
3. **Evaluation** — each fold's evolved winner is scored on a never-touched held-out test set
   against within-run baselines, with bootstrap confidence intervals and multiple-comparison
   correction; the two arms' held-out results are then compared directly to answer the central
   question in (1).

```
raw fundamentals/prices  →  [1] preprocessing  →  walk-forward folds (train/validation/test)
                                                          │
                              ┌───────────────────────────┴───────────────────────────┐
                              ▼                                                       ▼
                [2] temporal search — operators ON                    [2] temporal search — operators OFF
                              │                                                       │
                              ▼                                                       ▼
                [3] evaluation (held-out test, per fold)               [3] evaluation (held-out test, per fold)
                              │                                                       │
                              └───────────────────────────┬───────────────────────────┘
                                                          ▼
                                    ablation comparison: ON vs. OFF (the headline result)
```

## 2. Data Preprocessing

### 2.1 Data sources and universe

Fundamentals and price data are drawn from the FactSet universe, restricted to **Finance-sector**,
NYSE/NASDAQ-listed, USD-denominated securities with a point-in-time market value above a fixed
threshold (currently 5,000, in the pipeline's native FactSet units), covering roughly 2001–2026. **All results this methodology produces — including the
temporal-operator ablation in §4.4, the pipeline's headline comparison — are therefore scoped to
the Finance sector and should not be read as evidence about the broader market.** The sector
restriction, the exchange/currency filters, and the market-value threshold are all applied at
extraction time, before any preprocessing below runs.

The pipeline runs against the complete Finance universe and fundamentals feature set.

### 2.2 Causal safety

Two data-cleaning steps that previously used a centered (past + future) average to smooth
anomalies and fill missing values were corrected to be strictly backward-looking: outlier
replacement and null interpolation now only ever use the most recent prior observation, never a
future one. Market value — used to filter the investable universe — is reconstructed as a genuine
point-in-time quantity (contemporaneous price × contemporaneous shares outstanding, as-of joined)
rather than taken from a current-day snapshot table, which had been silently applying today's
market value to decide universe membership at every past date and introducing a look-ahead/
survivorship bias.

### 2.3 Point-in-time alignment

Fundamentals data carries a fiscal period-end date, which is not the date the figures actually
became public. Using period-end date as a proxy for "known" understated the true reporting lag for
the large majority of filings. Each report is instead assigned a **filing date** derived from its
actual source filing dates (falling back to a fitted lag — the 90th percentile of the observed
filing-lag distribution — when a source date is unavailable). A report is only treated as usable
starting the month after its true filing date, and its values are carried forward (as-of joined)
month by month until superseded by that company's next report. This closes a lookahead channel
that had let the pipeline "know" fundamentals figures before they were actually public.

### 2.4 Feature engineering

Two backward-only return-lag features — a company's own prior-month return and its sector's
prior-month average return — are computed once, globally, and always retained as baseline
predictors, exempt from feature selection.

### 2.5 Feature selection

For each fold, the raw candidate fundamentals columns pass through three filtering stages, in
order — every stage below is fit **per fold, on that fold's training rows only**, never globally
across the full date range, so that a feature which happens to look informative only because of
data available in a later, still-unseen period cannot influence which features a fold is even
allowed to consider. Two features — a company's own prior-month return and its sector's
prior-month average return (§2.4) — are exempt from all three stages and always survive into the
final feature list unconditionally.

1. **Missingness filter** — a candidate feature is dropped if more than 30% of its values are
   missing within the fold's training rows.
2. **Static/near-constant filter** — across every company present in the fold's training data
   (no sampling — an earlier version sampled ~50 companies via a fixed random seed that turned
   out not to be reproducible in practice, since Spark's `.sample()`+`.limit()` combination isn't
   deterministic across separately-materialized DataFrames; checking every company removed that
   noise source rather than trying to pin the sample down further), a feature is flagged as
   static in a given company if that company's own SD/RMS ratio for the feature falls below a
   fixed threshold (0.10) — i.e. its within-company standard deviation is small relative to its
   own root-mean-square level, a bounded, scale-invariant stand-in for the ordinary coefficient
   of variation (equivalent to CV below ~10%, but well-behaved as a company's own mean
   approaches zero, unlike CV itself). An earlier version instead compared a company's local
   variance to one variance pooled across *every* company — found, from a real fold, to
   systematically misclassify small companies as static purely because `variance()` scales with
   the square of a value's magnitude: a handful of large companies' huge absolute swings
   dominated the pooled figure, so a small company's proportionally-normal (but absolutely tiny)
   variance looked negligible by comparison regardless of how much it genuinely varied relative
   to its own scale. A feature is dropped if it is static (by this per-company test) in more
   than half of the companies checked — that aggregation rule is unchanged.
3. **Consensus procedure** — every surviving feature is scored by three independent criteria
   (Spearman |correlation| with the target, mutual information, and random-forest permutation
   importance, each computed across every company in the fold — no company sampling; an earlier
   version sampled companies with a fixed random seed that turned out not to be reproducible in
   practice, for the same reason described for the static/near-constant filter above, and was
   removed rather than patched) and retained only if it ranks in the top 50% by at least two of
   the three. Random forest fitting, permutation importance, and mutual information estimation
   each fix their own `random_state` to the fold's seed, so the residual randomness internal to
   those methods (bootstrap sampling, feature subsampling, permutation shuffling, KSG
   tie-breaking noise) is reproducible even though the company set they run on is no longer
   sampled at all. Retained features are then de-duplicated by hierarchical (average-linkage)
   clustering on pairwise |Spearman correlation| distance, computed across every company too,
   cutting at |ρ| ≥ 0.80, keeping one representative per cluster (highest consensus score, ties
   broken by lower missingness, then company/sector coverage, then feature name). The final list is
   **uncapped** — every surviving representative is kept, however many that is.

Every threshold above (30% missingness, 0.10 per-company SD/RMS staticness, the 30%/2-of-3
consensus rule, the 0.80 clustering cutoff) is a fixed constant applied identically to every fold —
none of it is tuned or searched, and none of it differs between the temporal-ON and temporal-OFF
arms of the ablation (§1), since feature selection has no temporal-operator awareness at all.

### 2.6 Walk-forward structure

Fold boundaries are derived from the years actually present in the data rather than fixed dates.
Every dev-validation and final-test fold covers a fixed **window width**, in years, rather than a
single calendar year (currently 2). The final-test region's size is pinned to a fixed **fold
count** (currently 5) times that window width, counted back from the most recent available year —
overriding what a fixed fraction of history would otherwise select, and, if the requested fold
count/width combination demands more years than a fraction-based split would have given it, eating
into what would otherwise be development or initial-training years to fit. A fixed number of
development folds (currently 3), each the same window width, sit immediately before the final-test
region; every year before that is available as initial training history. Over the pipeline's
~2001–2026 coverage, this currently works out to initial training 2001–2010, three 2-year development folds
(2011–12, 2013–14, 2015–16), and five 2-year final-test folds (2017–18 … 2025–26) — a materially
smaller initial-training window than a 1-year-per-fold split would leave, in exchange for pinning
the final-test fold count and width outright rather than letting either fall out of a fraction.

- **Development folds** — one per development window, each with an *expanding* training window
  (from the earliest available year through the year immediately before that fold's own
  validation window begins) and that window held out as validation. A simple two-way
  train/validation split.
- **Final-test folds** — one per final-test window, each with a further-expanded training window,
  but a **three-way** split rather than two-way: training data ends one window-width before the
  **inner validation window**, which is itself exactly one window-width wide and sits immediately
  before the test window, used the same way development folds' validation is (fitness scoring
  during the search, §3.3/§4.1); the test window itself is the true held-out split, touched exactly
  once (§4.1). The inner validation window is not extra data — it is the same years that would
  otherwise be the tail of the training window — carved out specifically so a final-test fold's
  actual test split stays genuinely untouched by any decision (feature selection above, GA
  selection during the search) until after a winning expression is already locked in. (A
  consequence of the window width and expanding train window sharing the same increment: each
  final-test fold's inner-validation window is identical to the *previous* final-test fold's own
  test window — that reuse is intentional, since the inner validation window only ever feeds
  in-search fitness scoring, never the frozen training fit, and the prior fold's genuine held-out
  score was already locked in before this fold's search began.)

Every fold additionally purges a fixed-width embargo (currently one month) from the tail of its
training window, so the last training observation is never immediately adjacent to the first
validation observation — and, for final-test folds, the same embargo is purged from the tail of
the inner-validation window too, so it is never immediately adjacent to the test window either.

The result of preprocessing is not a single flat dataset but a tree of walk-forward folds, each
with its own independently-selected feature list (§2.5) and its own train / validation /
(final-test folds only) held-out test split.

## 3. Temporal Feature Search

### 3.1 Representation

A candidate feature ("individual") is a symbolic expression built from selected leaf features
combined with arithmetic operators (`+ − × ÷`). The search operates on a population of these
expressions per fold.

### 3.2 Temporal operators

The search is given a family of **live temporal operators** (lag, delta, growth, rolling mean,
rolling std) that can wrap an *arbitrary evolved sub-expression*, not only a single raw feature —
for example, the one-period lag of (market value ÷ operating margin), a quantity that has no
precomputed column because it doesn't exist until the search constructs it. This is what allows
the search to discover genuinely temporal relationships between derived quantities rather than
only between raw fields.

An earlier version of this pipeline additionally precomputed the same five static transforms
(one-period lag, delta, growth, and 3-period rolling mean/std) of every raw feature as ordinary
candidate columns during preprocessing, before the live operators existed. That precomputation is
now switched off by default — validated once against the live evaluator on real fold data
(99.5–99.8% exact match on the small remaining divergence between the two, understood and
explained, not an open bug) before being retired — since the live operators can reconstruct the
same single-feature transforms and more. It remains available as an optional, off-by-default
toggle, not part of the default candidate feature set either arm of the ablation sees.

Temporal operators are windowed over each company's own **reporting cadence** — one step means
"one report ago" — rather than the expanded monthly grain the rest of the pipeline operates at, so
that a lag of a quarterly-reported field reflects roughly four changes a year, not a monthly
artifact of forward-filling.

The default operator family is exactly five operators (one lag period, one rolling-window width):
`lag1`, `delta1`, `growth1`, `mean3`, `std3` — the same five transforms the retired precomputation
above used to produce as static columns, just evaluable live over an arbitrary sub-expression
instead of only a raw feature. The lag period(s) and window width(s) are configurable but fixed at
these single defaults for both arms of the ablation.

The search can structurally wrap a contiguous evolved sub-expression in a temporal operator, or
remove such a wrap, as a pair of symmetric mutation operations, bounded by complexity limits (a
cap on total feature count and on stacking depth — see §3.3/§5 for the current numeric caps).
Redundant wraps are excluded from the search space: temporal operators that are linear in their
input (lag, delta, rolling mean) distribute over addition and subtraction, so wrapping a purely
additive sub-expression in one of these is always reachable more directly at the leaf level and is
not offered as a distinct search move. Non-linear operators (growth, rolling standard deviation)
have no such shortcut and are always available as genuine subtree wraps — this is precisely the
class of relationship the temporal-operator family exists to reach.

Because introducing a temporal wrap purely through mutation was found to be adopted too slowly to
meaningfully bootstrap from a population with none (single-digit percent adoption after several
generations), the temporal-ON arm additionally pre-wraps a fixed fraction of generation-0
individuals (currently 30%) in a randomly chosen temporal operator before the search begins, on
top of whatever the wrap/unwrap mutation moves introduce afterward. This seeding step has no
effect in the temporal-OFF arm (there is no operator family to seed from).

### 3.3 Search mechanics

Each generation: candidate expressions are scored, the population is advanced via tournament
selection (tournament size 4), crossover, and mutation (feature substitution, operator
substitution, or temporal wrap/unwrap), and the single best individual is carried forward
unmutated (elitism). Mutation is applied at an adaptive rate rather than a fixed one: it scales
between a floor and ceiling (currently 10%–40%) as a function of how much population diversity has
been lost, so mutation pressure rises automatically as the population converges. A candidate
expression is capped at 5 leaf features and a nesting depth of 5; a temporal wrap can stack to a
depth of 2. Newly created individuals (from crossover or elitism) that would exactly duplicate one
already accepted earlier in the same generation are forced through an extra mutation (retried up
to 3 times) rather than admitted as-is, so the population doesn't collapse toward repeats of the
same expression.

A candidate's fitness is computed by combining its expression into a single derived feature,
**winsorizing that combined feature column to its 1st/99th percentile** (bounds fit on the fold's
training rows only, applied to both training and validation rows), adding it to the two fixed
baseline predictors, fitting a gradient-boosted tree regressor (a fixed
number of boosting iterations, 10, otherwise default hyperparameters — the same model class and
configuration used for every baseline in §4.3 too) on the fold's training data, and scoring RMSE
on the fold's validation split (the inner validation year for final-test folds, §2.6) — fitness is
the negative of this RMSE. Winsorization guards against expressions whose combined value has no
fixed precomputed column and can occasionally explode — a division, or a `growth` temporal
operator (§3.2) — so it is applied fresh to each individual's own combined feature value, every
generation and fold, never to the raw candidate columns during preprocessing (§2), since the
expression producing an extreme value doesn't exist until the search constructs it.

Population size is 100, capped down further for any fold with fewer evolvable features than that.
The search runs for up to 500 generations, but stops earlier once either of two independent
conditions is met:

- **Converged**: the best fitness has been unchanged for 10 generations, *and* over that same
  window a large fraction of newly created individuals (currently averaging ≥15%) needed the
  forced-mutation duplicate-avoidance step above — i.e. the search keeps landing on expressions it
  has already tried, not merely a population that happens not to have improved yet. (Raw
  population diversity — the count of distinct expressions — is not used for this check: the
  duplicate-avoidance policy above keeps that count high almost by construction, even once the
  population has behaviorally converged around one dominant building block, so it doesn't
  distinguish a converged search from a still-exploring one.)
- **Stagnation ceiling**: an unconditional cap — currently 50 generations with no improvement at
  all — regardless of the condition above, so a fold that stays nominally diverse without ever
  improving still has a bound rather than running to the full generation limit for no benefit.

Each fold's search is run completely independently — no population, cache, or state is shared
across folds — and is fully reproducible from a single fixed base random seed, with each fold
deriving its own independent seed from it.

## 4. Evaluation

### 4.1 Held-out protocol

The validation split is used only for fitness scoring during the search. The final-test split is
touched exactly once, after the search for that fold has concluded, to score the winning
expression — it is never seen during any generation that produced it.

### 4.2 Metrics

- **RMSE**, on both the validation split (search-time) and the true held-out test split.
- **Rank IC**: the monthly Spearman rank correlation between predicted and realized returns,
  computed separately within each calendar month (never pooled across months) and then averaged
  across months.
- **IC-IR**: mean monthly IC divided by its standard deviation across months — a measure of how
  consistently, not just how strongly, a feature's ranking tracks realized returns.

### 4.3 Within-run baselines

Within each arm of the ablation (temporal ON, and separately temporal OFF), every fold's evolved
winner is compared against three baselines, run under that same arm's configuration:

- **Baseline A** — the two fixed lag predictors alone, with no evolved feature.
- **Baseline B** — the single best raw feature found at generation 0, with no evolution applied.
- **Baseline C** — a matched-compute-budget random search: the same number of candidate
  evaluations as the genetic algorithm actually ran (i.e. matching however many generations the
  real search's own early-termination logic, §3.3, stopped at, not the configured maximum), but
  without any selection pressure — uniform random sampling of expressions each round instead of
  tournament selection/crossover. This isolates how much of the genetic algorithm's advantage, if
  any, comes from evolution itself rather than simply from trying many candidates. Baseline C is
  computed only for final-test folds (development folds get baselines A/B only) and can optionally
  be skipped entirely for a faster run, in which case only A and B are compared.

These three baselines answer "did the search do anything useful at all, within this arm" — a
prerequisite check, distinct from the ablation comparison in §4.4, which is what answers "did
adding temporal operators help."

### 4.4 Primary comparison: temporal operators vs. no operators

This is the comparison the whole pipeline exists to produce, and implements the primary hypothesis
from `evaluation_framework.md` (the project's pre-registered evaluation spec — see
`docs/EVALUATION.md` §8 for the full mechanics). Because the two arms differ only in whether the
temporal-operator family is available to the search (§1), any difference between the temporal-ON
arm's held-out results and the temporal-OFF arm's held-out results — matched fold by fold, since
both arms share the identical walk-forward split — is attributable to the temporal operators
themselves, not to a confound in the data, the folds, or the feature selection.

The no-temporal-operators arm is therefore the **control** for this experiment: its role is not to
be evaluated as a standalone result, but to establish what the search achieves *without* the
capability under test, so that the temporal-ON arm's result can be read as an effect size rather
than an absolute number.

**Estimand**: for each held-out month `t` in each matched final-test fold `f`, the paired Rank IC
difference is `d_{f,t} = IC_ON_{f,t} − IC_OFF_{f,t}`. The primary effect `δ̂` is the mean of
`d_{f,t}` over every matched fold-month (every month weighted equally, not fold-weighted).
`evaluation_framework.md` specifies averaging this paired difference across **10 pre-specified
matched seeds per arm** before aggregating across fold-months, so that GA randomness is treated as
nuisance variation rather than folded into the effect estimate — **not yet implemented**: this
pipeline currently produces one seed's worth of results per arm (`run_ga.py --seed N` supports a
seed sweep, but nothing yet aggregates a matched 10-seed set), so the primary comparison runs on a
single seed per arm and flags this explicitly in its output (`seed_averaging_applied=False`)
rather than silently treating it as the seed-averaged design. Tracked as upcoming work.

This comparison is automated, run once both arms have finished (`compare_ga_runs.py`): it reads
each arm's final-test `fold_result.json` tree directly, matches folds by name (warning if a fold's
`eval_year` has diverged between the two arms, which would mean their walk-forward boundaries no
longer line up), and tests `δ̂` **standalone** via the one-sided, null-centered block bootstrap
described in §4.5 — no Holm-Bonferroni correction, since it is the single pre-specified primary
hypothesis. This is a deliberate change from an earlier version of this pipeline, which bundled the
primary comparison together with the no-temporal arm's baseline checks, the temporal arm's own
baseline-C check, and a DiD attribution test into one six-comparison Holm-Bonferroni family — the
baseline checks now live entirely in each arm's own per-arm family (§4.3/§4.5), and the DiD test is
retained only as a descriptive diagnostic (below), not a gate.

**Overall verdict**: two conditions, both required for "H1 SUPPORTED":

- **Magnitude** — the observed `δ̂` (temporal − no_temporal, across matched final-test folds) is
  positive.
- **Statistical reliability** — the one-sided 95% lower bound (§4.5) exceeds zero, equivalently the
  p-value is below 0.05.

This replaces an earlier four-condition design that additionally required the effect to clear a
**Delta** minimum-detectable-effect threshold and a difference-in-differences (DiD) attribution
test, with cross-fold consistency evaluated as a third gate. `evaluation_framework.md` has no
minimum-detectable-effect threshold at all, and is explicit that fold-by-fold consistency and a
DiD-style mechanism check are *"reported separately as a robustness diagnostic"* /
*"not an additional hypothesis gate"* — so both are now reported purely descriptively:

- **Fold-by-fold consistency** (descriptive) — how many of the matched final-test folds temporal
  beats no_temporal on mean IC.
- **Difference-in-differences (DiD) attribution** (optional descriptive diagnostic) — isolates
  whether an IC edge is attributable to the temporal operators specifically, or just to the
  temporal arm having a strictly larger search space to try. Baseline C (§4.3) — a
  matched-compute-budget random search within each arm's own leaf vocabulary — already isolates
  "selection pressure vs. trying many candidates" *within* one arm; the DiD check reuses it to look
  at the same thing *across* arms: per arm, `lift = winner's mean IC − that arm's own baseline C
  mean IC`, and the DiD statistic is `lift(temporal) − lift(no_temporal)`. Bootstrapped with the
  same one-sided/null-centered machinery for transparency, but it is not a hypothesis gate and is
  not part of any correction family.

Any condition whose inputs aren't yet available (e.g. too few matched folds for the bootstrap's
`min_folds` guard, §4.5) leaves the verdict **"PENDING"** rather than silently skipping it. All of
the above — the primary comparison, the attribution table, the winner-composition deltas, and the
final verdict — are written to `comparison_outputs/` (or `comparison_outputs_fast/`) as
`primary_comparison.csv`, `attribution_table.csv`, `final_test_summary_comparison.csv`, and
`h1_verdict.json` respectively.

### 4.5 Statistical significance

Confidence intervals and p-values on IC deltas are computed via a **dependence-preserving block
bootstrap**, per `evaluation_framework.md`. Resampling is performed **separately within each
final-test fold**, so blocks never cross fold boundaries: for each fold independently, contiguous
blocks of its own chronologically-ordered monthly IC-delta series are resampled with replacement
(block length currently 3 months — a provisional default, not yet calibrated against development
data as the spec calls for), preserving local month-to-month dependence within that fold. The
resampled per-fold series are then pooled across folds — every month contributes equally to one
bootstrap replicate, matching the `δ̂` estimand (a plain mean over all fold-months, not a
fold-weighted average) — repeated 3,000 times to build the bootstrap distribution. This is a
change from an earlier version of this pipeline, which resampled whole **fold identities** with
replacement rather than month-blocks within each fold.

The bootstrap distribution is then **null-centered** ("basic"/reflected bootstrap) to test against
the boundary null `δ=0`: because the raw distribution of bootstrap replicates is centered near the
*observed* `δ̂`, not zero, it's reflected around `δ̂` to approximate the sampling distribution under
the null — the one-sided 95% lower bound is `2×δ̂ − the 95th percentile of the replicates`, and the
p-value is the fraction of replicates at or above `2×δ̂`. Every hypothesis this project tests this
way is **directional** — "the comparator's Rank IC exceeds the baseline's/the other arm's," not
merely "they differ" — so this is one-sided throughout, both for §4.4's primary comparison and for
§4.3's within-arm baseline checks, not a symmetric two-sided convention.

This bootstrap needs a minimum number of distinct final-test folds (currently 5, matching the
current final-test fold count from §2.6 exactly, with no slack) to produce a meaningful
distribution — below that, too large a share of resamples would repeat the same one or two folds,
making the result look artificially precise from very little real data; when that minimum isn't
met, the comparison is still reported but flagged as unreliable (`insufficient_folds=True`), with
no lower bound or p-value computed. Because the minimum currently matches the final-test fold count
exactly, any future reduction in that fold count (§2.6) will silently disable this significance
testing again unless the minimum is lowered to match.

This same bootstrap machinery computes both correction families in this pipeline:

- The three **within-run** baseline comparisons per arm (winner vs. A, winner vs. B, winner vs.
  C — §4.3), each aggregating every final-test fold via the bootstrap above, then corrected with a
  single Holm-Bonferroni pass **per arm, across that arm's own 2–3 comparisons** (temporal-ON's
  baseline checks and temporal-OFF's baseline checks are corrected as two separate families, not
  pooled together).
- The **primary** temporal-vs-no_temporal comparison (§4.4), computed the same way but reported
  **standalone with no correction** — it is the single pre-specified primary hypothesis, not part
  of either arm's baseline family.

### 4.6 Winner composition

Each final-test fold's winning expression, in the temporal-ON arm, is decomposed leaf by leaf into
temporal vs. raw composition, reporting whether it uses a temporal operator at all and what
fraction of its leaves do. This is the primary diagnostic for *why* the ablation comparison in
§4.4 comes out the way it does: if the ON arm outperforms the OFF arm but winning expressions
rarely use a temporal operator, that gap cannot be attributed to the operators themselves and the
result needs a different explanation. It is reported alongside expression size (leaf count) as a
complexity measure, per fold and aggregated across all final-test folds.

Development folds are also run end-to-end and produce their own winner/rank-IC/composition results
by the same mechanism, but they feed hyperparameter tuning (§5) rather than the headline
aggregation below — only final-test-fold results are aggregated into the reported summary.

### 4.7 Aggregation

Results across all final-test folds are aggregated into a single summary per arm — winning
expression, true held-out RMSE, rank IC and IC-IR against each within-run baseline, and leaf
composition — spanning every held-out year. The two arms' summaries are then placed side by side
to produce the ablation result from §4.4, which is the headline result of the pipeline.

## 5. Hyperparameter Tuning

Genetic algorithm hyperparameters (population size, generation count, mutation rate and schedule,
crossover complexity limits, temporal wrap/unwrap rates, and the early-termination thresholds)
are tuned exclusively against the development validation folds — the training-side portion of the
walk-forward split — and never against the final-test folds. This keeps the final-test evaluation
genuinely held out and uncontaminated by any tuning decision made elsewhere in the pipeline.

Every hyperparameter shared by both arms of the ablation (§1) is held identical between the
temporal-ON and temporal-OFF runs; only `enable_temporal_operators` itself and the
temporal-specific parameters (wrap/unwrap rates, the seeded-fraction bootstrap, lag periods/window
widths) differ — all of which are inert (have no effect) when the family is disabled. Tuning
either arm's shared hyperparameters separately from the other would confound the ablation — an
ON-vs-OFF gap could then reflect a hyperparameter difference rather than the temporal operators
themselves — so any tuning pass is applied to both arms' shared settings at once, not independently
per arm.

Current defaults:

| Parameter | Default | Shared or temporal-only |
|---|---|---|
| Population size | 100 (capped down for folds with fewer evolvable features) | shared |
| Generations (max) | 500 | shared |
| Tournament size | 4 | shared |
| Mutation rate | adaptive, 10%–40% (§3.3) | shared |
| Max expression size / nesting depth | 5 leaves / depth 5 | shared |
| Early-termination: converged | 10-gen fitness plateau + ≥15% admitted-duplicate rate (§3.3) | shared |
| Early-termination: stagnation ceiling | 50 generations with no improvement | shared |
| Temporal operator family | lag1, delta1, growth1, mean3, std3 | temporal-only |
| Temporal wrap / unwrap rate (mutation) | 15% / 15% | temporal-only |
| Seeded-fraction bootstrap (gen-0 pre-wrapping) | 30% | temporal-only |
| Max temporal stacking depth | 2 | temporal-only |

The early-termination thresholds (converged / stagnation ceiling) are the most recently introduced
of these. The admitted-duplicate-rate threshold has had one calibration pass: an initial 50%
starting point turned out to be unreachable in practice — two live full-scale runs, checked
mid-flight, showed the admitted-duplicate rate oscillating around 10–13% even after 17–24
generations of stagnant best fitness, so the converged branch never fired and every fold fell
through to the stagnation ceiling instead. It was lowered to 15%, just above that observed noise
level, so the intended early-convergence path can actually trigger. This has not yet been checked
against real development-fold runs specifically, and the stagnation ceiling itself is still an
uncalibrated reasoned starting point (see §3.3) — both are to be revisited as more runs accumulate
under this logic.
