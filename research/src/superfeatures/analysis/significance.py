"""
Ported from `research/GA_test.ipynb` cell 22 — the bootstrap/paired-difference significance
helpers (items 4/5 of ga_methodology_additions_prompt.md: Holm-Bonferroni correction, block
bootstrap on IC deltas). See docs/RESTRUCTURING_TODO.md / the port plan. Split out of the
former `ga/methodology.py` (now `evaluation/metrics.py`, which keeps the rank-IC/expression-
composition/baseline-C-sampler helpers) since these two are specifically significance-testing
primitives, not metrics themselves - `summary.py`'s `build_pairwise_comparisons` (the driver
function that calls both) lives alongside them here.

`block_bootstrap_ic_delta` below was later REDESIGNED (package-only, notebook left frozen per
"port faithfully first") to match `evaluation_framework.md`'s pre-registered evaluation spec -
one-sided/null-centered, resampling contiguous month-blocks within each fold rather than whole
fold identities. See that function's own docstring. This is the same category of deliberate
package/notebook divergence as the `ConsensusFeatureSelector` singleton-cluster fix and
`remove_static_features`'s ratio-based rewrite documented in CLAUDE.md - the notebook's own copy
of this helper (`GA_test.ipynb` cell 22) is untouched and still two-sided/fold-level, and
`research/tests/test_ga_methodology_additions.py` tests the two shapes separately rather than
via its usual notebook/package parametrization.
"""
import math
import random
import numpy as np


# ---- Item 5: Holm-Bonferroni correction ----

def holm_bonferroni(pvalues: list) -> list:
    """
    Manual Holm-Bonferroni step-down correction (statsmodels isn't in pyspark-venv and this is
    short enough not to be worth adding the dependency for). Returns adjusted p-values in the
    SAME order as the input `pvalues` list.
    """
    n = len(pvalues)
    order = sorted(range(n), key=lambda i: pvalues[i])
    adjusted_sorted = []
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * pvalues[idx]
        running_max = max(running_max, adj)
        adjusted_sorted.append(min(running_max, 1.0))
    adjusted = [None] * n
    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]
    return adjusted


# ---- Item 4: block bootstrap on IC deltas (one-sided, null-centered, block = contiguous
# months within a fold) - see evaluation_framework.md and this module's own docstring ----

def _moving_block_resample(values: list, block_length: int, rng: random.Random) -> list:
    """
    One fold's contribution to one bootstrap replicate: draws overlapping contiguous blocks of
    `block_length` months, with replacement, from every valid starting position in `values`
    (assumed already sorted into chronological order by the caller), concatenating blocks until
    the resampled series reaches at least len(values), then truncating back to that length. If
    the fold has fewer months than block_length, the whole series is used as a single block
    (nothing shorter to draw from).
    """
    n = len(values)
    if n == 0:
        return []
    if block_length >= n:
        return list(values)
    max_start = n - block_length
    n_blocks = math.ceil(n / block_length)
    resampled = []
    for _ in range(n_blocks):
        start = rng.randint(0, max_start)
        resampled.extend(values[start:start + block_length])
    return resampled[:n]


def block_bootstrap_ic_delta(fold_month_deltas: dict, block_length: int = 3, n_resamples: int = 3000,
                              seed: int = 42, min_folds: int = 5) -> dict:
    """
    Dependence-preserving block bootstrap per evaluation_framework.md's primary-hypothesis
    specification. Every hypothesis this project tests with this function is DIRECTIONAL - "the
    comparator's Rank IC exceeds the baseline's/the other arm's", not merely "they differ" - so
    this is one-sided throughout, not just for the temporal-ON-vs-OFF comparison.

    `fold_month_deltas` is {fold_name: [delta_1, delta_2, ...]}, one list of month-level
    (comparator_ic - baseline_ic) deltas per final-test fold, EACH LIST ALREADY SORTED INTO
    CHRONOLOGICAL ORDER by the caller - the block structure below is meaningless over an
    arbitrarily ordered list.

    Resampling is performed separately within each fold: contiguous blocks of `block_length`
    months are resampled with replacement from that fold's own chronological series (see
    `_moving_block_resample`), preserving local month-to-month dependence instead of treating
    months as exchangeable or (as an earlier version of this function did) resampling whole fold
    identities. The resampled per-fold series are then pooled across folds - every month
    contributes equally to one bootstrap replicate, matching the estimand this project reports
    (delta-hat = the mean over ALL fold-months, not a fold-weighted average) - repeated
    `n_resamples` times to build the bootstrap distribution.

    `block_length` defaults to 3 as a PROVISIONAL placeholder. evaluation_framework.md specifies
    this should be "fixed using development data before final-test results are examined" - that
    calibration pass hasn't been done yet (tracked as upcoming work in docs/EVALUATION.md), so
    treat this default as not yet validated rather than a deliberate choice.

    Null-centering ("basic"/reflected bootstrap): the raw bootstrap distribution of delta*_b is
    centered near the OBSERVED delta, not zero, so it can't be read directly against a delta=0
    null. Instead it's reflected around the observed delta to approximate what the estimator
    would look like if the true effect were exactly zero:
      - one-sided 95% lower bound = 2*observed_delta - the 95th percentile of {delta*_b}
      - one-sided p-value = fraction of replicates with delta*_b >= 2*observed_delta
    (equivalently: shift {delta*_b - observed_delta} to be centered at 0, then ask how often that
    shifted distribution reaches or exceeds the observed delta.)

    min_folds guards against a degenerate bootstrap: with very few distinct final-test folds, a
    large fraction of resamples can draw mostly from the same one or two folds, making the
    bootstrap distribution artificially narrow and producing an exact-looking p-value from very
    little real data. Below min_folds, the resampling loop is skipped entirely and
    lower_bound_95/p_value come back as NaN with insufficient_folds=True - same dict shape (every
    key still present), so callers only need to notice the new flag, not handle a different shape.
    """
    fold_names = list(fold_month_deltas.keys())
    all_deltas = [d for deltas in fold_month_deltas.values() for d in deltas]
    observed_delta = float(np.mean(all_deltas)) if all_deltas else float('nan')

    if len(fold_names) < min_folds:
        return {
            "observed_delta": observed_delta,
            "lower_bound_95": float('nan'),
            "p_value": float('nan'),
            "n_resamples": n_resamples,
            "block_length": block_length,
            "insufficient_folds": True,
        }

    rng = random.Random(seed)
    boot_deltas = []
    for _ in range(n_resamples):
        pooled = []
        for fold_name in fold_names:
            pooled.extend(_moving_block_resample(fold_month_deltas[fold_name], block_length, rng))
        boot_deltas.append(float(np.mean(pooled)) if pooled else float('nan'))
    boot_deltas = np.array(boot_deltas)

    lower_bound_95 = float(2 * observed_delta - np.nanpercentile(boot_deltas, 95))
    p_value = float(np.mean(boot_deltas >= 2 * observed_delta))

    return {
        "observed_delta": observed_delta,
        "lower_bound_95": lower_bound_95,
        "p_value": p_value,
        "n_resamples": n_resamples,
        "block_length": block_length,
        "insufficient_folds": False,
    }
