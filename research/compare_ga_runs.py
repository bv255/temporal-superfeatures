"""
Package-backed entry point for the temporal-on vs temporal-off head-to-head comparison,
mechanically extracted from `compare_metrics.ipynb`'s 9 code cells (cell indices noted per
function below) - the notebook stays untouched as the parity reference, same convention as
`run_preprocessing.py`/`run_ga.py`. Unlike those two scripts, there's no `superfeatures.compare`
package module (yet) - this notebook never got that port, so the logic lives directly in this
script, same structure as the notebook itself.

The notebook's own variables/labels were "v3"/"v2" (the retired `GA_test.ipynb`/`GA_test_v2.ipynb`
notebook-pair naming - see CLAUDE.md's "Package port" section on why that pair no longer exists).
Renamed throughout to `temporal`/`no_temporal` here, matching `GAConfig.enable_temporal_operators`
- the actual axis being compared, not which notebook a run happened to come from.

One real fix, not just a rename: the notebook's cell 5 confirmed `max_features` matched between
runs by regex-reading `crossover_deep`'s hardcoded literal out of TWO SEPARATE notebook files'
cell 19 source - that assumption doesn't hold anymore, since both a temporal-on and temporal-off
run now come from the SAME `run_ga.py`/`GAConfig`, just with different flags. `run_ga.py` now
writes a `run_config.json` (`dataclasses.asdict(config)`) into its own output_dir for exactly
this - this script reads `max_features` back from both runs' `run_config.json` instead.

REDESIGNED (package-only, notebook left frozen) to match evaluation_framework.md's pre-registered
evaluation spec - see docs/EVALUATION.md for the full description. In brief:
  - The primary hypothesis (temporal-ON vs temporal-OFF) is tested standalone, NOT folded into a
    Holm-Bonferroni family with the baseline checks - it's the single pre-specified primary
    hypothesis, so no multiple-comparison correction applies to it.
  - The "GA winner vs its baselines" secondary family now lives per-arm, in each run's own
    pairwise_comparisons.csv (analysis/summary.py, computed at run_ga.py time) - this script no
    longer recomputes no_temporal-vs-A/B/C or temporal-vs-C itself.
  - The old four-condition H1 verdict (magnitude vs. a Delta minimum-detectable-effect threshold,
    statistical reliability, fold-consistency majority, DiD attribution) is retired. H1 is now
    supported purely by delta-hat > 0 and the one-sided 95% lower bound > 0 (p < 0.05). Delta/MDE
    is gone entirely; fold-consistency and the DiD attribution test are still reported, but as
    descriptive/mechanism diagnostics only - they no longer gate the verdict.
  - evaluation_framework.md's primary estimand is defined over 10 pre-specified MATCHED SEEDS per
    arm (average the paired monthly delta across seeds before aggregating across fold-months).
    `--seeds N N N` now implements this (build_primary_comparison_seed_averaged /
    seed_averaged_fold_month_deltas / main_multi_seed below) - it runs on however many seeds are
    passed in, not necessarily all 10 of the pre-registered set, and every seed-averaged output
    records `seed_averaging_applied=True` and `n_seeds` so a partial-seed average is never
    confused with the full pre-registered one. The single-seed/no-`--seeds` path (`main`,
    `build_primary_comparison`) is unchanged and still writes `seed_averaging_applied=False`.
  - `--seeds` also writes seed-level Rank IC (`seed_level_rank_ic_by_fold.csv`/
    `seed_level_rank_ic.csv`), seed-averaged GA-vs-baseline comparisons
    (`seed_averaged_baseline_comparisons.csv`), seed-averaged DiD (`seed_averaged_did.csv`), and an
    all-predictors-matched comparison (`predictor_comparison_matched.csv`) - see the "Three Rank IC
    estimands" comment block above `holm_bonferroni` below for what distinguishes these from
    `seed_averaged_aggregate_metrics.csv`'s numbers (three different, deliberately-not-equal
    quantities, not redundant copies of the same one).
  - `--seeds` also writes a SECONDARY robustness check alongside the primary seed-averaged test:
    `secondary_seed_block_bootstrap.csv` (`build_secondary_seed_block_bootstrap` /
    `seed_and_block_bootstrap_ic_delta`) resamples the matched GA seeds themselves, with
    replacement, in addition to the primary bootstrap's within-fold month blocks - the primary test
    treats the 10 matched seeds as fixed and only bootstraps temporal uncertainty; this one reflects
    GA-seed uncertainty too. Same point estimate (observed_delta) as the primary test, its own
    lower_bound_95/p_value, and `secondary_seed_block_bootstrap_null_distribution.csv` (the full
    null-centered resampling distribution, one row per bootstrap replicate) for a one-sided test
    consistent with the primary analysis. Does not gate the H1 verdict (descriptive/robustness only,
    same status as fold-consistency and DiD attribution).

Three flags (--fast, --seed N, --seeds N N N), matching run_preprocessing.py/run_ga.py's own
convention - no arbitrary --output-dir pair, since the pair(s) to compare are always derived the
same way run_ga.py itself names its output_dir (base +_no_temporal +_seedN). --seed and --seeds
are mutually exclusive. Every comparison writes under research/comparison_outputs/<label>/ (see
the __main__ block below for the exact label rules) - "default" (no flags), "fast" (--fast alone),
"seedN" (--seed N alone), "fast_seedN" (both), "seeds_N_N_N" (--seeds N N N), "fast_seeds_N_N_N"
(both) - the last two also nest each seed's own per-seed diagnostics under a `seedN/` subdirectory.
.gitignore's research/comparison_outputs/fast*/ pattern keeps fast-scale labels out of git
(regenerable) while full-scale labels stay tracked as committed research artifacts.

Run both matching run_ga.py invocations first (e.g. `run_ga.py --fast` and
`run_ga.py --fast --no-temporal-operators`) before running this - same "needs a live cluster-
shaped run" caveat as run_ga.py itself. For --seeds, run both invocations (temporal and
no_temporal) for every seed first.

Usage: ~/pyspark-venv/bin/python3 compare_ga_runs.py
       ~/pyspark-venv/bin/python3 compare_ga_runs.py --fast
       ~/pyspark-venv/bin/python3 compare_ga_runs.py --seed 7
       ~/pyspark-venv/bin/python3 compare_ga_runs.py --fast --seed 7
       ~/pyspark-venv/bin/python3 compare_ga_runs.py --seeds 10 11 12
       ~/pyspark-venv/bin/python3 compare_ga_runs.py --fast --seeds 10 11 12
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------------------------
# Shared helpers - cell 3, duplicated (not imported) from ga/methodology.py's equivalents,
# same rationale the notebook's own cell 3 docstring gave: keeps this script independently
# runnable without depending on run_ga.py's internals, and both are pure functions with no
# Spark/package-specific state, safe to duplicate exactly.
# ---------------------------------------------------------------------------------------------
import random


def _moving_block_resample(values: list, block_length: int, rng: random.Random) -> list:
    """
    One fold's contribution to one bootstrap replicate - see block_bootstrap_ic_delta below.
    Duplicated verbatim from analysis/significance.py (same "independently runnable, no
    package-internals dependency" rationale documented at the top of this file).
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
    Dependence-preserving block bootstrap per evaluation_framework.md - duplicated verbatim from
    analysis/significance.py's own copy (see that module's docstring for the full derivation).
    ONE-SIDED and NULL-CENTERED throughout (every comparison this script computes - the primary
    temporal-vs-no_temporal test and the DiD attribution diagnostic - is directional).

    `fold_month_deltas` is {fold_name: [delta_1, delta_2, ...]}, one CHRONOLOGICALLY ORDERED list
    of month-level deltas per final-test fold. Resampling draws contiguous `block_length`-month
    blocks with replacement from each fold's own series independently, pools the resampled series
    across folds (every month contributes equally - matches the delta-hat estimand, a plain mean
    over all fold-months), and repeats `n_resamples` times.

    `block_length` defaults to 3 as a PROVISIONAL placeholder - not yet calibrated against
    development data (evaluation_framework.md calls for that; tracked as upcoming work).

    Null-centering ("basic"/reflected bootstrap): the raw bootstrap distribution of delta*_b is
    centered near the OBSERVED delta, not zero. Reflecting it around the observed delta gives:
    - lower_bound_95 = 2*observed_delta - the 95th percentile of {delta*_b}
    - p_value = fraction of replicates with delta*_b >= 2*observed_delta

    min_folds guards against a degenerate bootstrap at very low final-test fold counts - below
    it, lower_bound_95/p_value come back NaN with insufficient_folds=True.
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


def matched_month_deltas_cross_variant(results_a: dict, results_b: dict, key: str, folds) -> dict:
    """
    Same idea as matched_month_deltas, but key comes from the SAME field name (e.g.
    "winner_monthly_ic") read out of two DIFFERENT result trees (temporal's vs no_temporal's) -
    this is what the "temporal vs no_temporal" comparison needs, since there's no single
    fold_result.json that has both.
    """
    fold_month_deltas = {}
    for fold_name in folds:
        ic_a = results_a.get(fold_name, {}).get(key)
        ic_b = results_b.get(fold_name, {}).get(key)
        if not ic_a or not ic_b:
            continue
        series_a, series_b = pd.Series(ic_a), pd.Series(ic_b)
        matched_dates = series_a.index.intersection(series_b.index).sort_values()
        if len(matched_dates) == 0:
            continue
        fold_month_deltas[fold_name] = (series_a[matched_dates] - series_b[matched_dates]).tolist()
    return fold_month_deltas


def seed_averaged_fold_month_deltas(temporal_results_by_seed: dict, no_temporal_results_by_seed: dict,
                                     key: str, folds, seeds: list) -> dict:
    """
    Per evaluation_framework.md: "The primary analysis averages paired ON-OFF differences across
    the ... matched seeds" - for each (fold, month), average the temporal-minus-no_temporal delta
    across `seeds` FIRST, producing one delta series per fold, then that series is what
    block_bootstrap_ic_delta bootstraps (same downstream test as the single-seed primary
    comparison, just fed a seed-averaged series). This is what makes it a paired comparison across
    seeds rather than pooling 3x as many raw fold-months into one bootstrap.

    Within a fold, a date must be present for a given seed's temporal AND no_temporal series to
    count for that seed (same per-seed matching matched_month_deltas_cross_variant does); the
    per-fold average is then taken across whichever seeds have that date, and a fold is dropped
    entirely if no seed has any usable date. Seeds/folds with partial coverage print a WARNING
    rather than silently averaging over fewer seeds than the caller asked for.
    """
    fold_month_deltas = {}
    for fold_name in folds:
        per_seed_series = {}
        for seed in seeds:
            t_res = temporal_results_by_seed[seed].get(fold_name, {})
            n_res = no_temporal_results_by_seed[seed].get(fold_name, {})
            ic_t, ic_n = t_res.get(key), n_res.get(key)
            if not ic_t or not ic_n:
                continue
            st, sn = pd.Series(ic_t), pd.Series(ic_n)
            dates = st.index.intersection(sn.index)
            if len(dates) == 0:
                continue
            per_seed_series[seed] = st[dates] - sn[dates]

        missing = [s for s in seeds if s not in per_seed_series]
        if missing:
            print(f"WARNING: fold {fold_name} has no matched (ON, OFF) monthly IC for seed(s) "
                  f"{missing} - averaged over the remaining {len(per_seed_series)}/{len(seeds)} "
                  f"seed(s) for this fold instead.")
        if not per_seed_series:
            continue

        combined = pd.concat(per_seed_series, axis=1)
        partial_dates = combined.isna().any(axis=1) & combined.notna().any(axis=1)
        if partial_dates.any():
            print(f"WARNING: fold {fold_name} has {int(partial_dates.sum())} month(s) where only "
                  f"some seeds have a matched IC - averaged over whichever seeds are present per "
                  f"month rather than dropped.")
        averaged = combined.mean(axis=1, skipna=True).sort_index()
        fold_month_deltas[fold_name] = averaged.tolist()
    return fold_month_deltas


def build_per_seed_fold_deltas(temporal_results_by_seed: dict, no_temporal_results_by_seed: dict,
                                key: str, folds, seeds: list) -> dict:
    """
    Same per-(fold, seed) matching seed_averaged_fold_month_deltas does internally (a date counts
    for a seed only if BOTH arms have a value there), just returned BEFORE collapsing across
    seeds - {fold_name: {seed: pd.Series}}, each series date-indexed. This is the input
    seed_and_block_bootstrap_ic_delta needs, since it has to resample the seed axis itself rather
    than working from an already seed-averaged series.
    """
    per_seed_fold_deltas = {}
    for fold_name in folds:
        per_seed_fold_deltas[fold_name] = {}
        for s in seeds:
            t_res = temporal_results_by_seed[s].get(fold_name, {})
            n_res = no_temporal_results_by_seed[s].get(fold_name, {})
            ic_t, ic_n = t_res.get(key), n_res.get(key)
            if not ic_t or not ic_n:
                continue
            st, sn = pd.Series(ic_t), pd.Series(ic_n)
            dates = st.index.intersection(sn.index)
            if len(dates) == 0:
                continue
            per_seed_fold_deltas[fold_name][s] = (st[dates] - sn[dates]).sort_index()
    return per_seed_fold_deltas


def seed_and_block_bootstrap_ic_delta(per_seed_fold_deltas: dict, seeds: list, block_length: int = 3,
                                       n_resamples: int = 3000, seed: int = 42, min_folds: int = 5) -> dict:
    """
    Secondary robustness check on top of the primary seed-averaged analysis
    (seed_averaged_fold_month_deltas + block_bootstrap_ic_delta): that primary analysis treats the
    matched GA seeds as FIXED and only bootstraps temporal (within-fold month-block) uncertainty.
    This adds a second resampling axis - the GA seeds themselves - with replacement, so the
    resulting interval/p-value reflects both temporal AND stochastic GA-seed uncertainty, matching
    what evaluation_framework.md's "10 pre-specified matched seeds" pairing is actually meant to
    guard against (a result that only looks robust because it happens to be averaged over this
    particular set of 10 seeds).

    `per_seed_fold_deltas` is {fold_name: {seed: pd.Series}} from build_per_seed_fold_deltas - one
    date-indexed (temporal - no_temporal) monthly IC delta series per (fold, seed) pair, not yet
    collapsed across seeds.

    Each bootstrap replicate:
      1. Resample len(seeds) seeds WITH replacement from `seeds` (a seed drawn twice contributes
         its full paired ON-OFF trajectory twice below - not deduplicated).
      2. Within each fold, align the resampled seeds' delta series on the date union and average
         them (skipna) - the fold's seed-resampled monthly delta series, built the same way the
         primary analysis's per-fold average is, just over this replicate's resampled multiset
         instead of the fixed full set.
      3. Resample contiguous `block_length`-month blocks with replacement from that averaged series
         (same `_moving_block_resample` primitive/block_length as the primary bootstrap - one
         shared block draw per fold per replicate, applied to the already seed-averaged series).
         Averaging over seeds and resampling-by-position commute (a fixed position sequence applied
         to an average equals averaging that same fixed position drawn from each seed), so this is
         equivalent to drawing one month sequence per fold and applying it identically to every
         resampled seed before averaging, without the extra bookkeeping - "the same resampled month
         sequence for every sampled seed within a fold" falls out for free. Blocks never cross fold
         boundaries, same as the primary bootstrap.
      4. Pool the block-resampled series across folds (every month weighted equally, matching the
         delta-hat estimand this project reports) and take the mean -> one bootstrap draw delta*_b.

    Repeating `n_resamples` times gives a distribution reflecting both temporal and GA-seed
    resampling uncertainty. Reported the same null-centered/reflected way as block_bootstrap_ic_delta:
      - lower_bound_95 = 2*observed_delta - 95th percentile of {delta*_b}
      - p_value = fraction of delta*_b >= 2*observed_delta
    `null_deltas` is the reflected/null-centered distribution itself ({delta*_b - observed_delta}) -
    the resampling distribution under delta=0 - for callers that want the full null curve rather
    than just its tail probability at the observed delta.

    observed_delta is computed WITHOUT seed-resampling (averaged across the full fixed `seeds` list,
    same construction as seed_averaged_fold_month_deltas's point estimate) - only the bootstrap
    replicates resample seeds; the point estimate itself is unchanged from the primary analysis, so
    this secondary bootstrap's interval/p-value are directly comparable to the primary one's.

    min_folds mirrors block_bootstrap_ic_delta's guard: below it the loop is skipped and
    lower_bound_95/p_value come back NaN (boot_deltas/null_deltas empty) with insufficient_folds=True.
    """
    fold_names = list(per_seed_fold_deltas.keys())

    def _fold_average(fold_name: str, seed_list: list) -> pd.Series:
        cols = {i: per_seed_fold_deltas[fold_name][s] for i, s in enumerate(seed_list)
                if s in per_seed_fold_deltas[fold_name]}
        if not cols:
            return pd.Series(dtype=float)
        combined = pd.concat(cols, axis=1)
        return combined.mean(axis=1, skipna=True).sort_index()

    observed_series = {f: _fold_average(f, seeds) for f in fold_names}
    all_observed = [v for s in observed_series.values() for v in s.tolist()]
    observed_delta = float(np.mean(all_observed)) if all_observed else float('nan')

    if len(fold_names) < min_folds:
        return {
            "observed_delta": observed_delta,
            "lower_bound_95": float('nan'),
            "p_value": float('nan'),
            "n_resamples": n_resamples,
            "block_length": block_length,
            "n_seeds": len(seeds),
            "insufficient_folds": True,
            "boot_deltas": [],
            "null_deltas": [],
        }

    rng_seeds = random.Random(seed)
    rng_blocks = random.Random(seed + 1)
    boot_deltas = []
    for _ in range(n_resamples):
        resampled_seeds = [rng_seeds.choice(seeds) for _ in range(len(seeds))]
        pooled = []
        for fold_name in fold_names:
            avg_values = _fold_average(fold_name, resampled_seeds).tolist()
            pooled.extend(_moving_block_resample(avg_values, block_length, rng_blocks))
        boot_deltas.append(float(np.mean(pooled)) if pooled else float('nan'))
    boot_deltas = np.array(boot_deltas)

    lower_bound_95 = float(2 * observed_delta - np.nanpercentile(boot_deltas, 95))
    p_value = float(np.mean(boot_deltas >= 2 * observed_delta))
    null_deltas = boot_deltas - observed_delta

    return {
        "observed_delta": observed_delta,
        "lower_bound_95": lower_bound_95,
        "p_value": p_value,
        "n_resamples": n_resamples,
        "block_length": block_length,
        "n_seeds": len(seeds),
        "insufficient_folds": False,
        "boot_deltas": boot_deltas.tolist(),
        "null_deltas": null_deltas.tolist(),
    }


def seed_averaged_monthly_ic(temporal_results_by_seed: dict, no_temporal_results_by_seed: dict,
                              key: str, folds, seeds: list) -> tuple:
    """
    Per-arm counterpart to seed_averaged_fold_month_deltas: averages EACH arm's own monthly IC
    series across seeds, per fold-month - the \\bar{IC}_{a,f,t} construction in
    evaluation_framework.md's Aggregation section (§sec:aggregation), used to report each arm's
    own seed-averaged Rank IC/IC-IR level rather than just the ON-OFF delta
    build_primary_comparison_seed_averaged computes.

    FIXED (previously took a single arm's results_by_seed and averaged it in isolation, with no
    cross-arm date matching): a (fold, seed, date) triple now counts for EITHER arm's average only
    if BOTH arms have a value there, using the exact same matched-dates logic
    seed_averaged_fold_month_deltas already applies for the delta itself. Before this fix, the two
    arms' returned series could be built from different underlying date sets (confirmed on the
    100-109 seed set: temporal had 113 months, no_temporal had 115 - 2 extra months no_temporal
    alone had a value for), which meant mean(ON) - mean(OFF) computed from
    build_seed_averaged_aggregate_metrics did NOT equal build_primary_comparison_seed_averaged's
    observed_delta despite this module's own (previously incorrect) docstring elsewhere claiming
    they were "algebraically equal" - the two were silently pooling different month sets. With
    matched dates enforced here the same way the delta computation already enforces them, the
    identity now holds exactly (verified: 0.035142 - 0.035003 = 0.000139, matching
    primary_comparison_seed_averaged.csv's observed_delta to 6 decimal places).

    Returns (temporal_fold_month_ic, no_temporal_fold_month_ic), each {fold_name: pd.Series}
    (index = date, chronologically ordered).
    """
    temporal_fold_month_ic, no_temporal_fold_month_ic = {}, {}
    for fold_name in folds:
        t_series, n_series = {}, {}
        for seed in seeds:
            ic_t = temporal_results_by_seed[seed].get(fold_name, {}).get(key)
            ic_n = no_temporal_results_by_seed[seed].get(fold_name, {}).get(key)
            if not ic_t or not ic_n:
                continue
            st, sn = pd.Series(ic_t), pd.Series(ic_n)
            dates = st.index.intersection(sn.index)
            if len(dates) == 0:
                continue
            t_series[seed] = st[dates]
            n_series[seed] = sn[dates]
        if not t_series:
            continue
        temporal_fold_month_ic[fold_name] = pd.concat(t_series, axis=1).mean(axis=1, skipna=True).sort_index()
        no_temporal_fold_month_ic[fold_name] = pd.concat(n_series, axis=1).mean(axis=1, skipna=True).sort_index()
    return temporal_fold_month_ic, no_temporal_fold_month_ic


def matched_month_lift_deltas(temporal_results: dict, no_temporal_results: dict, folds,
                               winner_key: str = "winner_monthly_ic",
                               baseline_key: str = "baseline_c_monthly_ic") -> dict:
    """
    The difference-in-differences attribution test: lift_temporal - lift_no_temporal, where
    lift = (real GA winner's IC - baseline C's IC) within a variant. Per matched fold:
    temporal_lift[month] = temporal's winner IC - temporal's baseline-C IC (months common to
    temporal's own two series), no_temporal_lift[month] the same way within no_temporal's own
    series, then DiD[month] = temporal_lift[month] - no_temporal_lift[month] over months common
    to BOTH lift series. Feeds the same block_bootstrap_ic_delta machinery as every other
    comparison. This isolates whether an IC edge is attributable to the temporal operators
    specifically rather than just a larger search space (baseline C is a matched-budget random
    search within each variant's own leaf vocabulary).
    """
    fold_month_deltas = {}
    for fold_name in folds:
        tr, nr = temporal_results.get(fold_name, {}), no_temporal_results.get(fold_name, {})
        t_winner, t_baseline = tr.get(winner_key), tr.get(baseline_key)
        n_winner, n_baseline = nr.get(winner_key), nr.get(baseline_key)
        if not t_winner or not t_baseline or not n_winner or not n_baseline:
            continue

        t_winner_s, t_baseline_s = pd.Series(t_winner), pd.Series(t_baseline)
        t_dates = t_winner_s.index.intersection(t_baseline_s.index)
        t_lift = t_winner_s[t_dates] - t_baseline_s[t_dates]

        n_winner_s, n_baseline_s = pd.Series(n_winner), pd.Series(n_baseline)
        n_dates = n_winner_s.index.intersection(n_baseline_s.index)
        n_lift = n_winner_s[n_dates] - n_baseline_s[n_dates]

        matched_dates = t_lift.index.intersection(n_lift.index).sort_values()
        if len(matched_dates) == 0:
            continue
        fold_month_deltas[fold_name] = (t_lift[matched_dates] - n_lift[matched_dates]).tolist()
    return fold_month_deltas


# ---------------------------------------------------------------------------------------------
# Three "Rank IC" estimands - added 2026-08-24 after a round of external review found several
# reported Rank IC figures across different tables/figures didn't reconcile. THREE genuinely
# different quantities are computed across this file, distinguished by WHICH held-out months get
# pooled and how they're weighted - not by any modeling difference. Every function below is
# tagged with which one it computes. See CLAUDE.md's "Three Rank IC estimands" section for the
# full narrative (including the two real bugs this surfaced, both fixed).
#
# ESTIMAND I - canonical Rank IC_a (evaluation_framework.md's own primary estimand): a held-out
#   month only counts if BOTH arms (temporal and no_temporal) have a value there, for that seed.
#   Pooled across all matched seeds/folds, every month weighted equally. This is the one number
#   the primary hypothesis test (delta-hat) is built on - see seed_averaged_fold_month_deltas/
#   seed_averaged_monthly_ic/build_primary_comparison_seed_averaged/
#   build_seed_averaged_aggregate_metrics above. Used by: primary_comparison_seed_averaged.csv,
#   seed_averaged_aggregate_metrics.csv (paper's Tables 4.1/4.2/4.3/4.4, Figure 4.1).
#
# ESTIMAND II - seed-level Rank IC, RankIC_{a,s} (seed_level_monthly_ic/build_seed_level_rank_ic
#   below): computed independently WITHIN one arm - no cross-arm matching at all - specifically to
#   preserve each seed's own individual value rather than pooling all 10 seeds into one number.
#   RankIC_{a,s} = that seed's own held-out months, pooled across its folds, weighted equally by
#   month. Its cross-seed MEAN is not expected to equal Estimand I's Rank IC_a - different
#   constructions answering different questions ("how much does this vary by seed" vs "what's the
#   pooled effect"). Used by: seed_level_rank_ic.csv, seed_level_rank_ic_by_fold.csv (paper's
#   Tables 4.5/4.6, Figure 4.2).
#
# ESTIMAND III - predictor-matched Rank IC (seed_averaged_within_arm_matched/
#   seed_averaged_within_arm_matched_multi below): WITHIN one arm (no cross-arm matching), but a
#   month only counts if EVERY predictor being compared in that table has a value there. Two
#   flavors: 2-way (GA matched against exactly one baseline at a time - Table 4.7's per-baseline
#   rows, Table 4.8's DiD, which only ever involves GA and Random) and N-way (GA matched against
#   ALL baselines at once, for a table/figure that displays every predictor from one shared date
#   set - Figure 4.3). GA's own reported value can differ slightly between a 2-way and an N-way
#   table, purely because a different set of months survives the intersection - this is expected,
#   not an error. Used by: seed_averaged_baseline_comparisons.csv, seed_averaged_did.csv,
#   predictor_comparison_matched.csv (paper's Tables 4.7/4.8, Figure 4.3).
# ---------------------------------------------------------------------------------------------

def holm_bonferroni(pvalues: list) -> list:
    """
    Manual Holm-Bonferroni step-down correction, duplicated verbatim from
    analysis/significance.py (same "independently runnable, no package-internals dependency"
    rationale documented at the top of this file for block_bootstrap_ic_delta/
    _moving_block_resample). Returns adjusted p-values in the SAME order as the input list.
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


def seed_level_monthly_ic(results_by_seed: dict, key: str, folds, seeds: list) -> dict:
    """
    ESTIMAND II, raw data: {seed: {fold_name: pd.Series}} - one arm's own monthly IC per seed per
    fold, completely unmatched against the other arm (unlike seed_averaged_monthly_ic above, there
    is no seed-averaging here either - every seed's own series is kept separate, which is the
    entire point of this estimand).
    """
    out = {}
    for seed in seeds:
        seed_folds = {}
        for fold_name in folds:
            monthly_ic = results_by_seed[seed].get(fold_name, {}).get(key)
            if monthly_ic:
                seed_folds[fold_name] = pd.Series(monthly_ic)
        out[seed] = seed_folds
    return out


def seed_level_rank_ic_by_fold(results_by_seed: dict, folds, seeds: list,
                                key: str = "winner_monthly_ic") -> pd.DataFrame:
    """
    ESTIMAND II per fold (paper's Table 4.5): RankIC_{a,s,f} - one value per (seed, fold), the
    plain mean of that seed's own held-out months within that single fold. No cross-fold pooling
    (each fold already has one unambiguous month set, so there's no weighting choice at this
    granularity - that only arises once folds get combined, see seed_level_rank_ic below).
    Returns a long DataFrame: columns seed, fold_name, rank_ic.
    """
    fold_month_ic = seed_level_monthly_ic(results_by_seed, key, folds, seeds)
    rows = []
    for seed in seeds:
        for fold_name in folds:
            series = fold_month_ic[seed].get(fold_name)
            if series is None or len(series) == 0:
                continue
            rows.append({"seed": seed, "fold_name": fold_name, "rank_ic": float(series.mean()),
                         "n_months": len(series)})
    return pd.DataFrame(rows)


def seed_level_rank_ic(results_by_seed: dict, folds, seeds: list,
                        key: str = "winner_monthly_ic") -> pd.DataFrame:
    """
    ESTIMAND II pooled across folds (paper's Table 4.6): RankIC_{a,s} - one value per seed, that
    seed's own held-out months POOLED across all its folds and weighted equally by month (NOT an
    average of the per-fold means from seed_level_rank_ic_by_fold above - folds hold out different
    numbers of months in this study, 18-24, so pooling months directly vs. averaging fold-means
    equally are different statistics; this function does the former, matching the paper's stated
    definition). Returns a long DataFrame: columns seed, rank_ic, n_months.
    """
    fold_month_ic = seed_level_monthly_ic(results_by_seed, key, folds, seeds)
    rows = []
    for seed in seeds:
        all_months = [v for series in fold_month_ic[seed].values() for v in series.tolist()]
        if not all_months:
            continue
        rows.append({"seed": seed, "rank_ic": float(np.mean(all_months)), "n_months": len(all_months)})
    return pd.DataFrame(rows)


def seed_averaged_within_arm_matched(results_by_seed: dict, key_a: str, key_b: str,
                                      folds, seeds: list) -> dict:
    """
    ESTIMAND III, 2-way engine: WITHIN one arm, a (fold, seed, month) triple only counts if BOTH
    key_a and key_b (e.g. "winner_monthly_ic" and "baseline_c_monthly_ic") have a value there -
    no cross-arm matching. Seed-averages each of the two matched series per fold-month, same
    convention as seed_averaged_fold_month_deltas (Estimand I) but applied within one arm against
    a baseline instead of across arms. Powers both Table 4.7 (GA vs each baseline, one baseline at
    a time) and Table 4.8 (GA vs Random specifically - DiD never involves any other baseline, so
    it must use this 2-way engine, not the N-way one below - see CLAUDE.md for the bug this
    distinction fixed).

    Returns (fold_month_a, fold_month_b), each {fold_name: pd.Series}.
    """
    fold_month_a, fold_month_b = {}, {}
    for fold_name in folds:
        sa, sb = {}, {}
        for seed in seeds:
            r = results_by_seed[seed].get(fold_name, {})
            ic_a, ic_b = r.get(key_a), r.get(key_b)
            if not ic_a or not ic_b:
                continue
            s1, s2 = pd.Series(ic_a), pd.Series(ic_b)
            dates = s1.index.intersection(s2.index)
            if len(dates) == 0:
                continue
            sa[seed] = s1[dates]
            sb[seed] = s2[dates]
        if not sa:
            continue
        fold_month_a[fold_name] = pd.concat(sa, axis=1).mean(axis=1, skipna=True)
        fold_month_b[fold_name] = pd.concat(sb, axis=1).mean(axis=1, skipna=True)
    return fold_month_a, fold_month_b


def seed_averaged_within_arm_matched_multi(results_by_seed: dict, keys: dict,
                                            folds, seeds: list) -> dict:
    """
    ESTIMAND III, N-way engine: WITHIN one arm, a (fold, seed, month) triple only counts if EVERY
    key in `keys` ({name: monthly_ic field name}, e.g. {"GA": "winner_monthly_ic", "Fixed":
    "baseline_a_monthly_ic", ...}) has a value there simultaneously - the strictest of the three
    estimands, since every extra predictor added shrinks the matched date set further. Powers
    Figure 4.3 (all four predictors matched to one common set, per arm) - NOT Table 4.7/4.8, which
    only ever compare two predictors at a time and should use the 2-way engine above instead (a
    real bug found and fixed during this project's writeup: an earlier version of Table 4.8
    incorrectly borrowed this N-way engine even though DiD never involves Fixed/Raw).

    Returns {name: {fold_name: pd.Series}}.
    """
    pooled = {name: {} for name in keys}
    for fold_name in folds:
        per_seed = {name: {} for name in keys}
        for seed in seeds:
            r = results_by_seed[seed].get(fold_name, {})
            series = {name: r.get(key) for name, key in keys.items()}
            if not all(series.values()):
                continue
            s = {name: pd.Series(v) for name, v in series.items()}
            dates = None
            for name in keys:
                dates = s[name].index if dates is None else dates.intersection(s[name].index)
            if dates is None or len(dates) == 0:
                continue
            for name in keys:
                per_seed[name][seed] = s[name][dates]
        if not per_seed[next(iter(keys))]:
            continue
        for name in keys:
            pooled[name][fold_name] = pd.concat(per_seed[name], axis=1).mean(axis=1, skipna=True)
    return pooled


# ---------------------------------------------------------------------------------------------
# Loading fold_result.json trees - cell 1
# ---------------------------------------------------------------------------------------------

def load_final_test_fold_results(output_dir: str) -> dict:
    """
    {fold_name: fold_result.json content}, final-test folds only - the scope every comparison
    below is limited to (development folds are for debugging/iteration, not reported).
    """
    results = {}
    for path in sorted(glob.glob(f"{output_dir}/final_test/fold_*/fold_result.json")):
        with open(path) as f:
            result = json.load(f)
        results[result["fold_name"]] = result
    return results


def load_run_config(output_dir: str) -> dict:
    """run_ga.py writes this once per run (see its main()) - returns {} if it's missing (e.g. a
    run from before this was added), so callers can degrade gracefully instead of crashing."""
    path = f"{output_dir}/run_config.json"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def comparison_output_label(fast: bool, fitness_metric: str = None, seeds: list = None,
                             seed: int = None) -> str:
    """
    comparison_outputs/<label>/ label built from whichever flags are active, rather than one
    hardcoded string per flag combination - keeps every flag (including --fitness-metric) from
    needing its own branch. Matches run_output_dir's own suffix order (no_temporal is folded into
    main()/main_multi_seed() separately, not part of this label) so e.g. fast=True,
    fitness_metric="rank_ic", seeds=[10, 11, 12] gives 'fast_rank_ic_seeds_10_11_12'.
    .gitignore's research/comparison_outputs/fast*/ pattern keeps every fast-scale label
    (regenerable) out of git while full-scale labels stay tracked as committed research artifacts.

    Extracted as its own function (2026-08-25) specifically so callers outside this file (e.g.
    docs/make_figures.py) can compute the exact same label instead of re-deriving it by hand and
    risking drift from whatever this script actually writes to.
    """
    label_parts = []
    if fast:
        label_parts.append("fast")
    if fitness_metric == "rank_ic":
        label_parts.append("rank_ic")
    if seeds is not None:
        label_parts.append("seeds_" + "_".join(str(s) for s in seeds))
    elif seed is not None:
        label_parts.append(f"seed{seed}")
    return "_".join(label_parts) if label_parts else "default"


def run_output_dir(base: str, no_temporal: bool, fitness_metric: str = None, seed: int = None, local: bool = False) -> str:
    """
    Reproduces run_ga.py's own output_dir suffix logic - which changed on 2026-08-24 (see
    CLAUDE.md's "Other flags" section and run_ga.py's own --fit-backend/--fitness-metric help
    text): before that date, GAConfig defaulted to fit_backend="spark"/fitness_metric="rmse", and
    run_ga.py's own --fit-backend local/--fitness-metric rank_ic flags each appended a
    '_local'/'_rank_ic' suffix (this is why the seed7/10-13 runs are named
    'ga_rank_ic_seed10' etc). As of 2026-08-24, run_ga.py's OWN defaults flipped to
    fit_backend="local"/fitness_metric="rank_ic" with NO output_dir suffix for either ("this
    script's default ... no output_dir suffix either way" - run_ga.py's own --fitness-metric help
    text) - so newer runs (seed100+) are named 'ga_seed100' with no '_rank_ic'/'_local' despite
    genuinely using rank_ic/local. Passing fitness_metric="rank_ic"/local=True here is therefore
    only correct for a LEGACY pre-2026-08-24 run; for a current run, pass neither and the fact
    that it's rank_ic/local is implicit.

    To avoid silently pointing at the wrong (nonexistent, or worse, a stale same-named-without-
    suffix) directory when a caller gets this wrong, this checks both the naively-constructed path
    and the un-suffixed fallback, relative to cwd (callers always run this from research/), and
    warns loudly if it had to fall back or if neither exists.
    """
    output_dir = base
    if local:
        output_dir += "_local"
    if no_temporal:
        output_dir += "_no_temporal"
    if fitness_metric == "rank_ic":
        output_dir += "_rank_ic"
    if seed is not None:
        output_dir += f"_seed{seed}"

    if (local or fitness_metric == "rank_ic") and not os.path.isdir(output_dir):
        fallback = base
        if no_temporal:
            fallback += "_no_temporal"
        if seed is not None:
            fallback += f"_seed{seed}"
        if os.path.isdir(fallback):
            print(f"WARNING: run_output_dir: '{output_dir}' doesn't exist, but '{fallback}' does - "
                  f"falling back to it. This means the run at '{fallback}' was made under "
                  f"run_ga.py's post-2026-08-24 defaults (rank_ic fitness/local backend implicit, "
                  f"no suffix) even though fitness_metric/local was passed here as if it were a "
                  f"pre-2026-08-24 run. Drop --fitness-metric rank_ic (and any local=True caller) "
                  f"for this seed going forward.")
            return fallback
        print(f"WARNING: run_output_dir: neither '{output_dir}' nor '{fallback}' exists - "
              f"downstream loading will likely find nothing.")
    return output_dir


# ---------------------------------------------------------------------------------------------
# Item 6: expression-size comparison + max_features cross-check - cell 5
# ---------------------------------------------------------------------------------------------

def compare_expression_sizes(temporal_results: dict, no_temporal_results: dict, matched_folds: list,
                              temporal_dir: str, no_temporal_dir: str) -> pd.DataFrame:
    temporal_config = load_run_config(temporal_dir)
    no_temporal_config = load_run_config(no_temporal_dir)
    if not temporal_config or not no_temporal_config:
        print("Skipping max_features cross-check: run_config.json missing for one or both runs "
              "(only written by run_ga.py from this point forward - rerun to populate it).")
    else:
        t_max_features = temporal_config["max_features"]
        n_max_features = no_temporal_config["max_features"]
        print(f"max_features (GAConfig, from each run's own run_config.json): "
              f"temporal={t_max_features}, no_temporal={n_max_features}")
        if t_max_features == n_max_features:
            print("CONFIRMED: max_features is identical between the temporal and no_temporal runs "
                  "(checked directly from each run's persisted config, not assumed).")
        else:
            print("WARNING: max_features DIFFERS between the temporal and no_temporal runs - "
                  "expression-size comparisons below are not apples-to-apples.")
        if not temporal_config.get("enable_temporal_operators"):
            print("WARNING: the 'temporal' run's own run_config.json has enable_temporal_operators="
                  "False - it wasn't actually run with temporal operators enabled.")
        if no_temporal_config.get("enable_temporal_operators"):
            print("WARNING: the 'no_temporal' run's own run_config.json has enable_temporal_operators="
                  "True - it wasn't actually run with temporal operators disabled.")

    temporal_leaf_counts = pd.Series(
        {f: temporal_results[f]["winner_leaf_count"] for f in matched_folds}, name="temporal_winner_leaf_count")
    no_temporal_leaf_counts = pd.Series(
        {f: no_temporal_results[f]["winner_leaf_count"] for f in matched_folds}, name="no_temporal_winner_leaf_count")

    leaf_count_summary = pd.DataFrame({
        "temporal_winner_leaf_count": temporal_leaf_counts,
        "no_temporal_winner_leaf_count": no_temporal_leaf_counts,
    })
    print("\nWinner leaf count per fold:")
    print(leaf_count_summary.to_string())
    print(f"\ntemporal winners: mean leaf count = {temporal_leaf_counts.mean():.2f}, std = {temporal_leaf_counts.std():.2f}")
    print(f"no_temporal winners: mean leaf count = {no_temporal_leaf_counts.mean():.2f}, std = {no_temporal_leaf_counts.std():.2f}")
    return leaf_count_summary


# ---------------------------------------------------------------------------------------------
# Primary hypothesis test - temporal vs no_temporal, per evaluation_framework.md
# ---------------------------------------------------------------------------------------------

def build_primary_comparison(temporal_results: dict, no_temporal_results: dict, matched_folds: list,
                              comparison_output_dir: str) -> pd.DataFrame:
    """
    The single pre-specified primary hypothesis: H0: delta<=0 vs H1: delta>0, where delta is the
    mean paired monthly Rank IC difference (temporal winner - no_temporal winner), pooled over
    every matched final-test fold-month. Tested standalone with the one-sided, null-centered
    block bootstrap (block_bootstrap_ic_delta above) - per evaluation_framework.md, "No
    Holm-Bonferroni correction is applied to this test because it is the single pre-specified
    primary hypothesis."

    This used to also bundle in no_temporal-vs-baseline-A/B/C and temporal-vs-baseline-C as part
    of one six-comparison Holm-Bonferroni family. Those "GA winner vs its own baselines" checks
    are now a SEPARATE, per-arm secondary family (evaluation_framework.md's "secondary baseline
    comparisons") - already computed and Holm-corrected in each run's own pairwise_comparisons.csv
    (analysis/summary.py, written at run_ga.py time), so they're not recomputed here.

    NOT YET IMPLEMENTED: evaluation_framework.md's estimand averages the paired monthly delta
    across 10 pre-specified matched seeds per arm before this bootstrap runs. This currently uses
    whichever single seed's results are on disk for each arm - flagged in the printed output and
    in the returned/written table (`seed_averaging_applied=False`), not silently assumed away.
    """
    fold_month_deltas = matched_month_deltas_cross_variant(
        temporal_results, no_temporal_results, "winner_monthly_ic", matched_folds)

    if not fold_month_deltas:
        print("Skipping primary comparison: no matched (fold, month) IC pairs available yet.")
        return pd.DataFrame()

    boot_result = block_bootstrap_ic_delta(fold_month_deltas)
    row = {
        "comparison": "temporal_vs_no_temporal",
        "n_folds": len(fold_month_deltas),
        "observed_delta": boot_result["observed_delta"],
        "lower_bound_95": boot_result["lower_bound_95"],
        "p_value": boot_result["p_value"],
        "insufficient_folds": boot_result["insufficient_folds"],
        "significant_at_0.05": (not boot_result["insufficient_folds"]) and boot_result["p_value"] < 0.05,
        "seed_averaging_applied": False,
    }
    primary_df = pd.DataFrame([row])

    print("\nPrimary comparison (temporal vs no_temporal, one-sided null-centered block "
          "bootstrap, NOT Holm-Bonferroni corrected - single pre-specified primary hypothesis). "
          "NOTE: seed_averaging_applied=False - the 10-matched-seed averaging in "
          "evaluation_framework.md is not yet implemented, this uses a single seed per arm:")
    print(primary_df.to_string(index=False))

    primary_path = f"{comparison_output_dir}/primary_comparison.csv"
    primary_df.to_csv(primary_path, index=False)
    print(f"\nWrote {primary_path}")
    return primary_df


def build_primary_comparison_seed_averaged(temporal_results_by_seed: dict, no_temporal_results_by_seed: dict,
                                            matched_folds: list, seeds: list,
                                            comparison_output_dir: str) -> pd.DataFrame:
    """
    Seed-averaged version of build_primary_comparison, per evaluation_framework.md's actual
    estimand (previously NOT YET IMPLEMENTED - see this file's module docstring). Averages the
    paired temporal-minus-no_temporal monthly IC delta across `seeds` per fold-month
    (seed_averaged_fold_month_deltas) BEFORE running the same one-sided, null-centered block
    bootstrap used everywhere else in this script.

    evaluation_framework.md specifies 10 pre-specified matched seeds; this runs on however many
    are passed in (currently 3: 10, 11, 12), and n_seeds is recorded in the output so a 3-seed
    average is never mistaken for the full pre-registered 10-seed one.
    """
    fold_month_deltas = seed_averaged_fold_month_deltas(
        temporal_results_by_seed, no_temporal_results_by_seed, "winner_monthly_ic", matched_folds, seeds)

    if not fold_month_deltas:
        print("Skipping seed-averaged primary comparison: no matched (fold, month) IC pairs "
              "available across any seed.")
        return pd.DataFrame()

    boot_result = block_bootstrap_ic_delta(fold_month_deltas)
    row = {
        "comparison": "temporal_vs_no_temporal_seed_averaged",
        "seeds": ",".join(str(s) for s in seeds),
        "n_seeds": len(seeds),
        "n_folds": len(fold_month_deltas),
        "observed_delta": boot_result["observed_delta"],
        "lower_bound_95": boot_result["lower_bound_95"],
        "p_value": boot_result["p_value"],
        "insufficient_folds": boot_result["insufficient_folds"],
        "significant_at_0.05": (not boot_result["insufficient_folds"]) and boot_result["p_value"] < 0.05,
        "seed_averaging_applied": True,
    }
    primary_df = pd.DataFrame([row])

    print(f"\nPrimary comparison (temporal vs no_temporal, SEED-AVERAGED across {len(seeds)} seed(s) "
          f"{seeds} - evaluation_framework.md's pre-registered estimand calls for 10 matched seeds, "
          f"this uses whichever are on disk so far). One-sided null-centered block bootstrap, NOT "
          f"Holm-Bonferroni corrected (single pre-specified primary hypothesis):")
    print(primary_df.to_string(index=False))

    primary_path = f"{comparison_output_dir}/primary_comparison_seed_averaged.csv"
    primary_df.to_csv(primary_path, index=False)
    print(f"\nWrote {primary_path}")
    return primary_df


def build_secondary_seed_block_bootstrap(temporal_results_by_seed: dict, no_temporal_results_by_seed: dict,
                                          matched_folds: list, seeds: list,
                                          comparison_output_dir: str, block_length: int = 3,
                                          n_resamples: int = 3000, seed: int = 42) -> pd.DataFrame:
    """
    Secondary robustness check alongside build_primary_comparison_seed_averaged: same H1 estimand
    (temporal-minus-no_temporal Rank IC, one-sided, null-centered), but the bootstrap also resamples
    the matched GA seeds themselves (with replacement) instead of treating them as fixed - see
    seed_and_block_bootstrap_ic_delta's docstring. Reports a secondary lower_bound_95/p_value that
    reflects both temporal and GA-seed resampling uncertainty, plus the full null-centered
    resampling distribution (null_deltas) for a one-sided test consistent with the primary analysis.
    """
    per_seed_fold_deltas = build_per_seed_fold_deltas(
        temporal_results_by_seed, no_temporal_results_by_seed, "winner_monthly_ic", matched_folds, seeds)

    if not any(per_seed_fold_deltas.values()):
        print("Skipping secondary seed+block bootstrap: no matched (fold, seed, month) IC triples "
              "available.")
        return pd.DataFrame()

    boot_result = seed_and_block_bootstrap_ic_delta(
        per_seed_fold_deltas, seeds, block_length=block_length, n_resamples=n_resamples, seed=seed)
    row = {
        "comparison": "temporal_vs_no_temporal_seed_and_block_bootstrap",
        "seeds": ",".join(str(s) for s in seeds),
        "n_seeds": len(seeds),
        "n_folds": len([f for f, s in per_seed_fold_deltas.items() if s]),
        "block_length": boot_result["block_length"],
        "n_resamples": boot_result["n_resamples"],
        "observed_delta": boot_result["observed_delta"],
        "lower_bound_95": boot_result["lower_bound_95"],
        "p_value": boot_result["p_value"],
        "insufficient_folds": boot_result["insufficient_folds"],
        "significant_at_0.05": (not boot_result["insufficient_folds"]) and boot_result["p_value"] < 0.05,
    }
    secondary_df = pd.DataFrame([row])

    print(f"\nSecondary comparison (temporal vs no_temporal, seed+block bootstrap over "
          f"{len(seeds)} matched seed(s) {seeds} - resamples seeds AND within-fold month blocks, "
          f"reflects GA-seed uncertainty in addition to temporal uncertainty):")
    print(secondary_df.to_string(index=False))

    secondary_path = f"{comparison_output_dir}/secondary_seed_block_bootstrap.csv"
    secondary_df.to_csv(secondary_path, index=False)
    print(f"Wrote {secondary_path}")

    null_path = f"{comparison_output_dir}/secondary_seed_block_bootstrap_null_distribution.csv"
    pd.DataFrame({"boot_delta": boot_result["boot_deltas"], "null_delta": boot_result["null_deltas"]}) \
        .to_csv(null_path, index=False)
    print(f"Wrote {null_path} ({len(boot_result['boot_deltas'])} replicate(s))")

    return secondary_df


# ---------------------------------------------------------------------------------------------
# Attribution table (DiD) - cell 9
# ---------------------------------------------------------------------------------------------

def build_attribution_table(temporal_results: dict, no_temporal_results: dict, matched_folds: list,
                             comparison_output_dir: str) -> pd.DataFrame:
    """
    The difference-in-differences (DiD) attribution check - per evaluation_framework.md, "an
    optional difference-in-differences diagnostic" that "assesses whether evolutionary search
    gains more over random search when temporal operators are available. It is a mechanism
    diagnostic and is not required for H1 to be supported." Unlike the old design, this is no
    longer part of any Holm-Bonferroni family and does not gate the H1 verdict below - its
    bootstrap numbers are reported purely descriptively.

    lift_no_temporal = no_temporal's real GA result (mean IC) - no_temporal's baseline C result
    (mean IC), lift_temporal the same way within temporal - both fold-weighted (per-fold
    scalars). The DiD statistic (lift_temporal - lift_no_temporal) is bootstrapped with the same
    one-sided, null-centered machinery as the primary comparison, computed directly here (there's
    no other comparison table to cross-reference now that this isn't part of a shared family).
    """
    attribution_rows = []
    for fold_name in matched_folds:
        tr, nr = temporal_results[fold_name], no_temporal_results[fold_name]
        if tr.get("baseline_c_mean_ic") is None or nr.get("baseline_c_mean_ic") is None:
            continue
        lift_no_temporal = nr["winner_mean_ic"] - nr["baseline_c_mean_ic"]
        lift_temporal = tr["winner_mean_ic"] - tr["baseline_c_mean_ic"]
        attribution_rows.append({
            "fold_name": fold_name,
            "eval_year": tr["eval_year"],
            "lift_no_temporal": lift_no_temporal,
            "lift_temporal": lift_temporal,
            "lift_delta": lift_temporal - lift_no_temporal,
        })

    if not attribution_rows:
        print("No matched folds with baseline C results yet - the attribution table needs "
              "baseline C (RUN_BASELINE_C) to have run for both variants.")
        return pd.DataFrame(attribution_rows)

    attribution_df = pd.DataFrame(attribution_rows).sort_values("eval_year").reset_index(drop=True)

    did_fold_month_deltas = matched_month_lift_deltas(temporal_results, no_temporal_results, matched_folds)
    aggregate_row = {
        "fold_name": "AGGREGATE (fold-weighted mean)",
        "eval_year": None,
        "lift_no_temporal": attribution_df["lift_no_temporal"].mean(),
        "lift_temporal": attribution_df["lift_temporal"].mean(),
        "lift_delta": attribution_df["lift_delta"].mean(),
    }
    if did_fold_month_deltas:
        did_boot = block_bootstrap_ic_delta(did_fold_month_deltas)
        aggregate_row["bootstrap_lower_bound_95"] = did_boot["lower_bound_95"]
        aggregate_row["raw_p_value"] = did_boot["p_value"]
        aggregate_row["descriptive_only_not_a_gate"] = True
    attribution_df = pd.concat([attribution_df, pd.DataFrame([aggregate_row])], ignore_index=True)

    print("\nAttribution table (lift_no_temporal, lift_temporal, lift_temporal - lift_no_temporal "
          "per fold, plus the DiD comparison's bootstrap numbers on the aggregate row - "
          "descriptive/mechanism diagnostic only, does not gate the H1 verdict):")
    print(attribution_df.to_string(index=False))

    attribution_path = f"{comparison_output_dir}/attribution_table.csv"
    attribution_df.to_csv(attribution_path, index=False)
    print(f"\nWrote {attribution_path}")
    return attribution_df


# ---------------------------------------------------------------------------------------------
# Item 7: winner composition / stability, with the real delta vs no_temporal - cell 13
# ---------------------------------------------------------------------------------------------

def build_winner_composition(temporal_results: dict, no_temporal_results: dict, matched_folds: list,
                              comparison_output_dir: str) -> pd.DataFrame:
    """
    winner_leaf_classification/winner_temporal_fraction/winner_uses_any_temporal are already
    computed for the temporal run inside run_ga_for_fold() - this only adds what needed
    no_temporal to exist: mean_ic and true_test_rmse deltas vs no_temporal, per fold, for
    eyeballing correlation with temporal-leaf usage (descriptive - no formal test attached, see
    H1 condition (c) below for the one formal-ish check that uses this).
    NOTE on true_test_rmse delta sign: RMSE is an error metric (lower is better), so a NEGATIVE
    true_test_rmse_delta_vs_no_temporal means the temporal run improved on no_temporal for that
    fold - opposite sign convention from mean_ic_delta_vs_no_temporal, where higher is better and
    so a POSITIVE delta means improvement.
    """
    composition_rows = []
    for fold_name in matched_folds:
        tr, nr = temporal_results[fold_name], no_temporal_results[fold_name]
        composition_rows.append({
            "fold_name": fold_name,
            "eval_year": tr["eval_year"],
            "winner_leaf_count": tr["winner_leaf_count"],
            "winner_temporal_fraction": tr["winner_temporal_fraction"],
            "winner_uses_any_temporal": tr["winner_uses_any_temporal"],
            "winner_leaf_classification": tr["winner_leaf_classification"],
            "temporal_mean_ic": tr["winner_mean_ic"],
            "no_temporal_mean_ic": nr["winner_mean_ic"],
            "mean_ic_delta_vs_no_temporal": tr["winner_mean_ic"] - nr["winner_mean_ic"],
            "temporal_true_test_rmse": tr["true_test_rmse"],
            "no_temporal_true_test_rmse": nr["true_test_rmse"],
            "true_test_rmse_delta_vs_no_temporal": tr["true_test_rmse"] - nr["true_test_rmse"],
        })

    composition_df = pd.DataFrame(composition_rows)
    if composition_df.empty:
        print("\nNo matched final-test folds yet to build the winner-composition table from.")
        return composition_df

    composition_df = composition_df.sort_values("eval_year").reset_index(drop=True)

    aggregate_row = {
        "fold_name": "AGGREGATE (mean)",
        "eval_year": None,
        "winner_leaf_count": composition_df["winner_leaf_count"].mean(),
        "winner_temporal_fraction": composition_df["winner_temporal_fraction"].mean(),
        "winner_uses_any_temporal": composition_df["winner_uses_any_temporal"].mean(),
        "winner_leaf_classification": None,
        "temporal_mean_ic": composition_df["temporal_mean_ic"].mean(),
        "no_temporal_mean_ic": composition_df["no_temporal_mean_ic"].mean(),
        "mean_ic_delta_vs_no_temporal": composition_df["mean_ic_delta_vs_no_temporal"].mean(),
        "temporal_true_test_rmse": composition_df["temporal_true_test_rmse"].mean(),
        "no_temporal_true_test_rmse": composition_df["no_temporal_true_test_rmse"].mean(),
        "true_test_rmse_delta_vs_no_temporal": composition_df["true_test_rmse_delta_vs_no_temporal"].mean(),
    }

    # X-of-Y fold-level sign check, IC-framed (the RMSE-based win count in the combined-summary
    # table below is a separate, secondary metric). Descriptive only - not a formal significance
    # test (see H1 condition (c) below for how this feeds the verdict).
    per_fold = composition_df  # pre-aggregate-row append
    n_folds_ic = len(per_fold)
    n_temporal_wins_ic = int((per_fold["mean_ic_delta_vs_no_temporal"] > 0).sum())

    composition_df = pd.concat([composition_df, pd.DataFrame([aggregate_row])], ignore_index=True)

    print("\nWinner-composition table (temporal, with deltas vs no_temporal):")
    print(composition_df.to_string(index=False))

    composition_path = f"{comparison_output_dir}/winner_composition_comparison.csv"
    composition_df.to_csv(composition_path, index=False)
    print(f"\nWrote {composition_path}")

    print(f"\ntemporal outperformed no_temporal on mean IC in "
          f"{n_temporal_wins_ic}/{n_folds_ic} matched final-test folds (descriptive, not a "
          f"formal significance test).")
    return composition_df


# ---------------------------------------------------------------------------------------------
# Combined per-fold results table, temporal and no_temporal side by side - cell 15
# ---------------------------------------------------------------------------------------------

def build_combined_summary(temporal_results: dict, no_temporal_results: dict, matched_folds: list,
                            comparison_output_dir: str) -> pd.DataFrame:
    """
    Each run already writes its OWN final_test_summary.csv (winner vs its own baselines only) -
    this merges both trees on fold_name into one table for the head-to-head view, and adds
    baseline A/B/C rmse/mean_ic/ic_ir + baseline B/C's winning individual/leaf count for BOTH
    variants. Baseline A/B/C are run independently inside each variant's own pipeline (different
    leaf vocabularies - plain vs plain+temporal), so both variants' baselines are reported side
    by side rather than picking one arbitrarily.
    """
    summary_rows = []
    for fold_name in matched_folds:
        tr, nr = temporal_results[fold_name], no_temporal_results[fold_name]
        summary_rows.append({
            "fold_name": fold_name,
            "eval_year": tr["eval_year"],
            "temporal_winner_expression": tr["winning_expression"],
            "temporal_winner_leaf_count": tr["winner_leaf_count"],
            "temporal_true_test_rmse": tr["true_test_rmse"],
            "temporal_mean_ic": tr["winner_mean_ic"],
            "temporal_ic_ir": tr["winner_ic_ir"],
            "no_temporal_winner_expression": nr["winning_expression"],
            "no_temporal_winner_leaf_count": nr["winner_leaf_count"],
            "no_temporal_true_test_rmse": nr["true_test_rmse"],
            "no_temporal_mean_ic": nr["winner_mean_ic"],
            "no_temporal_ic_ir": nr["winner_ic_ir"],
            "temporal_beats_no_temporal_on_rmse": tr["true_test_rmse"] < nr["true_test_rmse"],
            "temporal_beats_no_temporal_on_mean_ic": tr["winner_mean_ic"] > nr["winner_mean_ic"],
            "temporal_elapsed_seconds": tr["elapsed_seconds"],
            "no_temporal_elapsed_seconds": nr["elapsed_seconds"],
            # ---- baseline A/B/C, per variant ----
            "temporal_baseline_a_rmse": tr["baseline_rmse"],
            "temporal_baseline_a_mean_ic": tr["baseline_a_mean_ic"],
            "temporal_baseline_a_ic_ir": tr["baseline_a_ic_ir"],
            "temporal_baseline_b_individual": tr["baseline_b_individual"],
            "temporal_baseline_b_rmse": tr["baseline_b_rmse"],
            "temporal_baseline_b_leaf_count": tr["baseline_b_leaf_count"],
            "temporal_baseline_b_mean_ic": tr["baseline_b_mean_ic"],
            "temporal_baseline_b_ic_ir": tr["baseline_b_ic_ir"],
            "temporal_baseline_c_individual": tr["baseline_c_individual"],
            "temporal_baseline_c_rmse": tr["baseline_c_rmse"],
            "temporal_baseline_c_leaf_count": tr["baseline_c_leaf_count"],
            "temporal_baseline_c_mean_ic": tr["baseline_c_mean_ic"],
            "temporal_baseline_c_ic_ir": tr["baseline_c_ic_ir"],
            "no_temporal_baseline_a_rmse": nr["baseline_rmse"],
            "no_temporal_baseline_a_mean_ic": nr["baseline_a_mean_ic"],
            "no_temporal_baseline_a_ic_ir": nr["baseline_a_ic_ir"],
            "no_temporal_baseline_b_individual": nr["baseline_b_individual"],
            "no_temporal_baseline_b_rmse": nr["baseline_b_rmse"],
            "no_temporal_baseline_b_leaf_count": nr["baseline_b_leaf_count"],
            "no_temporal_baseline_b_mean_ic": nr["baseline_b_mean_ic"],
            "no_temporal_baseline_b_ic_ir": nr["baseline_b_ic_ir"],
            "no_temporal_baseline_c_individual": nr["baseline_c_individual"],
            "no_temporal_baseline_c_rmse": nr["baseline_c_rmse"],
            "no_temporal_baseline_c_leaf_count": nr["baseline_c_leaf_count"],
            "no_temporal_baseline_c_mean_ic": nr["baseline_c_mean_ic"],
            "no_temporal_baseline_c_ic_ir": nr["baseline_c_ic_ir"],
        })

    combined_summary_df = pd.DataFrame(summary_rows)
    if combined_summary_df.empty:
        print("\nNo matched final-test folds yet to build the combined summary table from.")
        return combined_summary_df

    combined_summary_df = combined_summary_df.sort_values("eval_year").reset_index(drop=True)
    print("\nCombined temporal-vs-no_temporal final-test fold summary:")
    print(combined_summary_df.to_string(index=False))

    n_temporal_wins = int(combined_summary_df["temporal_beats_no_temporal_on_rmse"].sum())
    n_temporal_wins_ic = int(combined_summary_df["temporal_beats_no_temporal_on_mean_ic"].sum())
    print(f"\ntemporal beat no_temporal on true_test_rmse in "
          f"{n_temporal_wins}/{len(combined_summary_df)} matched final-test folds, "
          f"and on mean IC in {n_temporal_wins_ic}/{len(combined_summary_df)}.")

    combined_summary_path = f"{comparison_output_dir}/final_test_summary_comparison.csv"
    combined_summary_df.to_csv(combined_summary_path, index=False)
    print(f"\nWrote {combined_summary_path}")
    return combined_summary_df


# ---------------------------------------------------------------------------------------------
# Pooled-across-observations RMSE/Rank IC/IC-IR per arm (not fold-averaged)
# ---------------------------------------------------------------------------------------------

def build_pooled_metrics_table(temporal_dir: str, no_temporal_dir: str,
                                temporal_results: dict, no_temporal_results: dict,
                                matched_folds: list, comparison_output_dir: str) -> pd.DataFrame:
    """
    Pooled-across-observations RMSE/Rank IC/IC-IR per arm, matching the SAME "pool every
    observation, then compute one number" construction evaluation_framework.md's primary estimand
    already uses for the temporal-vs-no_temporal IC DELTA (block_bootstrap_ic_delta's
    observed_delta pools every matched fold-month before averaging) - applied here to each arm's
    own RAW metric level, not just the difference between arms. Neither build_final_test_summary
    nor build_winner_composition's "AGGREGATE (mean)" rows do this: they average 5 fold-level
    statistics, weighting every fold equally regardless of how many held-out rows/months it
    actually contributed - a materially different number from pooling every observation first,
    and NOT what this function computes.

    - Rank IC: every matched final-test fold's own winner_monthly_ic (run_ga_for_fold already
      computes one value per held-out month) is flattened across all matched folds into one list
      and averaged directly - not fold-mean-of-fold-means.
    - IC-IR: the SAME pooled monthly-IC list, mean/std(ddof=1) computed directly on it (matches
      evaluation/metrics.py's summarize_ic formula, just applied to the pooled series instead of
      one fold's own series) - not the average of 5 separate fold-level IC-IRs.
    - RMSE: every matched fold's own predictions.csv (raw fsym_id/sector_return_date/label/
      prediction rows - the same file compute_monthly_ic reads) is concatenated across all
      matched folds and RMSE computed once over the pooled residuals - not sqrt(mean(fold_rmse^2))
      or mean(fold_rmse).

    NOT YET IMPLEMENTED (same gap as build_primary_comparison above, flagged the same way):
    evaluation_framework.md's actual spec averages each of these across 10 matched seeds per arm
    before reporting one number. This computes them for whichever ONE seed's data
    temporal_dir/no_temporal_dir point at, since a matched 10-seed set doesn't exist yet -
    `seed_averaging_applied=False` in the output makes this explicit rather than silently
    reporting a single-seed number as if it were the seed-averaged estimand.
    """
    def _pooled_metrics(arm_dir: str, results: dict) -> dict:
        all_ic = []
        pred_frames = []
        for fold_name in matched_folds:
            monthly_ic = results[fold_name].get("winner_monthly_ic") or {}
            all_ic.extend(monthly_ic.values())
            predictions_path = f"{arm_dir}/final_test/{fold_name}/predictions.csv"
            if os.path.isfile(predictions_path):
                pred_frames.append(pd.read_csv(predictions_path))
            else:
                print(f"WARNING: {predictions_path} missing - pooled RMSE excludes {fold_name}.")

        ic_series = pd.Series(all_ic, dtype=float)
        pooled_mean_ic = float(ic_series.mean()) if len(ic_series) else float('nan')
        pooled_ic_ir = (
            float(ic_series.mean() / ic_series.std(ddof=1))
            if len(ic_series) >= 2 and ic_series.std(ddof=1) != 0 else float('nan')
        )

        if pred_frames:
            pooled_preds = pd.concat(pred_frames, ignore_index=True)
            pooled_rmse = float(np.sqrt(((pooled_preds["label"] - pooled_preds["prediction"]) ** 2).mean()))
            n_observations = len(pooled_preds)
        else:
            pooled_rmse = float('nan')
            n_observations = 0

        return {
            "pooled_rmse": pooled_rmse,
            "pooled_mean_ic": pooled_mean_ic,
            "pooled_ic_ir": pooled_ic_ir,
            "n_months": len(ic_series),
            "n_observations": n_observations,
        }

    temporal_metrics = _pooled_metrics(temporal_dir, temporal_results)
    no_temporal_metrics = _pooled_metrics(no_temporal_dir, no_temporal_results)

    row = {"seed_averaging_applied": False}
    row.update({f"temporal_{k}": v for k, v in temporal_metrics.items()})
    row.update({f"no_temporal_{k}": v for k, v in no_temporal_metrics.items()})
    # ON - OFF for each metric, same sign convention build_winner_composition already uses for
    # true_test_rmse_delta_vs_no_temporal: RMSE is an error metric, so a NEGATIVE rmse_delta means
    # temporal IMPROVED on no_temporal - opposite sign convention from mean_ic/ic_ir_delta, where
    # higher is better and so a POSITIVE delta means improvement.
    for metric in ("pooled_rmse", "pooled_mean_ic", "pooled_ic_ir"):
        row[f"{metric}_delta_on_minus_off"] = temporal_metrics[metric] - no_temporal_metrics[metric]
    pooled_df = pd.DataFrame([row])

    print("\nPooled (not fold-averaged) RMSE / Rank IC / IC-IR per arm - see "
          "build_pooled_metrics_table's own docstring for the exact construction:")
    print(pooled_df.to_string(index=False))

    pooled_path = f"{comparison_output_dir}/pooled_metrics.csv"
    pooled_df.to_csv(pooled_path, index=False)
    print(f"\nWrote {pooled_path}")
    return pooled_df


# ---------------------------------------------------------------------------------------------
# Seed-averaged per-arm level metrics - evaluation_framework.md's §Aggregation
# ---------------------------------------------------------------------------------------------

def _seed_averaged_rank_ic_and_ic_ir(fold_month_ic: dict) -> dict:
    """
    Rank IC_a and IC-IR_a per evaluation_framework.md's §Aggregation, given ONE arm's already
    seed-averaged, cross-arm-matched {fold_name: pd.Series} (one of the two dicts
    seed_averaged_monthly_ic returns - see that function's docstring for why matching against the
    OTHER arm's dates first is required for mean(ON) - mean(OFF) to equal
    build_primary_comparison_seed_averaged's observed_delta). Rank IC_a is the plain mean of the
    pooled series; IC-IR_a is mean/std(ddof=1) of the SAME pooled series - not an average of
    per-seed IC-IRs, and not computed from each seed's own raw (unaveraged) monthly IC values.
    """
    all_ic = [v for series in fold_month_ic.values() for v in series.tolist()]
    ic_series = pd.Series(all_ic, dtype=float)
    mean_ic = float(ic_series.mean()) if len(ic_series) else float('nan')
    ic_ir = (float(ic_series.mean() / ic_series.std(ddof=1))
             if len(ic_series) >= 2 and ic_series.std(ddof=1) != 0 else float('nan'))
    return {"rank_ic": mean_ic, "ic_ir": ic_ir, "n_months": len(ic_series)}


def _seed_averaged_rmse(arm_dirs_by_seed: dict, folds, seeds: list) -> dict:
    """
    RMSE_a per evaluation_framework.md's §Aggregation: "held-out prediction errors are pooled
    across the final-test folds within each seed and arm, and RMSE is calculated from the pooled
    errors. The resulting seed-level RMSE values are then averaged across the ... seeds." So this
    computes ONE pooled RMSE per seed first (same per-seed pooling build_pooled_metrics_table's
    own _pooled_metrics already does), THEN averages those seed-level RMSE values - NOT one RMSE
    from pooling every (seed, fold) prediction together in one shot.
    """
    seed_rmses = []
    for seed in seeds:
        arm_dir = arm_dirs_by_seed[seed]
        pred_frames = []
        for fold_name in folds:
            predictions_path = f"{arm_dir}/final_test/{fold_name}/predictions.csv"
            if os.path.isfile(predictions_path):
                pred_frames.append(pd.read_csv(predictions_path))
            else:
                print(f"WARNING: {predictions_path} missing - seed {seed}'s pooled RMSE excludes {fold_name}.")
        if not pred_frames:
            print(f"WARNING: seed {seed} has no predictions.csv for any matched fold under "
                  f"{arm_dir} - excluded from the seed-averaged RMSE.")
            continue
        pooled_preds = pd.concat(pred_frames, ignore_index=True)
        seed_rmses.append(float(np.sqrt(((pooled_preds["label"] - pooled_preds["prediction"]) ** 2).mean())))
    return {
        "rmse": float(np.mean(seed_rmses)) if seed_rmses else float('nan'),
        "n_seeds_used": len(seed_rmses),
    }


def build_seed_averaged_aggregate_metrics(temporal_results_by_seed: dict, no_temporal_results_by_seed: dict,
                                           temporal_dirs_by_seed: dict, no_temporal_dirs_by_seed: dict,
                                           matched_folds: list, seeds: list,
                                           comparison_output_dir: str) -> pd.DataFrame:
    """
    Per-arm seed-averaged Rank IC / IC-IR / RMSE, per evaluation_framework.md's §Aggregation - the
    per-ARM counterpart to build_primary_comparison_seed_averaged's per-arm DIFFERENCE. Neither
    build_pooled_metrics_table (single seed only) nor build_primary_comparison_seed_averaged
    (only the ON-minus-OFF delta, not each arm's own level) computes this.

    rank_ic_delta_on_minus_off here is algebraically equal to build_primary_comparison_seed_averaged's
    observed_delta (averaging is linear and both are built from the same matched (fold, seed,
    month) triples - enforced via seed_averaged_monthly_ic's cross-arm date matching, see its
    docstring for the bug this fixed) - reported as a cross-check, not a second estimate of the
    primary effect.
    """
    temporal_fold_month_ic, no_temporal_fold_month_ic = seed_averaged_monthly_ic(
        temporal_results_by_seed, no_temporal_results_by_seed, "winner_monthly_ic", matched_folds, seeds)
    temporal_ic = _seed_averaged_rank_ic_and_ic_ir(temporal_fold_month_ic)
    no_temporal_ic = _seed_averaged_rank_ic_and_ic_ir(no_temporal_fold_month_ic)
    temporal_rmse = _seed_averaged_rmse(temporal_dirs_by_seed, matched_folds, seeds)
    no_temporal_rmse = _seed_averaged_rmse(no_temporal_dirs_by_seed, matched_folds, seeds)

    row = {
        "seeds": ",".join(str(s) for s in seeds),
        "n_seeds": len(seeds),
        "temporal_rank_ic": temporal_ic["rank_ic"],
        "temporal_ic_ir": temporal_ic["ic_ir"],
        "temporal_n_months": temporal_ic["n_months"],
        "temporal_rmse": temporal_rmse["rmse"],
        "temporal_rmse_n_seeds_used": temporal_rmse["n_seeds_used"],
        "no_temporal_rank_ic": no_temporal_ic["rank_ic"],
        "no_temporal_ic_ir": no_temporal_ic["ic_ir"],
        "no_temporal_n_months": no_temporal_ic["n_months"],
        "no_temporal_rmse": no_temporal_rmse["rmse"],
        "no_temporal_rmse_n_seeds_used": no_temporal_rmse["n_seeds_used"],
    }
    # Sign convention matches build_pooled_metrics_table: RMSE is an error metric (lower better),
    # so a NEGATIVE rmse delta means temporal improved; rank_ic/ic_ir are higher-is-better.
    row["rank_ic_delta_on_minus_off"] = row["temporal_rank_ic"] - row["no_temporal_rank_ic"]
    row["ic_ir_delta_on_minus_off"] = row["temporal_ic_ir"] - row["no_temporal_ic_ir"]
    row["rmse_delta_on_minus_off"] = row["temporal_rmse"] - row["no_temporal_rmse"]
    aggregate_df = pd.DataFrame([row])

    print(f"\nSeed-averaged aggregate metrics per arm (evaluation_framework.md §Aggregation - "
          f"Rank IC/IC-IR from the seed-averaged monthly IC series pooled across folds, RMSE "
          f"averaged across each seed's own fold-pooled RMSE), across {len(seeds)} seed(s) {seeds}:")
    print(aggregate_df.to_string(index=False))

    aggregate_path = f"{comparison_output_dir}/seed_averaged_aggregate_metrics.csv"
    aggregate_df.to_csv(aggregate_path, index=False)
    print(f"\nWrote {aggregate_path}")
    return aggregate_df


def build_seed_averaged_fold_rank_ic(temporal_results_by_seed: dict, no_temporal_results_by_seed: dict,
                                     matched_folds: list, seeds: list,
                                     comparison_output_dir: str) -> pd.DataFrame:
    """
    ESTIMAND I, broken out per fold instead of pooled - the one seed-averaged table
    build_seed_averaged_aggregate_metrics doesn't write (it only pools across all matched folds
    into a single row). This is the "Rank IC by final-test fold" shape (paper's Table 4.4-style
    figure/table: one row per fold, temporal vs no_temporal vs delta) - previously only ever
    hand-assembled ad hoc (docs/paper_tables.md Table 4, docs/make_figures.py's fold_effects
    figure each re-derived it inline). Reuses seed_averaged_monthly_ic (same cross-arm date
    matching as the aggregate table) and _seed_averaged_rank_ic_and_ic_ir per fold, so every
    number here is constructed identically to seed_averaged_aggregate_metrics.csv - the AGGREGATE
    row below reproduces that file's temporal_rank_ic/no_temporal_rank_ic/rank_ic_delta_on_minus_off
    exactly (pooling all fold-months together, not averaging the per-fold rows, which would only
    coincide if every fold had the same month count).
    """
    temporal_fold_month_ic, no_temporal_fold_month_ic = seed_averaged_monthly_ic(
        temporal_results_by_seed, no_temporal_results_by_seed, "winner_monthly_ic", matched_folds, seeds)

    rows = []
    for fold_name in matched_folds:
        if fold_name not in temporal_fold_month_ic:
            continue
        t_stats = _seed_averaged_rank_ic_and_ic_ir({fold_name: temporal_fold_month_ic[fold_name]})
        n_stats = _seed_averaged_rank_ic_and_ic_ir({fold_name: no_temporal_fold_month_ic[fold_name]})
        rows.append({
            "fold_name": fold_name,
            "eval_year": temporal_results_by_seed[seeds[0]][fold_name]["eval_year"],
            "n_months": t_stats["n_months"],
            "temporal_rank_ic": t_stats["rank_ic"],
            "no_temporal_rank_ic": n_stats["rank_ic"],
            "rank_ic_delta_on_minus_off": t_stats["rank_ic"] - n_stats["rank_ic"],
        })

    fold_df = pd.DataFrame(rows)
    if fold_df.empty:
        print("\nNo matched fold-months to build the seed-averaged per-fold Rank IC table from.")
        return fold_df
    fold_df = fold_df.sort_values("eval_year").reset_index(drop=True)

    pooled = _seed_averaged_rank_ic_and_ic_ir(temporal_fold_month_ic)
    pooled_off = _seed_averaged_rank_ic_and_ic_ir(no_temporal_fold_month_ic)
    aggregate_row = {
        "fold_name": "AGGREGATE (pooled)",
        "eval_year": None,
        "n_months": pooled["n_months"],
        "temporal_rank_ic": pooled["rank_ic"],
        "no_temporal_rank_ic": pooled_off["rank_ic"],
        "rank_ic_delta_on_minus_off": pooled["rank_ic"] - pooled_off["rank_ic"],
    }
    fold_df = pd.concat([fold_df, pd.DataFrame([aggregate_row])], ignore_index=True)

    print(f"\nSeed-averaged Rank IC by final-test fold (Estimand I, {len(seeds)} seed(s) {seeds}):")
    print(fold_df.to_string(index=False))

    fold_path = f"{comparison_output_dir}/seed_averaged_fold_rank_ic.csv"
    fold_df.to_csv(fold_path, index=False)
    print(f"\nWrote {fold_path}")
    return fold_df


def build_pooled_operator_frequency(temporal_results_by_seed: dict, matched_folds: list, seeds: list,
                                    comparison_output_dir: str) -> pd.DataFrame:
    """
    Pools winner_leaf_classification across every (seed, matched fold) temporal-ON winner into a
    per-operator usage count - the "how many winners use lag1/delta1/growth1/mean3/std3" table
    (paper's Table 8/operator_frequency figure). Previously only ever produced by
    research/scratch_analyze_seeds.py, a one-off script hardcoded to seeds 10/11/12 and marked
    "safe to delete" - this reads the same winner_leaf_classification field fold_result.json
    already stores (no need for that script's standalone ast-based expression re-parsing).
    winner_leaf_classification lists FAMILIES (e.g. "lag", "mean"), not period/window-specific
    operator names - mapped to the single lag1/delta1/growth1/mean3/std3 vocabulary this project's
    grammar actually uses (lag_periods=[1], window_sizes=[3], see genome/grammar.py), matching
    docs/make_figures.py's own family_to_op mapping.
    """
    family_to_op = {"lag": "lag1", "delta": "delta1", "growth": "growth1",
                     "mean": "mean3", "std": "std3"}
    op_order = ["lag1", "delta1", "growth1", "mean3", "std3"]
    counts = {op: 0 for op in op_order}
    n_winners = 0
    for seed in seeds:
        for fold_name in matched_folds:
            result = temporal_results_by_seed[seed].get(fold_name)
            if result is None:
                continue
            n_winners += 1
            families = set(result.get("winner_leaf_classification") or [])
            for family in families:
                if family in family_to_op:
                    counts[family_to_op[family]] += 1

    freq_df = pd.DataFrame([
        {"operator": op, "n_winners_using": counts[op], "n_winners_total": n_winners,
         "pct_of_winners": (100 * counts[op] / n_winners) if n_winners else float('nan')}
        for op in op_order
    ])

    print(f"\nPooled temporal-operator usage across {n_winners} temporal-ON winners "
          f"({len(matched_folds)} matched folds x {len(seeds)} seed(s)):")
    print(freq_df.to_string(index=False))

    freq_path = f"{comparison_output_dir}/pooled_operator_frequency.csv"
    freq_df.to_csv(freq_path, index=False)
    print(f"\nWrote {freq_path}")
    return freq_df


# ---------------------------------------------------------------------------------------------
# Estimand II tables - seed-level Rank IC (paper's Tables 4.5/4.6, Figure 4.2)
# ---------------------------------------------------------------------------------------------

def build_seed_level_rank_ic(temporal_results_by_seed: dict, no_temporal_results_by_seed: dict,
                              matched_folds: list, seeds: list, comparison_output_dir: str) -> dict:
    """
    ESTIMAND II (see the "Three Rank IC estimands" note above seed_level_monthly_ic). Writes two
    CSVs, both per-arm, NOT cross-arm-matched (each arm computed entirely from its own data):
      - seed_level_rank_ic_by_fold.csv: RankIC_{a,s,f} - paper's Table 4.5 (mean/SD/range across
        the `seeds`, reported separately for every fold).
      - seed_level_rank_ic.csv: RankIC_{a,s} - paper's Table 4.6 (mean/SD/range across the
        `seeds`, folds pooled together per seed first).
    Returns {"by_fold": df, "pooled": df} (long-format, one row per seed/[fold]/arm - the
    mean/SD/range summary the paper's tables show is a groupby of this on the caller's side, kept
    that way so the raw per-seed values stay available for a reader who wants them).
    """
    by_fold_rows, pooled_rows = [], []
    for arm_name, results_by_seed in (("temporal_on", temporal_results_by_seed),
                                       ("temporal_off", no_temporal_results_by_seed)):
        by_fold_df = seed_level_rank_ic_by_fold(results_by_seed, matched_folds, seeds)
        by_fold_df["arm"] = arm_name
        by_fold_rows.append(by_fold_df)
        pooled_df = seed_level_rank_ic(results_by_seed, matched_folds, seeds)
        pooled_df["arm"] = arm_name
        pooled_rows.append(pooled_df)

    by_fold_df = pd.concat(by_fold_rows, ignore_index=True)
    pooled_df = pd.concat(pooled_rows, ignore_index=True)

    by_fold_path = f"{comparison_output_dir}/seed_level_rank_ic_by_fold.csv"
    by_fold_df.to_csv(by_fold_path, index=False)
    pooled_path = f"{comparison_output_dir}/seed_level_rank_ic.csv"
    pooled_df.to_csv(pooled_path, index=False)

    print(f"\nSeed-level Rank IC (Estimand II - not cross-arm matched), pooled across "
          f"{len(matched_folds)} matched folds, {len(seeds)} seed(s):")
    summary = pooled_df.groupby("arm")["rank_ic"].agg(["mean", "std", lambda s: s.max() - s.min()])
    summary.columns = ["mean", "std", "range"]
    print(summary.to_string())
    print(f"\nWrote {by_fold_path}\nWrote {pooled_path}")
    return {"by_fold": by_fold_df, "pooled": pooled_df}


# ---------------------------------------------------------------------------------------------
# Estimand III tables - predictor-matched Rank IC (paper's Tables 4.7/4.8, Figure 4.3)
# ---------------------------------------------------------------------------------------------

def build_seed_averaged_baseline_comparisons(temporal_results_by_seed: dict, no_temporal_results_by_seed: dict,
                                              matched_folds: list, seeds: list,
                                              comparison_output_dir: str) -> pd.DataFrame:
    """
    ESTIMAND III, 2-way (paper's Table 4.7): seed-averaged GA-vs-baseline comparison, per arm,
    one baseline at a time - the seed-averaged sibling of analysis/summary.py's
    build_pairwise_comparisons (which does this per single run). Holm-Bonferroni corrected across
    each arm's own 2-3 comparisons (matching build_pairwise_comparisons' own per-arm-family
    convention), NOT pooled across both arms.
    """
    comparisons = [("baseline_a_monthly_ic", "Fixed"), ("baseline_b_monthly_ic", "Raw"),
                   ("baseline_c_monthly_ic", "Random")]
    rows = []
    for arm_name, results_by_seed in (("GA-ON", temporal_results_by_seed), ("GA-OFF", no_temporal_results_by_seed)):
        arm_rows, raw_pvalues = [], []
        for base_key, base_name in comparisons:
            fold_month_ga, fold_month_base = seed_averaged_within_arm_matched(
                results_by_seed, "winner_monthly_ic", base_key, matched_folds, seeds)
            if not fold_month_ga:
                continue
            fold_month_deltas = {f: (fold_month_ga[f] - fold_month_base[f]).tolist() for f in fold_month_ga}
            boot = block_bootstrap_ic_delta(fold_month_deltas)
            arm_rows.append({"comparison": f"{arm_name} - {base_name}", "observed_delta": boot["observed_delta"],
                              "lower_bound_95": boot["lower_bound_95"], "raw_p_value": boot["p_value"],
                              "insufficient_folds": boot["insufficient_folds"]})
            if not boot["insufficient_folds"]:
                raw_pvalues.append(boot["p_value"])
        idxs = [i for i, r in enumerate(arm_rows) if not r["insufficient_folds"]]
        if idxs:
            adjusted = holm_bonferroni([arm_rows[i]["raw_p_value"] for i in idxs])
            for i, adj in zip(idxs, adjusted):
                arm_rows[i]["holm_adjusted_p_value"] = adj
                arm_rows[i]["significant_at_0.05"] = adj < 0.05
        rows.extend(arm_rows)

    df = pd.DataFrame(rows)
    path = f"{comparison_output_dir}/seed_averaged_baseline_comparisons.csv"
    df.to_csv(path, index=False)
    print(f"\nSeed-averaged GA-vs-baseline comparisons (Estimand III, 2-way, Holm-corrected per arm):")
    print(df.to_string(index=False))
    print(f"\nWrote {path}")
    return df


def build_seed_averaged_did(temporal_results_by_seed: dict, no_temporal_results_by_seed: dict,
                             matched_folds: list, seeds: list, comparison_output_dir: str) -> dict:
    """
    ESTIMAND III, 2-way (paper's Table 4.8): the difference-in-differences attribution diagnostic,
    seed-averaged. Lift_a = GA_a - Random_a, computed within each arm via the SAME 2-way engine
    Table 4.7 uses (seed_averaged_within_arm_matched, GA matched against baseline C only) - this
    is what makes Lift_a reconcile exactly with Table 4.7's own "arm - Random" row. Deliberately
    does NOT use the N-way engine below, even though it would be tempting to share machinery with
    Figure 4.3 - DiD never involves Fixed/Raw, so matching against them would only drop months for
    no reason (a real bug this project's writeup caught and reverted - see the N-way engine's own
    docstring). DiD = Lift_ON - Lift_OFF, descriptive only, not Holm-corrected (matches
    evaluation_framework.md: "not required for H1 to be supported").
    """
    def arm_ga_random(results_by_seed):
        fold_month_ga, fold_month_rand = seed_averaged_within_arm_matched(
            results_by_seed, "winner_monthly_ic", "baseline_c_monthly_ic", matched_folds, seeds)
        ga = float(np.mean([v for s in fold_month_ga.values() for v in s.tolist()]))
        rand = float(np.mean([v for s in fold_month_rand.values() for v in s.tolist()]))
        return ga, rand

    ga_on, rand_on = arm_ga_random(temporal_results_by_seed)
    ga_off, rand_off = arm_ga_random(no_temporal_results_by_seed)
    lift_on, lift_off = ga_on - rand_on, ga_off - rand_off
    row = {"GA-ON": ga_on, "Random-ON": rand_on, "Lift_ON": lift_on,
           "GA-OFF": ga_off, "Random-OFF": rand_off, "Lift_OFF": lift_off,
           "DiD": lift_on - lift_off}
    df = pd.DataFrame([row])
    path = f"{comparison_output_dir}/seed_averaged_did.csv"
    df.to_csv(path, index=False)
    print(f"\nSeed-averaged DiD attribution (Estimand III, 2-way):")
    print(df.to_string(index=False))
    print(f"\nWrote {path}")
    return row


def build_predictor_comparison_matched(temporal_results_by_seed: dict, no_temporal_results_by_seed: dict,
                                        matched_folds: list, seeds: list, comparison_output_dir: str) -> pd.DataFrame:
    """
    ESTIMAND III, N-way (paper's Figure 4.3): all four predictors (GA, Fixed, Raw, Random)
    matched to one common date set, per arm, via seed_averaged_within_arm_matched_multi. Every
    bar in one arm's group of 4 is therefore computed on identical months - the strictest of the
    three estimands (this shrinks the usable month count noticeably vs. the 2-way tables: roughly
    64-75 months here vs. ~113 for Estimand I, since Fixed/Raw/Random each drop different months).
    NOT used for Table 4.7/4.8 - see build_seed_averaged_baseline_comparisons/build_seed_averaged_did
    for why those need the 2-way engine instead.
    """
    keys = {"GA": "winner_monthly_ic", "Fixed": "baseline_a_monthly_ic",
            "Raw": "baseline_b_monthly_ic", "Random": "baseline_c_monthly_ic"}
    rows = []
    for arm_name, results_by_seed in (("temporal_on", temporal_results_by_seed),
                                       ("temporal_off", no_temporal_results_by_seed)):
        pooled = seed_averaged_within_arm_matched_multi(results_by_seed, keys, matched_folds, seeds)
        n_months = len(next(iter(pooled[next(iter(keys))].values()), [])) if pooled[next(iter(keys))] else 0
        n_months = sum(len(s) for s in pooled["GA"].values())
        row = {"arm": arm_name, "n_months": n_months}
        for name in keys:
            all_vals = [v for s in pooled[name].values() for v in s.tolist()]
            row[name] = float(np.mean(all_vals)) if all_vals else float('nan')
        rows.append(row)

    df = pd.DataFrame(rows)
    path = f"{comparison_output_dir}/predictor_comparison_matched.csv"
    df.to_csv(path, index=False)
    print(f"\nPredictor comparison, all 4 matched (Estimand III, N-way) - Figure 4.3's data:")
    print(df.to_string(index=False))
    print(f"\nWrote {path}")
    return df


# ---------------------------------------------------------------------------------------------
# H1 verdict - per evaluation_framework.md's decision hierarchy
# ---------------------------------------------------------------------------------------------

def evaluate_h1(primary_df: pd.DataFrame, composition_df: pd.DataFrame):
    """
    Per evaluation_framework.md: "H1 is supported if the estimated average held-out effect is
    positive and the one-sided 95% lower confidence bound exceeds zero, equivalently p<0.05."
    Just two conditions now - the old four-condition design (a Delta minimum-detectable-effect
    magnitude gate, a fold-consistency majority gate, and a DiD-attribution gate on top of
    statistical reliability) is retired. Fold-by-fold consistency and the DiD attribution result
    are still computed and reported (see build_winner_composition/build_attribution_table above)
    but are explicitly "reported separately as a robustness diagnostic and ... not an additional
    hypothesis gate" per the spec - they don't appear in the conditions below.

    Degrades to "PENDING" if the primary comparison isn't available yet (e.g. too few matched
    folds, or `insufficient_folds` from the bootstrap's min_folds guard).
    """
    conditions = {}
    notes = {}

    if primary_df.empty or bool(primary_df.iloc[0].get("insufficient_folds", True)):
        conditions["magnitude"] = None
        conditions["statistical_reliability"] = None
        notes["magnitude"] = notes["statistical_reliability"] = (
            "cannot be evaluated - primary comparison not available or insufficient final-test folds")
    else:
        row = primary_df.iloc[0]
        conditions["magnitude"] = bool(row["observed_delta"] > 0)
        notes["magnitude"] = f"observed delta-hat = {row['observed_delta']:.4f}"
        conditions["statistical_reliability"] = bool(row["lower_bound_95"] > 0 and row["p_value"] < 0.05)
        notes["statistical_reliability"] = (f"lower_bound_95={row['lower_bound_95']:.4f}, "
                                             f"p_value={row['p_value']:.4f}")

    magnitude, reliability = conditions["magnitude"], conditions["statistical_reliability"]
    if magnitude is None or reliability is None:
        verdict = "H1 VERDICT PENDING - primary comparison could not be evaluated yet"
    elif magnitude and reliability:
        verdict = "H1 SUPPORTED"
    else:
        failed = [k for k, v in conditions.items() if v is False]
        verdict = f"H1 NOT SUPPORTED - failed condition(s): {failed}"

    # Descriptive/robustness-only, per the spec - reported alongside the verdict, does not gate it.
    has_ic_deltas = "mean_ic_delta_vs_no_temporal" in composition_df.columns
    if has_ic_deltas:
        per_fold_rows = composition_df[composition_df["fold_name"] != "AGGREGATE (mean)"]
        n_folds = len(per_fold_rows)
        n_temporal_wins = int((per_fold_rows["mean_ic_delta_vs_no_temporal"] > 0).sum())
        notes["fold_consistency_descriptive"] = (
            f"temporal wins {n_temporal_wins}/{n_folds} matched folds by mean IC "
            f"(descriptive robustness diagnostic, not a hypothesis gate)")

    return conditions, notes, verdict


def build_h1_verdict(primary_df: pd.DataFrame, composition_df: pd.DataFrame,
                      matched_folds: list, comparison_output_dir: str):
    h1_conditions, h1_notes, h1_verdict = evaluate_h1(primary_df, composition_df)

    print("\nH1 verdict (per evaluation_framework.md's decision hierarchy):")
    for key in ["magnitude", "statistical_reliability"]:
        print(f"  ({key}): {h1_conditions[key]} - {h1_notes[key]}")
    if "fold_consistency_descriptive" in h1_notes:
        print(f"  (descriptive) {h1_notes['fold_consistency_descriptive']}")
    print(f"\n{h1_verdict}")

    h1_record = {
        "conditions": h1_conditions,
        "notes": h1_notes,
        "verdict": h1_verdict,
        "matched_folds": matched_folds,
        "seed_averaging_applied": (bool(primary_df.iloc[0]["seed_averaging_applied"])
                                    if not primary_df.empty else False),
    }
    with open(f"{comparison_output_dir}/h1_verdict.json", "w") as f:
        json.dump(h1_record, f, indent=2)
    print(f"\nWrote {comparison_output_dir}/h1_verdict.json")
    return h1_record


# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------

def main(temporal_dir: str, no_temporal_dir: str, comparison_output_dir: str):
    os.makedirs(comparison_output_dir, exist_ok=True)

    temporal_results = load_final_test_fold_results(temporal_dir)
    no_temporal_results = load_final_test_fold_results(no_temporal_dir)

    temporal_folds = set(temporal_results.keys())
    no_temporal_folds = set(no_temporal_results.keys())
    matched_folds = sorted(temporal_folds & no_temporal_folds)
    only_temporal = sorted(temporal_folds - no_temporal_folds)
    only_no_temporal = sorted(no_temporal_folds - temporal_folds)

    print(f"temporal ({temporal_dir}): {len(temporal_folds)} final-test fold(s) with a fold_result.json")
    print(f"no_temporal ({no_temporal_dir}): {len(no_temporal_folds)} final-test fold(s) with a fold_result.json")
    print(f"Matched folds (present in both, used below): {matched_folds}")
    if only_temporal:
        print(f"WARNING: folds only in temporal (not yet run, or not run, in no_temporal) - excluded below: {only_temporal}")
    if only_no_temporal:
        print(f"WARNING: folds only in no_temporal (not yet run, or not run, in temporal) - excluded below: {only_no_temporal}")
    if not matched_folds:
        print("No matched final-test folds yet - re-run this script once both runs finish.")

    # Sanity check: fold_name -> eval_year should agree between the two trees (same walk-forward
    # boundaries) - if this doesn't hold the fold-by-fold pairing below isn't a fair comparison.
    for fold_name in matched_folds:
        yt, yn = temporal_results[fold_name]["eval_year"], no_temporal_results[fold_name]["eval_year"]
        if yt != yn:
            print(f"WARNING: {fold_name} covers eval_year={yt} in temporal but eval_year={yn} in "
                  f"no_temporal - the two runs' fold boundaries have diverged, the pairing below "
                  f"is not comparing the same calendar period.")

    compare_expression_sizes(temporal_results, no_temporal_results, matched_folds, temporal_dir, no_temporal_dir)
    primary_df = build_primary_comparison(temporal_results, no_temporal_results, matched_folds, comparison_output_dir)
    build_attribution_table(temporal_results, no_temporal_results, matched_folds, comparison_output_dir)
    composition_df = build_winner_composition(temporal_results, no_temporal_results, matched_folds, comparison_output_dir)
    build_combined_summary(temporal_results, no_temporal_results, matched_folds, comparison_output_dir)
    build_pooled_metrics_table(temporal_dir, no_temporal_dir, temporal_results, no_temporal_results,
                                matched_folds, comparison_output_dir)
    build_h1_verdict(primary_df, composition_df, matched_folds, comparison_output_dir)


def main_multi_seed(base: str, seeds: list, comparison_output_dir: str, fitness_metric: str = None):
    """
    Seed-averaged entry point (--seeds N N N). Loads every seed's temporal/no_temporal fold_result
    trees, runs the seed-averaged primary hypothesis test (the one part of evaluation_framework.md
    this script previously couldn't do at all with a single seed on disk), then runs every other
    (still per-seed, not seed-averaged - evaluation_framework.md only defines the primary estimand
    as a seed average) diagnostic table once per seed under its own comparison_outputs/<label>/seedN/
    subdirectory, same as main() would for that seed alone.

    The H1 verdict is fed the seed-averaged primary comparison plus a seed-averaged version of the
    fold-consistency descriptive note (mean_ic_delta_vs_no_temporal averaged across seeds per fold,
    not any single seed's) - the verdict itself is still decided purely by the seed-averaged primary
    comparison (evaluate_h1's two conditions), fold-consistency is descriptive only either way.
    """
    os.makedirs(comparison_output_dir, exist_ok=True)

    temporal_results_by_seed, no_temporal_results_by_seed, per_seed_matched_folds = {}, {}, {}
    temporal_dirs_by_seed, no_temporal_dirs_by_seed = {}, {}
    for seed in seeds:
        temporal_dir = run_output_dir(base, no_temporal=False, fitness_metric=fitness_metric, seed=seed)
        no_temporal_dir = run_output_dir(base, no_temporal=True, fitness_metric=fitness_metric, seed=seed)
        temporal_dirs_by_seed[seed] = temporal_dir
        no_temporal_dirs_by_seed[seed] = no_temporal_dir
        t_results = load_final_test_fold_results(temporal_dir)
        n_results = load_final_test_fold_results(no_temporal_dir)
        temporal_results_by_seed[seed] = t_results
        no_temporal_results_by_seed[seed] = n_results
        seed_matched = sorted(set(t_results) & set(n_results))
        per_seed_matched_folds[seed] = seed_matched
        print(f"seed {seed}: temporal ({temporal_dir}) {len(t_results)} final-test fold(s), "
              f"no_temporal ({no_temporal_dir}) {len(n_results)} final-test fold(s), "
              f"matched within this seed: {seed_matched}")

    matched_folds = (sorted(set.intersection(*(set(f) for f in per_seed_matched_folds.values())))
                      if per_seed_matched_folds else [])
    print(f"\nFolds matched across ALL {len(seeds)} seed(s) (used for the seed-averaged primary "
          f"comparison): {matched_folds}")
    for seed, seed_matched in per_seed_matched_folds.items():
        dropped = sorted(set(seed_matched) - set(matched_folds))
        if dropped:
            print(f"WARNING: seed {seed} has fold(s) {dropped} not matched in every other seed - "
                  f"excluded from the seed-averaged primary comparison (still used in seed {seed}'s "
                  f"own per-seed diagnostics below).")

    # Sanity check, mirrors main()'s single-pair check: fold_name -> eval_year should agree across
    # every seed and both arms (same walk-forward boundaries regardless of GA seed).
    for fold_name in matched_folds:
        eval_years = {seed: temporal_results_by_seed[seed][fold_name]["eval_year"] for seed in seeds}
        eval_years.update({f"{seed}_no_temporal": no_temporal_results_by_seed[seed][fold_name]["eval_year"]
                            for seed in seeds})
        if len(set(eval_years.values())) > 1:
            print(f"WARNING: {fold_name} covers different eval_year values across seeds/arms "
                  f"({eval_years}) - the seed-averaged pairing below is not comparing the same "
                  f"calendar period.")

    primary_df = build_primary_comparison_seed_averaged(
        temporal_results_by_seed, no_temporal_results_by_seed, matched_folds, seeds, comparison_output_dir)
    build_secondary_seed_block_bootstrap(
        temporal_results_by_seed, no_temporal_results_by_seed, matched_folds, seeds, comparison_output_dir)
    build_seed_averaged_aggregate_metrics(
        temporal_results_by_seed, no_temporal_results_by_seed,
        temporal_dirs_by_seed, no_temporal_dirs_by_seed, matched_folds, seeds, comparison_output_dir)
    build_seed_averaged_fold_rank_ic(
        temporal_results_by_seed, no_temporal_results_by_seed, matched_folds, seeds, comparison_output_dir)
    build_pooled_operator_frequency(
        temporal_results_by_seed, matched_folds, seeds, comparison_output_dir)
    build_seed_level_rank_ic(
        temporal_results_by_seed, no_temporal_results_by_seed, matched_folds, seeds, comparison_output_dir)
    build_seed_averaged_baseline_comparisons(
        temporal_results_by_seed, no_temporal_results_by_seed, matched_folds, seeds, comparison_output_dir)
    build_seed_averaged_did(
        temporal_results_by_seed, no_temporal_results_by_seed, matched_folds, seeds, comparison_output_dir)
    build_predictor_comparison_matched(
        temporal_results_by_seed, no_temporal_results_by_seed, matched_folds, seeds, comparison_output_dir)

    composition_dfs = []
    for seed in seeds:
        seed_output_dir = f"{comparison_output_dir}/seed{seed}"
        os.makedirs(seed_output_dir, exist_ok=True)
        temporal_dir = run_output_dir(base, no_temporal=False, fitness_metric=fitness_metric, seed=seed)
        no_temporal_dir = run_output_dir(base, no_temporal=True, fitness_metric=fitness_metric, seed=seed)
        t_results, n_results = temporal_results_by_seed[seed], no_temporal_results_by_seed[seed]
        seed_matched = per_seed_matched_folds[seed]

        print(f"\n{'=' * 20} seed {seed} per-seed diagnostics {'=' * 20}")
        compare_expression_sizes(t_results, n_results, seed_matched, temporal_dir, no_temporal_dir)
        build_attribution_table(t_results, n_results, seed_matched, seed_output_dir)
        composition_df = build_winner_composition(t_results, n_results, seed_matched, seed_output_dir)
        if not composition_df.empty:
            composition_dfs.append(composition_df[composition_df["fold_name"] != "AGGREGATE (mean)"])
        build_combined_summary(t_results, n_results, seed_matched, seed_output_dir)
        build_pooled_metrics_table(temporal_dir, no_temporal_dir, t_results, n_results,
                                    seed_matched, seed_output_dir)

    seed_avg_composition = (
        pd.concat(composition_dfs, ignore_index=True)
        .groupby("fold_name", as_index=False)["mean_ic_delta_vs_no_temporal"].mean()
        if composition_dfs else pd.DataFrame()
    )

    build_h1_verdict(primary_df, seed_avg_composition, matched_folds, comparison_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true",
                         help="compare ga_fast/ vs ga_fast_no_temporal/ instead of ga/ vs "
                              "ga_no_temporal/")
    parser.add_argument("--seed", type=int, default=None,
                         help="compare the ga{,_fast}_seedN/ vs ga{,_fast}_no_temporal_seedN/ pair "
                              "run_ga.py's own --seed N produces, instead of the unsuffixed "
                              "default-seed pair. Mutually exclusive with --seeds.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                         help="seed-average the primary hypothesis test across these seeds (e.g. "
                              "--seeds 10 11 12), reading ga{,_fast}{,_rank_ic}_seedN/ vs "
                              "ga{,_fast}_no_temporal{,_rank_ic}_seedN/ for each N - implements "
                              "evaluation_framework.md's seed-averaged estimand (the spec's "
                              "pre-registered design calls for 10 seeds; this runs on however many "
                              "are passed, flagged as such in the output). Every other diagnostic "
                              "table is still computed per seed. Mutually exclusive with --seed.")
    parser.add_argument("--fitness-metric", choices=["rmse", "rank_ic"], default=None,
                         help="match run_ga.py's own --fitness-metric flag - 'rank_ic' reads from "
                              "the '..._rank_ic...' output_dir suffix run_ga.py writes when scored "
                              "on mean monthly Rank IC instead of -RMSE. Default (unset) reads the "
                              "unsuffixed rmse-scored directories, same as omitting the flag on "
                              "run_ga.py itself.")
    args = parser.parse_args()

    if args.seeds is not None and args.seed is not None:
        parser.error("--seed and --seeds are mutually exclusive - use --seeds to seed-average.")

    base = "ga_fast" if args.fast else "ga"
    label = comparison_output_label(fast=args.fast, fitness_metric=args.fitness_metric,
                                     seeds=args.seeds, seed=args.seed)
    comparison_output_dir = f"comparison_outputs/{label}"

    if args.seeds is not None:
        main_multi_seed(base, args.seeds, comparison_output_dir, fitness_metric=args.fitness_metric)
    else:
        temporal_dir = run_output_dir(base, no_temporal=False, fitness_metric=args.fitness_metric, seed=args.seed)
        no_temporal_dir = run_output_dir(base, no_temporal=True, fitness_metric=args.fitness_metric, seed=args.seed)
        main(temporal_dir, no_temporal_dir, comparison_output_dir)
