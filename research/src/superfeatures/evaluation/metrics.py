"""
Ported from `research/GA_test.ipynb` cell 22 — the pure-Python rank-IC/expression-composition/
baseline-C-sampler helpers (items 1/3/6/7 of ga_methodology_additions_prompt.md). See
docs/RESTRUCTURING_TODO.md / the port plan. Self-contained in the notebook already (no
cross-cell dependencies - imports random/numpy/pandas/scipy itself), so extracted verbatim
with no import fixups needed. Already covered by tests/test_ga_methodology_additions.py
(24 tests against the notebook cell directly).

The cell's other two helpers - Holm-Bonferroni correction and the block bootstrap on IC deltas
(items 4/5, bootstrap/paired-difference significance testing) - now live in
`analysis/significance.py` instead, alongside the driver functions that call them
(`analysis/summary.py`'s `build_pairwise_comparisons`).
"""

# Shared helpers for the methodology additions (see ga_methodology_additions_prompt.md / the
# "Methodology additions" section of CLAUDE.md's GA_test.ipynb entry) - rank IC, expression-size
# reporting, temporal-leaf classification, and baseline C's random-expression sampler. Nothing
# here touches Spark - cell 19 (GeneticAlgorithm1) is untouched, these are all called from the
# driver (cell 23+, now ga/algorithms.py) the same way _leaf_features/evaluate_baseline_for_fold
# already are.
import random
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ---- Item 6: expression-size reporting ----

def _count_leaf_features(individual) -> int:
    """
    Total leaf *occurrences* in an expression - mirrors cell 19's crossover_deep-internal
    count_features exactly (reimplemented here rather than reaching into that closure, same
    convention as _leaf_features in cell 23). Distinct from _leaf_features: this counts
    occurrences (('a','+','a') -> 2), _leaf_features de-dupes (('a','+','a') -> ['a']).
    """
    if isinstance(individual, str):
        return 0 if individual in {'+', '-', '*', '/'} else 1
    elif isinstance(individual, tuple):
        # NOTE: plain sum() is shadowed by pyspark.sql.functions.sum in this kernel (cell 0's
        # unqualified `from pyspark.sql.functions import (..., sum, ...)`) - an explicit
        # accumulator loop is used here instead, per the documented pitfall in CLAUDE.md.
        total = 0
        for sub in individual:
            total += _count_leaf_features(sub)
        return total
    return 0


# ---- Item 7: winner composition (raw vs. temporal leaf) ----

# Matches PreProcessing_test.ipynb's actual call site: utils.add_temporal_features(
# base_fundamentals_df, candidate_feature_columns, lag_periods=[1], window_sizes=[3]) - these
# five suffixes are the only temporal variants it ever generates.
TEMPORAL_SUFFIXES = ['_lag1', '_delta1', '_growth1', '_mean3', '_std3']

def classify_leaf(feature_name: str) -> str:
    """Returns one of TEMPORAL_SUFFIXES (without the leading underscore) or 'raw'."""
    for suffix in TEMPORAL_SUFFIXES:
        if feature_name.endswith(suffix):
            return suffix.lstrip('_')
    return 'raw'


# ---- Item 1: Rank IC / Mean IC / IC-IR ----

def _monthly_ic_from_dataframe(df: pd.DataFrame, date_col: str = 'sector_return_date',
                                prediction_col: str = 'prediction', label_col: str = 'label') -> pd.Series:
    """
    Shared core of compute_monthly_ic - Spearman rank correlation between prediction_col and
    label_col, computed separately within each date_col group - never pooled across groups, per
    ga_methodology_additions_prompt.md item 1. Groups with fewer than 2 rows, or with zero
    variance in either column (spearmanr undefined), are skipped rather than producing a NaN
    entry. Returns a pandas Series of IC values indexed by date_col.

    Factored out of compute_monthly_ic so a caller already holding an in-memory DataFrame (e.g.
    engine.py's rank-IC fitness path, GeneticAlgorithm1.evaluate_fitness_static) doesn't need a
    round-trip through disk just to reuse this exact logic - compute_monthly_ic itself is now a
    thin wrapper (read the CSV, call this).
    """
    monthly_ic = {}
    for month, group in df.groupby(date_col):
        if len(group) < 2 or group[prediction_col].nunique() < 2 or group[label_col].nunique() < 2:
            continue
        ic, _ = spearmanr(group[prediction_col], group[label_col])
        if not np.isnan(ic):
            monthly_ic[month] = ic
    return pd.Series(monthly_ic, name='ic').sort_index()


def compute_monthly_ic(predictions_csv_path: str) -> pd.Series:
    """
    Spearman rank correlation between `prediction` and `label`, computed separately within each
    sector_return_date (month) - never pooled across months, per
    ga_methodology_additions_prompt.md item 1. Months with fewer than 2 rows, or with zero
    variance in either column (spearmanr undefined), are skipped rather than producing a NaN
    entry. Returns a pandas Series of IC values indexed by sector_return_date.
    """
    df = pd.read_csv(predictions_csv_path)
    return _monthly_ic_from_dataframe(df)


def summarize_ic(monthly_ic: pd.Series) -> tuple:
    """(mean_ic, ic_ir = mean_ic / std(ic, ddof=1)). ic_ir is NaN with <2 months or std==0."""
    ic_values = monthly_ic.dropna()
    if len(ic_values) == 0:
        return float('nan'), float('nan')
    if len(ic_values) < 2:
        return float(ic_values.iloc[0]), float('nan')
    mean_ic = float(ic_values.mean())
    std_ic = ic_values.std(ddof=1)
    ic_ir = mean_ic / std_ic if std_ic != 0 else float('nan')
    return mean_ic, ic_ir


# ---- Item 3: baseline C's uniform-random expression sampler ----

def _random_individual(features: list, max_features: int = 5, rng=random, grammar=None, temporal_rate: float = 0.0):
    """
    Uniform-random super-feature expression over `features`: leaf count sampled uniformly from
    1..max_features (mirrors crossover_deep's hardcoded max_features=5 cap - not otherwise
    exposed as a constant there, so duplicated here deliberately), leaves drawn with replacement,
    joined by random operators. Always structurally valid (alternating leaf/operator, odd length)
    by construction, matching is_valid_expression's checks without needing to call it.

    rng defaults to the bare `random` module (preserving prior behavior/existing tests); pass a
    seeded random.Random instance (as baseline C's driver does) for reproducibility.

    grammar/temporal_rate: optional - when grammar is given and rng.random() < temporal_rate,
    the flat expression built above is passed through grammar.wrap() (the same structural-wrap
    function initialize_population()/mutate() use), giving this individual a real temporal atom
    with the same distribution over which run of leaves gets wrapped and which operator wraps it.
    Both default to a no-op (grammar=None, temporal_rate=0.0) so every existing call site/test
    keeps producing leaf-only-arithmetic individuals unchanged. grammar.wrap() itself already
    returns None whenever grammar.temporal_ops is empty, so passing a temporal-off grammar here
    is also automatically a no-op - this parameter never needs its own on/off branch at the call
    site, only a grammar and a rate.
    """
    n_leaves = rng.randint(1, max_features)
    leaves = [rng.choice(features) for _ in range(n_leaves)]
    if n_leaves == 1:
        expr = leaves[0]
    else:
        expr = [leaves[0]]
        for leaf in leaves[1:]:
            expr.append(rng.choice(['+', '-', '*', '/']))
            expr.append(leaf)
        expr = tuple(expr)

    if grammar is not None and grammar.temporal_ops and rng.random() < temporal_rate:
        wrapped = grammar.wrap(expr, rng)
        if wrapped is not None:
            expr = wrapped
    return expr


