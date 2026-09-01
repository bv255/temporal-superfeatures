"""
The live temporal operator vocabulary (lag/delta/growth/mean/std) and its Spark-side evaluator -
merged from the former `temporal_ops.py` (vocabulary: `TemporalOpSpec`/`temporal_ops`) and
`ga/temporal_evaluator.py`'s `apply_temporal` (evaluator), which now live together as the
temporal sibling of `operators/arithmetic.py`'s `combine_dataframes` (TEMPORAL_SUBTREE_OPERATORS_PROMPT.md
section 6/7). Shared by `genome.grammar.ExpressionGrammar` (the GA's genome), `ga.engine`
(the live Spark evaluator call site), and leaf classification for winner-composition reporting.
Mirrors `Utils.add_temporal_features`'s own suffix naming (`preprocessing/utils.py`) so a
live `lag1` atom and the precomputed `_lag1` column mean the same thing.

`apply_temporal` operates on the evaluator's standard per-leaf frame shape: `fsym_id,
sector_return_date, prev_month_return, prev_month_sector_return, feature, target, report_date`
- one row per `(fsym_id, sector_return_date)` (a monthly, as-of-expanded grain), where
`report_date` is the underlying fundamentals report's date (see `data.panel.GAPreprocessing`
/`_feature_frame`), duplicated across every month in that report's coverage window.

It must window over the de-duplicated report grain, not the monthly-expanded grain directly:
because one report's `report_date` repeats across several monthly rows, a naive
`Window.orderBy(report_date)` over the full frame would treat each repeated-date row as a
distinct position, degenerating lag/delta/etc. into a near no-op (see spec section 2's warning
about this exact case). Instead: dedup to one row per `(entity, report_date)` - correct because
`feature` is constant across a report's coverage window by construction (every leaf value in a
monthly row traces back to the same underlying report) - apply the window function over that
reduced frame, then broadcast the result back onto every month sharing that report via a left
join.
"""
from dataclasses import dataclass
from typing import List

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.window import Window

SCHEMA_COLUMNS = (
    "fsym_id", "sector_return_date", "prev_month_return", "prev_month_sector_return",
    "feature", "target", "report_date",
)


@dataclass(frozen=True)
class TemporalOpSpec:
    name: str
    family: str  # "lag" | "delta" | "growth" | "mean" | "std"
    param: int  # k for lag/delta/growth, w for mean/std
    linear: bool  # True for lag/delta/mean - see ExpressionGrammar.is_redundant_wrap
    suffix: str  # e.g. "_lag1" - matches add_temporal_features' column naming


def temporal_ops(lag_periods: List[int] = None, window_sizes: List[int] = None) -> List[TemporalOpSpec]:
    lag_periods = [1] if lag_periods is None else lag_periods
    window_sizes = [3] if window_sizes is None else window_sizes
    ops = []
    for k in lag_periods:
        ops.append(TemporalOpSpec(f"lag{k}", "lag", k, True, f"_lag{k}"))
        ops.append(TemporalOpSpec(f"delta{k}", "delta", k, True, f"_delta{k}"))
        ops.append(TemporalOpSpec(f"growth{k}", "growth", k, False, f"_growth{k}"))
    for w in window_sizes:
        ops.append(TemporalOpSpec(f"mean{w}", "mean", w, True, f"_mean{w}"))
        ops.append(TemporalOpSpec(f"std{w}", "std", w, False, f"_std{w}"))
    return ops


def apply_temporal(
    df: DataFrame,
    op: TemporalOpSpec,
    date_col: str = "report_date",
    entity_col: str = "fsym_id",
) -> DataFrame:
    report_level = df.select(entity_col, date_col, "feature").dropDuplicates([entity_col, date_col])
    win = Window.partitionBy(entity_col).orderBy(date_col)

    if op.family == "lag":
        transformed = F.lag(F.col("feature"), op.param).over(win)
    elif op.family == "delta":
        transformed = F.col("feature") - F.lag(F.col("feature"), op.param).over(win)
    elif op.family == "growth":
        lag_val = F.lag(F.col("feature"), op.param).over(win)
        # Sign-preserving near-zero guard - see operators/arithmetic.py's _safe_denominator for
        # why a bare positive floor would flip the sign of small-but-negative lag values instead
        # of just bounding their magnitude.
        eps = 1e-6
        safe_lag_val = F.when(
            F.abs(lag_val) < eps,
            F.when(lag_val == 0, F.lit(eps)).otherwise(F.signum(lag_val) * F.lit(eps)),
        ).otherwise(lag_val)
        transformed = (F.col("feature") - lag_val) / safe_lag_val
    elif op.family == "mean":
        transformed = F.avg(F.col("feature")).over(win.rowsBetween(-(op.param - 1), 0))
    elif op.family == "std":
        transformed = F.stddev(F.col("feature")).over(win.rowsBetween(-(op.param - 1), 0))
    else:
        raise ValueError(f"Unsupported temporal op family: {op.family}")

    report_level = report_level.select(entity_col, date_col, transformed.alias("feature_temporal"))

    return (
        df.drop("feature")
        .join(report_level, on=[entity_col, date_col], how="left")
        .withColumnRenamed("feature_temporal", "feature")
        .select(*SCHEMA_COLUMNS)
    )
