"""
Pandas sibling of `operators/temporal.py`'s `apply_temporal`, for `GeneticAlgorithm1`'s
`fit_backend="local"` path (see `ga/engine.py`). Kept as a separate module for the same reasons
`arithmetic_local.py`'s docstring gives - Spark-only imports stay Spark-only, and the fork between
the two implementations stays visible at the file-tree level.

Same report-grain dedup-then-broadcast-back strategy as the Spark version (see `temporal.py`'s
module docstring for why: `report_date` repeats across every monthly row in a report's coverage
window, so a naive per-row window/rolling computation over the monthly-expanded grain would
degenerate into a near no-op). Dedup to one row per `(entity_col, date_col)`, apply the
lag/delta/growth/mean/std transform over that reduced, date-sorted frame, then left-merge the
transformed value back onto every monthly row sharing that report date.
"""
import numpy as np
import pandas as pd

from .arithmetic_local import _safe_denominator_local
from .temporal import SCHEMA_COLUMNS, TemporalOpSpec


def apply_temporal_local(
    df: pd.DataFrame,
    op: TemporalOpSpec,
    date_col: str = "report_date",
    entity_col: str = "fsym_id",
) -> pd.DataFrame:
    report_level = (
        df[[entity_col, date_col, "feature"]]
        .drop_duplicates(subset=[entity_col, date_col])
        .sort_values([entity_col, date_col])
    )
    grouped = report_level.groupby(entity_col, group_keys=False)["feature"]

    if op.family == "lag":
        transformed = grouped.shift(op.param)
    elif op.family == "delta":
        transformed = report_level["feature"] - grouped.shift(op.param)
    elif op.family == "growth":
        lag_val = grouped.shift(op.param)
        safe_lag_val = _safe_denominator_local(lag_val)
        transformed = (report_level["feature"].to_numpy() - lag_val.to_numpy()) / safe_lag_val
    elif op.family == "mean":
        # min_periods=1 matches Spark's rowsBetween(-(param-1), 0): the window naturally shrinks
        # near the start of a series rather than requiring `param` full observations first.
        transformed = grouped.transform(lambda s: s.rolling(op.param, min_periods=1).mean())
    elif op.family == "std":
        # pandas .std() defaults to ddof=1 (sample stddev), matching Spark's F.stddev (also
        # sample, not population) - no extra ddof argument needed.
        transformed = grouped.transform(lambda s: s.rolling(op.param, min_periods=1).std())
    else:
        raise ValueError(f"Unsupported temporal op family: {op.family}")

    report_level = report_level.assign(feature_temporal=np.asarray(transformed))[
        [entity_col, date_col, "feature_temporal"]
    ]

    merged = (
        df.drop(columns=["feature"])
        .merge(report_level, on=[entity_col, date_col], how="left")
        .rename(columns={"feature_temporal": "feature"})
    )
    return merged[list(SCHEMA_COLUMNS)]
