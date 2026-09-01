"""
Pandas/numpy sibling of `operators/arithmetic.py`, for `GeneticAlgorithm1`'s `fit_backend="local"`
path (see `ga/engine.py`). Deliberately a separate module rather than if-branches inside
`arithmetic.py`: keeps that file's import surface Spark-only (nothing here can destabilize the
Spark-path notebook-parity tests), and keeps the "these two are a deliberately-forked pair, not
meant to numerically agree" relationship visible at the file-tree level.

Operates on the same standard per-leaf frame shape as the Spark version: `fsym_id,
sector_return_date, prev_month_return, prev_month_sector_return, feature, target, report_date` -
one row per `(fsym_id, sector_return_date)` - except as plain `pandas.DataFrame`s instead of Spark
DataFrames.
"""
import numpy as np
import pandas as pd

DIVISION_GUARD_EPS = 1e-6

SCHEMA_COLUMNS = (
    "fsym_id", "sector_return_date", "prev_month_return", "prev_month_sector_return",
    "feature", "target", "report_date",
)


def _safe_denominator_local(denom, eps: float = DIVISION_GUARD_EPS):
    """
    Pandas/numpy equivalent of `arithmetic.py`'s `_safe_denominator` - floors a denominator away
    from zero without flipping its sign (see that function's docstring for why a bare positive
    floor would be a new bug, not a fix, for near-zero-but-nonzero negative denominators).
    """
    denom = denom.to_numpy() if hasattr(denom, "to_numpy") else np.asarray(denom)
    near_zero = np.abs(denom) < eps
    signed_floor = np.where(denom == 0, eps, np.sign(denom) * eps)
    return np.where(near_zero, signed_floor, denom)


def combine_dataframes_local(df1: pd.DataFrame, df2: pd.DataFrame, operator: str) -> pd.DataFrame:
    """
    Pandas equivalent of `arithmetic.py`'s `combine_dataframes`. Slims each side to only the
    columns actually needed before merging (rather than merging the full frames) so that columns
    shared identically by both sides - prev_month_return/target/report_date - don't get spuriously
    `_x`/`_y` suffixed by pandas' default merge collision handling; the Spark version avoids this
    the same way, by selecting df1's copy of those columns after the join.
    """
    left = df1.rename(columns={"feature": "feature_x"})
    right = df2[["fsym_id", "sector_return_date", "feature"]].rename(columns={"feature": "feature_y"})
    merged = left.merge(right, on=["fsym_id", "sector_return_date"], how="inner")

    if operator == "+":
        combined = merged["feature_x"] + merged["feature_y"]
    elif operator == "-":
        combined = merged["feature_x"] - merged["feature_y"]
    elif operator == "*":
        combined = merged["feature_x"] * merged["feature_y"]
    elif operator == "/":
        combined = merged["feature_x"].to_numpy() / _safe_denominator_local(merged["feature_y"])
    else:
        raise ValueError(f"Unsupported operator: {operator}")

    return merged.assign(feature=combined)[list(SCHEMA_COLUMNS)]


def winsorize_feature_local(
    train_df: pd.DataFrame, test_df: pd.DataFrame, column: str = "feature",
    lower_q: float = 0.01, upper_q: float = 0.99,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """
    Pandas equivalent of `arithmetic.py`'s `winsorize_feature` - clips `column` to the
    [lower_q, upper_q] quantile bounds of TRAIN data only, applied to both frames. Note:
    `pandas.Series.quantile` is EXACT, unlike the Spark original's `approxQuantile`
    (`relative_error=0.001`) - a real, understood, immaterial-at-this-data-scale difference from
    the Spark path, not a bug to reconcile.
    """
    lower, upper = train_df[column].quantile([lower_q, upper_q])
    clip = lambda df: df.assign(**{column: df[column].clip(lower, upper)})
    return clip(train_df), clip(test_df)
