"""
`combine_dataframes`: the GA genome's `+ - * /` arithmetic operator (with the `1e-6`
zero-denominator division guard), the sibling of `operators/temporal.py`'s lag/delta/growth/
mean/std operators. De-duplicated out of 3 verbatim copies in `ga/engine.py`
(TEMPORAL_SUBTREE_OPERATORS_PROMPT.md section 6).

Operates on the evaluator's standard per-leaf frame shape: `fsym_id, sector_return_date,
prev_month_return, prev_month_sector_return, feature, target, report_date` - one row per
`(fsym_id, sector_return_date)` (a monthly, as-of-expanded grain).
"""
import pyspark.sql.functions as F
from pyspark.sql import DataFrame


DIVISION_GUARD_EPS = 1e-6


def _safe_denominator(denom, eps: float = DIVISION_GUARD_EPS):
    """
    Floor a denominator away from zero without flipping its sign. A bare positive floor (the
    original `F.when(denom == 0, eps).otherwise(denom)`) only caught the exact-zero case; a
    near-zero-but-nonzero denominator (common after subtracting two close fundamentals) passed
    through untouched and could blow the ratio up by many orders of magnitude. Substituting a
    bare positive eps for every denominator in (-eps, eps) would be a NEW bug, not a fix: it
    would silently flip the sign of results whose true denominator was small-but-negative (e.g.
    -9e-7 today produces an extreme-but-correctly-signed result; flooring it to +eps would give a
    bounded-magnitude, WRONG-SIGN result instead) - exactly the band this guard exists to newly
    catch. F.signum(denom) preserves sign for the near-zero-nonzero case; exact zero
    (signum(0) == 0, which would otherwise floor to 0 * eps == 0, dividing by zero again) keeps
    the original convention of flooring to +eps.
    """
    return F.when(
        F.abs(denom) < eps,
        F.when(denom == 0, F.lit(eps)).otherwise(F.signum(denom) * F.lit(eps)),
    ).otherwise(denom)


def combine_dataframes(df1: DataFrame, df2: DataFrame, operator: str) -> DataFrame:
    df1 = df1.withColumnRenamed("feature", "feature_x")
    df2 = df2.withColumnRenamed("feature", "feature_y")
    joined = df1.join(df2, on=["fsym_id", "sector_return_date"])

    if operator == "+":
        combined = df1["feature_x"] + df2["feature_y"]
    elif operator == "-":
        combined = df1["feature_x"] - df2["feature_y"]
    elif operator == "*":
        combined = df1["feature_x"] * df2["feature_y"]
    elif operator == "/":
        combined = df1["feature_x"] / _safe_denominator(df2["feature_y"])
    else:
        raise ValueError(f"Unsupported operator: {operator}")

    return joined.select(
        df1["fsym_id"],
        df1["sector_return_date"],
        df1["prev_month_return"],
        df1["prev_month_sector_return"],
        combined.alias("feature"),
        df1["target"],
        df1["report_date"],
    )


def winsorize_feature(
    train_df: DataFrame, test_df: DataFrame, column: str = "feature",
    lower_q: float = 0.01, upper_q: float = 0.99, relative_error: float = 0.001,
) -> "tuple[DataFrame, DataFrame]":
    """
    Clip `column` to the [lower_q, upper_q] quantile bounds of TRAIN data only (no test-set
    leakage), applied to both train_df and test_df. A single exploding value from a division or
    growth computation previously flowed straight into GBT fitting uncapped; this caps it before
    that happens. Returns (train_df, test_df) with `column` clipped - every other column
    untouched.
    """
    lower, upper = train_df.approxQuantile(column, [lower_q, upper_q], relative_error)
    clip = lambda df: df.withColumn(
        column, F.greatest(F.lit(lower), F.least(F.lit(upper), F.col(column)))
    )
    return clip(train_df), clip(test_df)
