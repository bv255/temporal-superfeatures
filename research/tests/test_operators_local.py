"""
Unit tests for `superfeatures.operators.arithmetic_local`/`temporal_local` - the pandas siblings
of `arithmetic.py`/`temporal.py` used by `GeneticAlgorithm1`'s `fit_backend="local"` path. No
SparkSession needed - pure Python/pandas, exact-value assertions against small hand-built frames
(not "close enough" - these are meant to reproduce the Spark originals' formulas exactly, only the
execution engine differs).
"""
import numpy as np
import pandas as pd
import pytest

from superfeatures.operators.arithmetic_local import (
    combine_dataframes_local, winsorize_feature_local, _safe_denominator_local, SCHEMA_COLUMNS,
)
from superfeatures.operators.temporal_local import apply_temporal_local
from superfeatures.operators.temporal import TemporalOpSpec


def _leaf_frame(feature_values, entity="E1", start_date="2020-01-01", freq="MS", report_dates=None):
    """
    Build a small standard-shape leaf frame: one row per (entity, sector_return_date), with
    report_date optionally repeating across several monthly rows (the exact scenario the
    dedup-then-broadcast-back strategy in apply_temporal_local exists to handle).
    """
    n = len(feature_values)
    dates = pd.date_range(start_date, periods=n, freq=freq)
    if report_dates is None:
        report_dates = dates
    return pd.DataFrame({
        "fsym_id": [entity] * n,
        "sector_return_date": dates,
        "prev_month_return": np.zeros(n),
        "prev_month_sector_return": np.zeros(n),
        "feature": feature_values,
        "target": np.zeros(n),
        "report_date": report_dates,
    })


# ---- combine_dataframes_local ----

class TestCombineDataframesLocal:
    def _pair(self):
        df1 = _leaf_frame([1.0, 2.0, 3.0])
        df2 = df1.assign(feature=[10.0, 20.0, 30.0])
        return df1, df2

    def test_addition(self):
        df1, df2 = self._pair()
        result = combine_dataframes_local(df1, df2, "+")
        assert list(result["feature"]) == [11.0, 22.0, 33.0]
        assert list(result.columns) == list(SCHEMA_COLUMNS)

    def test_subtraction(self):
        df1, df2 = self._pair()
        result = combine_dataframes_local(df1, df2, "-")
        assert list(result["feature"]) == [-9.0, -18.0, -27.0]

    def test_multiplication(self):
        df1, df2 = self._pair()
        result = combine_dataframes_local(df1, df2, "*")
        assert list(result["feature"]) == [10.0, 40.0, 90.0]

    def test_division(self):
        df1, df2 = self._pair()
        result = combine_dataframes_local(df1, df2, "/")
        assert result["feature"].tolist() == pytest.approx([0.1, 0.1, 0.1])

    def test_unsupported_operator_raises(self):
        df1, df2 = self._pair()
        with pytest.raises(ValueError):
            combine_dataframes_local(df1, df2, "%")

    def test_shared_columns_not_suffixed(self):
        # prev_month_return/target/report_date are identical on both sides - a naive pd.merge of
        # the full frames would suffix them _x/_y; combine_dataframes_local must not do that.
        df1, df2 = self._pair()
        result = combine_dataframes_local(df1, df2, "+")
        assert "prev_month_return_x" not in result.columns
        assert "prev_month_return" in result.columns


class TestSafeDenominatorLocal:
    def test_normal_values_untouched(self):
        out = _safe_denominator_local(pd.Series([1.0, -5.0, 100.0]))
        assert list(out) == [1.0, -5.0, 100.0]

    def test_exact_zero_floors_to_positive_eps(self):
        out = _safe_denominator_local(pd.Series([0.0]))
        assert out[0] == pytest.approx(1e-6)

    def test_small_negative_preserves_sign(self):
        # A near-zero-but-nonzero NEGATIVE denominator must floor to -eps, not +eps - flipping
        # the sign here would silently invert the result (see arithmetic.py's own docstring).
        out = _safe_denominator_local(pd.Series([-9e-7]))
        assert out[0] == pytest.approx(-1e-6)

    def test_small_positive_preserves_sign(self):
        out = _safe_denominator_local(pd.Series([9e-7]))
        assert out[0] == pytest.approx(1e-6)


class TestWinsorizeFeatureLocal:
    def test_clips_outlier_to_train_quantile_bounds(self):
        train = _leaf_frame([1.0, 2.0, 3.0, 4.0, 1000.0])  # 1000.0 is a wild outlier
        test = _leaf_frame([2.5, 999.0])
        train_out, test_out = winsorize_feature_local(train, test, lower_q=0.01, upper_q=0.99)
        assert train_out["feature"].max() < 1000.0
        assert test_out["feature"].max() < 999.0

    def test_bounds_come_from_train_only_no_test_leakage(self):
        train = _leaf_frame([1.0, 2.0, 3.0])
        test = _leaf_frame([-500.0, 500.0])
        train_out, test_out = winsorize_feature_local(train, test)
        lower, upper = train["feature"].quantile([0.01, 0.99])
        assert test_out["feature"].min() == pytest.approx(lower)
        assert test_out["feature"].max() == pytest.approx(upper)

    def test_other_columns_untouched(self):
        train = _leaf_frame([1.0, 2.0, 3.0])
        test = _leaf_frame([1.0, 2.0])
        train_out, _ = winsorize_feature_local(train, test)
        assert list(train_out["target"]) == list(train["target"])


# ---- apply_temporal_local ----

class TestApplyTemporalLocal:
    def _op(self, family, param=1):
        return TemporalOpSpec(name=f"{family}{param}", family=family, param=param,
                               linear=(family in ("lag", "delta", "mean")), suffix=f"_{family}{param}")

    def test_lag(self):
        df = _leaf_frame([10.0, 20.0, 30.0, 40.0])
        result = apply_temporal_local(df, self._op("lag", 1))
        assert result["feature"].isna().iloc[0]
        assert list(result["feature"].iloc[1:]) == [10.0, 20.0, 30.0]

    def test_delta(self):
        df = _leaf_frame([10.0, 20.0, 35.0, 40.0])
        result = apply_temporal_local(df, self._op("delta", 1))
        assert result["feature"].isna().iloc[0]
        assert list(result["feature"].iloc[1:]) == [10.0, 15.0, 5.0]

    def test_growth(self):
        df = _leaf_frame([10.0, 20.0, 5.0])
        result = apply_temporal_local(df, self._op("growth", 1))
        assert result["feature"].isna().iloc[0]
        # (20-10)/10 = 1.0 ; (5-20)/20 = -0.75
        assert result["feature"].iloc[1] == pytest.approx(1.0)
        assert result["feature"].iloc[2] == pytest.approx(-0.75)

    def test_growth_near_zero_denominator_guarded(self):
        df = _leaf_frame([0.0, 5.0])
        result = apply_temporal_local(df, self._op("growth", 1))
        # lag value is exactly 0.0 -> floors to +eps -> (5 - 0) / 1e-6, a large finite number,
        # not inf/nan.
        assert np.isfinite(result["feature"].iloc[1])

    def test_mean_window_shrinks_at_series_start(self):
        # rolling(3, min_periods=1) - matches Spark's rowsBetween(-2, 0) boundary behavior: the
        # window naturally shrinks near the start rather than requiring 3 full observations.
        df = _leaf_frame([3.0, 6.0, 9.0, 12.0])
        result = apply_temporal_local(df, self._op("mean", 3))
        assert result["feature"].iloc[0] == pytest.approx(3.0)         # window: [3]
        assert result["feature"].iloc[1] == pytest.approx(4.5)         # window: [3,6]
        assert result["feature"].iloc[2] == pytest.approx(6.0)         # window: [3,6,9]
        assert result["feature"].iloc[3] == pytest.approx(9.0)         # window: [6,9,12]

    def test_std_uses_sample_stddev_ddof1(self):
        df = _leaf_frame([1.0, 2.0, 3.0])
        result = apply_temporal_local(df, self._op("std", 3))
        expected = pd.Series([1.0, 2.0, 3.0]).std(ddof=1)  # pandas default ddof=1, matches F.stddev
        assert result["feature"].iloc[2] == pytest.approx(expected)

    def test_repeated_report_date_across_monthly_rows_broadcasts_correctly(self):
        # One report covers several monthly rows (report_date repeats); feature is constant
        # across that window by construction. lag1 must be computed over the DEDUPED report
        # grain, not the monthly-expanded grain - a naive per-row lag would treat each repeated
        # date as a distinct position and degenerate into a near no-op.
        df = _leaf_frame(
            feature_values=[100.0, 100.0, 100.0, 200.0, 200.0],
            report_dates=["2020-01-01", "2020-01-01", "2020-01-01", "2020-04-01", "2020-04-01"],
        )
        result = apply_temporal_local(df, self._op("lag", 1))
        # First report (2020-01-01, 3 monthly rows): no prior report -> NaN for all three.
        assert result["feature"].iloc[:3].isna().all()
        # Second report (2020-04-01, 2 monthly rows): lag1 = first report's feature (100.0) for
        # BOTH monthly rows sharing it, not just the first.
        assert list(result["feature"].iloc[3:]) == [100.0, 100.0]

    def test_multi_entity_lag_is_independent_per_entity(self):
        df_a = _leaf_frame([1.0, 2.0, 3.0], entity="A")
        df_b = _leaf_frame([10.0, 20.0, 30.0], entity="B")
        df = pd.concat([df_a, df_b], ignore_index=True)
        result = apply_temporal_local(df, self._op("lag", 1))
        a_result = result[result["fsym_id"] == "A"].sort_values("sector_return_date")["feature"]
        b_result = result[result["fsym_id"] == "B"].sort_values("sector_return_date")["feature"]
        assert list(a_result.iloc[1:]) == [1.0, 2.0]
        assert list(b_result.iloc[1:]) == [10.0, 20.0]

    def test_unsupported_family_raises(self):
        df = _leaf_frame([1.0, 2.0])
        with pytest.raises(ValueError):
            apply_temporal_local(df, self._op("bogus", 1))

    def test_output_schema_matches(self):
        df = _leaf_frame([1.0, 2.0, 3.0])
        result = apply_temporal_local(df, self._op("lag", 1))
        assert list(result.columns) == list(SCHEMA_COLUMNS)
