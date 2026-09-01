"""
Tests for `superfeatures.operators.temporal.apply_temporal` against a synthetic
report-grain, as-of-expanded frame - local SparkSession only, same fixture style as
`test_ga_engine_parity.py`. This is also the automated version of
TEMPORAL_SUBTREE_OPERATORS_PROMPT.md section 2 step 4's spot-check and the retirement gate
from section 9 open question 1: a direct value comparison against
`Utils.add_temporal_features`'s output on identical report-level data.
"""
from datetime import date

import pytest
from pyspark.sql import Row, SparkSession

from superfeatures.operators.temporal import apply_temporal, temporal_ops
from superfeatures.operators.arithmetic import combine_dataframes
from superfeatures.preprocessing.utils import Utils


@pytest.fixture(scope="module")
def spark():
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("test_ga_temporal_evaluator")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()


# Two entities, three reports each, each report covering 2 calendar months (mirrors the
# as-of expansion's duplicated-report-date-across-months shape).
REPORTS = {
    "F1": [(date(2020, 1, 1), 10.0), (date(2020, 4, 1), 12.0), (date(2020, 7, 1), 9.0)],
    "F2": [(date(2020, 2, 1), 100.0), (date(2020, 5, 1), 80.0), (date(2020, 8, 1), 120.0)],
}
COVERAGE_MONTHS_PER_REPORT = 2


def _expanded_frame(spark):
    rows = []
    for fsym, reports in REPORTS.items():
        for report_date, value in reports:
            for m in range(COVERAGE_MONTHS_PER_REPORT):
                year = report_date.year + (report_date.month - 1 + m) // 12
                month = (report_date.month - 1 + m) % 12 + 1
                rows.append(Row(
                    fsym_id=fsym,
                    sector_return_date=date(year, month, 1),
                    prev_month_return=0.01,
                    prev_month_sector_return=0.02,
                    feature=value,
                    target=0.05,
                    report_date=report_date,
                ))
    return spark.createDataFrame(rows)


def _report_level_frame(spark):
    rows = []
    for fsym, reports in REPORTS.items():
        for report_date, value in reports:
            rows.append(Row(fsym=fsym, date=report_date, feature=value))
    return spark.createDataFrame(rows)


def _collect_by_report(df, value_col="feature"):
    return {
        (r["fsym_id"], r["report_date"]): r[value_col]
        for r in df.select("fsym_id", "report_date", value_col).dropDuplicates(["fsym_id", "report_date"]).collect()
    }


class TestApplyTemporalCorrectness:
    def test_lag1_broadcasts_across_coverage_window_not_monthly(self, spark):
        df = _expanded_frame(spark)
        op = temporal_ops()[0]  # lag1
        assert op.name == "lag1"
        result = apply_temporal(df, op)

        by_month = {(r["fsym_id"], r["sector_return_date"]): r["feature"] for r in result.collect()}
        # Both months in F1's 2nd report's coverage window (2020-04 and 2020-05) must carry
        # the SAME lag1 value (the 1st report's 10.0) - proof the window operated on the
        # de-duplicated report grain, not the monthly-expanded grain.
        assert by_month[("F1", date(2020, 4, 1))] == 10.0
        assert by_month[("F1", date(2020, 5, 1))] == 10.0

    def test_lag1_null_for_first_report(self, spark):
        df = _expanded_frame(spark)
        result = apply_temporal(df, temporal_ops()[0])
        by_month = {(r["fsym_id"], r["sector_return_date"]): r["feature"] for r in result.collect()}
        assert by_month[("F1", date(2020, 1, 1))] is None
        assert by_month[("F1", date(2020, 2, 1))] is None

    def test_delta1_matches_expected_difference(self, spark):
        df = _expanded_frame(spark)
        delta_op = next(op for op in temporal_ops() if op.name == "delta1")
        result = apply_temporal(df, delta_op)
        by_report = _collect_by_report(result)
        assert by_report[("F1", date(2020, 4, 1))] == pytest.approx(12.0 - 10.0)
        assert by_report[("F1", date(2020, 1, 1))] is None

    def test_growth1_matches_expected_ratio_and_zero_floor(self, spark):
        df = _expanded_frame(spark)
        growth_op = next(op for op in temporal_ops() if op.name == "growth1")
        result = apply_temporal(df, growth_op)
        by_report = _collect_by_report(result)
        expected = (12.0 - 10.0) / 10.0
        assert by_report[("F1", date(2020, 4, 1))] == pytest.approx(expected)

    def test_mean3_never_null_partial_window(self, spark):
        df = _expanded_frame(spark)
        mean_op = next(op for op in temporal_ops() if op.name == "mean3")
        result = apply_temporal(df, mean_op)
        by_report = _collect_by_report(result)
        assert by_report[("F1", date(2020, 1, 1))] == pytest.approx(10.0)
        assert by_report[("F1", date(2020, 4, 1))] == pytest.approx((10.0 + 12.0) / 2)
        assert by_report[("F1", date(2020, 7, 1))] == pytest.approx((10.0 + 12.0 + 9.0) / 3)

    def test_std3_null_with_fewer_than_two_values(self, spark):
        df = _expanded_frame(spark)
        std_op = next(op for op in temporal_ops() if op.name == "std3")
        result = apply_temporal(df, std_op)
        by_report = _collect_by_report(result)
        assert by_report[("F1", date(2020, 1, 1))] is None
        assert by_report[("F1", date(2020, 4, 1))] is not None

    def test_row_count_preserved(self, spark):
        df = _expanded_frame(spark)
        result = apply_temporal(df, temporal_ops()[0])
        assert result.count() == df.count()


class TestCombineDataframesCarriesReportDate:
    def test_addition_preserves_report_date(self, spark):
        df = _expanded_frame(spark)
        result = combine_dataframes(df, df, "+")
        row = result.filter(result.fsym_id == "F1").first()
        assert row["report_date"] is not None
        assert row["feature"] == pytest.approx(20.0) or row["feature"] == pytest.approx(24.0) or row["feature"] == pytest.approx(18.0)


class TestMatchesAddTemporalFeatures:
    """The retirement gate: apply_temporal's live values must match
    Utils.add_temporal_features's precomputed values exactly, on identical report-level data."""

    @pytest.mark.parametrize("op_name,suffix", [
        ("lag1", "_lag1"), ("delta1", "_delta1"), ("growth1", "_growth1"),
        ("mean3", "_mean3"), ("std3", "_std3"),
    ])
    def test_live_atom_matches_precomputed_column(self, spark, op_name, suffix):
        report_df = _report_level_frame(spark)
        precomputed_df, new_cols = Utils.add_temporal_features(report_df, ["feature"], lag_periods=[1], window_sizes=[3])
        precomputed_col = f"feature{suffix}"
        assert precomputed_col in new_cols
        precomputed = {
            (r["fsym"], r["date"]): r[precomputed_col]
            for r in precomputed_df.select("fsym", "date", precomputed_col).collect()
        }

        expanded_df = _expanded_frame(spark)
        op = next(o for o in temporal_ops() if o.name == op_name)
        live_result = apply_temporal(expanded_df, op)
        live = _collect_by_report(live_result)

        for fsym, reports in REPORTS.items():
            for report_date, _ in reports:
                expected = precomputed[(fsym, report_date)]
                actual = live[(fsym, report_date)]
                if expected is None:
                    assert actual is None, f"{fsym}/{report_date}: expected null, got {actual}"
                else:
                    assert actual == pytest.approx(expected), f"{fsym}/{report_date}: {actual} != {expected}"
