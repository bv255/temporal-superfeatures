"""
`GAPreprocessing._feature_frame` must carry the fold's `date` column through as `report_date`
for live temporal operators to window over (TEMPORAL_SUBTREE_OPERATORS_PROMPT.md section 2),
without it leaking into `get_feature_list()`'s candidate pool.
"""
from datetime import date

import pytest
from pyspark.sql import Row, SparkSession

from superfeatures.ga import GAPreprocessing


@pytest.fixture(scope="module")
def spark():
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("test_ga_preprocessing_report_date")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()


def _bare_gapreprocessing(spark, df, selected_features):
    gp = GAPreprocessing.__new__(GAPreprocessing)
    gp.spark = spark
    gp.train_df = df
    gp.eval_df = df
    gp.true_eval_df = df
    gp.selected_features = selected_features
    return gp


def test_feature_frame_carries_report_date(spark):
    df = spark.createDataFrame([
        Row(fsym="F1", date=date(2020, 1, 1), target_date=date(2020, 1, 1),
            prev_month_return=0.01, prev_month_sector_return=0.02,
            monthly_return=0.05, feat_a=1.0),
    ])
    gp = _bare_gapreprocessing(spark, df, ["feat_a", "prev_month_return", "prev_month_sector_return"])
    result = gp._feature_frame(df, "feat_a")
    row = result.first()
    assert "report_date" in result.columns
    assert row["report_date"] == date(2020, 1, 1)


def test_report_date_never_leaks_into_feature_list(spark):
    df = spark.createDataFrame([
        Row(fsym="F1", date=date(2020, 1, 1), target_date=date(2020, 1, 1),
            prev_month_return=0.01, prev_month_sector_return=0.02,
            monthly_return=0.05, feat_a=1.0),
    ])
    gp = _bare_gapreprocessing(spark, df, ["feat_a", "prev_month_return", "prev_month_sector_return"])
    assert "report_date" not in gp.get_feature_list()
    assert gp.get_feature_list() == ["feat_a"]


def test_feature_frame_dropna_ignores_null_report_date(spark):
    df = spark.createDataFrame([
        Row(fsym="F1", date=None, target_date=date(2020, 1, 1),
            prev_month_return=0.01, prev_month_sector_return=0.02,
            monthly_return=0.05, feat_a=1.0),
    ], schema="fsym string, date date, target_date date, prev_month_return double, "
              "prev_month_sector_return double, monthly_return double, feat_a double")
    gp = _bare_gapreprocessing(spark, df, ["feat_a"])
    result = gp._feature_frame(df, "feat_a")
    assert result.count() == 1
    assert result.first()["report_date"] is None
