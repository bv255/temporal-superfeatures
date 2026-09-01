"""
Draws the walk-forward fold structure (training/validation/inner-validation/test windows, and
the 1-month embargo at each split boundary) as a horizontal timeline figure, for use in the
methodology writeup/paper (see docs/METHODOLOGY.md's "Walk-forward structure" section and the
Expanding Window Walk-Forward Validation writeup).

Fold boundaries are computed by calling the REAL superfeatures.evaluation.splits.compute_fold_boundaries
function against the REAL base checkpoint (walk_forward_full/base/experiment_1_df.parquet on HDFS)
- not hand-copied arithmetic or synthetic dates - so this figure can't drift from what
run_preprocessing.py would actually produce. This matters at the edges specifically: the real data
does NOT span full calendar years (confirmed by querying the checkpoint directly: 2001-03-01 to
2026-07-01, not 2001-01-01 to 2026-12-31), so every fold's nominal calendar-year boundaries (from
compute_fold_boundaries, which only reasons in whole years) are clipped here to the true min/max
target_date at the two outer edges of the whole diagram - the first training bar's start and the
last final-test fold's test bar's end - so the last fold's test window is drawn at its true ~1.5-year
extent, not a misleadingly full 2 years.

Uses FULL_CONFIG's real data (walk_forward_namespace="walk_forward_full", so walk_forward_base_path
points at the real full-scale checkpoint) but with target_dev_folds overridden to 5 (DIAGRAM_CONFIG
below) rather than FULL_CONFIG's own value of 3 - this diagram illustrates the walk-forward
structure generically (see docs/METHODOLOGY.md's "Walk-forward structure" section), and 5 is the
dev-fold count actually exercised day-to-day now (--fast's FAST_CONFIG.target_dev_folds, used by
run_ga.py's --gbt-tree-search/--max-features-search sweeps) - not a claim that any single real
run_ga.py invocation of FULL_CONFIG itself produces 5 development folds; it still only produces 3.
target_final_test_folds is left at its FULL_CONFIG default (5) unchanged, so only the development
section of the figure differs from what a literal FULL_CONFIG run would produce - same mechanism
described in config.py's FAST_CONFIG comment for why growing target_dev_folds alone only ever eats
into the initial-training window, never the final-test region.

Needs a local Spark session with the real Hadoop conf pointed at HDFS (see CLAUDE.md's PySpark
env vars) - reads the base checkpoint once (a single min/max/distinct-years aggregation over an
already-materialized parquet, not a heavy job) - so this needs to run on bialobog, not a laptop.

Usage: ~/pyspark-venv/bin/python3 scripts/plot_walk_forward_folds.py
Output: scripts/output/walk_forward_folds.pdf and .png
"""
import os

os.environ.setdefault("SPARK_HOME", "/opt/spark")
os.environ.setdefault("HADOOP_HOME", "/opt/hadoop")
os.environ.setdefault("HADOOP_CONF_DIR", "/opt/hadoop-3.4.1/etc/hadoop")
os.environ.setdefault("YARN_CONF_DIR", "/opt/hadoop-3.4.1/etc/hadoop")
os.environ.setdefault("PYSPARK_PYTHON", "/home/bvail/pyspark-venv/bin/python3")
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", "/home/bvail/pyspark-venv/bin/python3")

import dataclasses
from datetime import date

import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

from superfeatures.config import FULL_CONFIG
from superfeatures.evaluation.splits import compute_fold_boundaries

REPO_ROOT = "/home/bvail/temporal-superfeatures"
OUTPUT_DIR = f"{REPO_ROOT}/scripts/output"

# 5 dev folds (not FULL_CONFIG's own 3) - see module docstring above for why.
DIAGRAM_CONFIG = dataclasses.replace(FULL_CONFIG, target_dev_folds=5)

EMBARGO_MONTHS = DIAGRAM_CONFIG.embargo_months  # currently 1 - drawn as a notch, not to scale

COLOR_TRAIN = "#4C72B0"
COLOR_VAL = "#DD8452"       # dev-fold validation AND final-test inner-validation share this color
COLOR_TEST = "#55A868"
COLOR_EMBARGO = "#C44E52"
COLOR_DATA_BOUND = "#555555"

BAR_HEIGHT = 0.6


def _build_local_spark() -> SparkSession:
    return (
        SparkSession.builder
        .master("local[2]")
        .appName("plot-walk-forward-folds")
        .getOrCreate()
    )


def _get_fold_boundaries():
    spark = _build_local_spark()
    try:
        df = spark.read.parquet(DIAGRAM_CONFIG.walk_forward_base_path)
        return compute_fold_boundaries(df, DIAGRAM_CONFIG)
    finally:
        spark.stop()


def _to_decimal_year(d: date) -> float:
    year_start = date(d.year, 1, 1)
    year_end = date(d.year + 1, 1, 1)
    return d.year + (d - year_start).days / (year_end - year_start).days


def _draw_embargo_notch(ax, x_year, y_center):
    """A 1-month-wide notch isn't visible at a multi-year x-scale, so draw a small fixed-width
    marker instead - width is illustrative, not to scale (see EMBARGO_MONTHS)."""
    notch_width = 0.15
    ax.add_patch(plt.Rectangle(
        (x_year - notch_width / 2, y_center - BAR_HEIGHT / 2 - 0.05),
        notch_width, BAR_HEIGHT + 0.1,
        facecolor=COLOR_EMBARGO, edgecolor="none", zorder=5,
    ))


def plot_folds(result: dict, output_path_no_ext: str):
    dev_folds = result["dev_folds"]
    final_test_folds = result["final_test_folds"]
    n_rows = len(dev_folds) + len(final_test_folds)

    # True data extent - every fold's train start is clipped to this (real data doesn't begin on
    # Jan 1 of the first year), and only the LAST final-test fold's test end is clipped to this
    # (every other fold's own boundaries are interior years with full data coverage).
    min_x = _to_decimal_year(result["min_target_date"])
    max_x = _to_decimal_year(result["max_target_date"])
    last_final_test_fold_index = final_test_folds[-1]["fold_index"]

    fig, ax = plt.subplots(figsize=(11, 0.6 * n_rows + 1.5))

    row_labels = []
    y = n_rows  # top row first

    def _draw_segment(train_years, val_years, test_years, y_pos, is_last_final_test_fold):
        train_start, train_end = train_years
        train_x0 = min_x  # every training window starts at the true beginning of the data
        ax.broken_barh([(train_x0, train_end + 1 - train_x0)], (y_pos - BAR_HEIGHT / 2, BAR_HEIGHT),
                        facecolors=COLOR_TRAIN)
        _draw_embargo_notch(ax, train_end + 1, y_pos)

        val_start, val_end = val_years
        # val_years is always a validation segment - a dev fold's plain validation window, or a
        # final-test fold's inner-validation window (see METHODOLOGY.md §2.6) - never the true
        # test window, which is drawn separately below with COLOR_TEST. Always COLOR_VAL.
        ax.broken_barh([(val_start, val_end - val_start + 1)], (y_pos - BAR_HEIGHT / 2, BAR_HEIGHT),
                        facecolors=COLOR_VAL)

        if test_years is not None:
            _draw_embargo_notch(ax, val_end + 1, y_pos)
            test_start, test_end = test_years
            test_x1 = max_x if is_last_final_test_fold else test_end + 1
            ax.broken_barh([(test_start, test_x1 - test_start)], (y_pos - BAR_HEIGHT / 2, BAR_HEIGHT),
                            facecolors=COLOR_TEST)

    for fold in dev_folds:
        _draw_segment(fold["train_years"], fold["eval_years"], None, y, False)
        row_labels.append(f"Dev {fold['fold_index']}")
        y -= 1

    y -= 0.5  # gap between sections

    for fold in final_test_folds:
        is_last = fold["fold_index"] == last_final_test_fold_index
        _draw_segment(fold["train_years"], fold["inner_val_years"], fold["eval_years"], y, is_last)
        row_labels.append(f"Final-test {fold['fold_index']}")
        y -= 1

    y_ticks = [n_rows - i for i in range(len(dev_folds))] + \
              [n_rows - len(dev_folds) - 0.5 - i for i in range(len(final_test_folds))]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(row_labels)
    ax.set_ylim(y - 1, n_rows + 1)

    # Reference lines at the true data boundaries, with exact dates labeled - every bar's edges
    # are computed from calendar years, but the underlying data doesn't start/end on Jan 1/Dec 31.
    # Manual override on the max-date label text only (line position still comes from the real
    # max_target_date) - TODO: revisit whether the real value (2026-07-01) or this (2026-06-30)
    # is the one that belongs here.
    MAX_DATE_LABEL_OVERRIDE = "2026-06-30"
    for x, label, ha in ((min_x, result["min_target_date"].isoformat(), "left"), (max_x, MAX_DATE_LABEL_OVERRIDE, "right")):
        ax.axvline(x, color=COLOR_DATA_BOUND, linestyle="--", linewidth=1, zorder=1)
        ax.annotate(label, xy=(x, n_rows + 0.6), ha=ha, va="bottom",
                    fontsize=8, color=COLOR_DATA_BOUND, rotation=0, annotation_clip=False)

    ax.set_xlim(min_x - 0.5, max_x + 0.5)
    ax.set_xlabel("Year")
    ax.set_title("Expanding-Window Walk-Forward Fold Structure")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_TRAIN, label="Training (expanding)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_VAL, label="Validation / inner-validation"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_TEST, label="Test"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_EMBARGO, label=f"{EMBARGO_MONTHS}-month embargo (not to scale)"),
        plt.Line2D([0], [0], color=COLOR_DATA_BOUND, linestyle="--", label="True data start/end"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)

    fig.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(f"{output_path_no_ext}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_path_no_ext}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path_no_ext}.pdf and .png")


if __name__ == "__main__":
    result = _get_fold_boundaries()
    plot_folds(result, f"{OUTPUT_DIR}/walk_forward_folds")
