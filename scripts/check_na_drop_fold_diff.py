"""
Standalone check: does drop_na_features (30% missingness threshold, PipelineConfig's
expanding walk-forward train window) ever drop a feature in an EARLIER fold but keep it in a
LATER fold, or vice versa? Answers the question from the walk_forward_full preprocessing run
without re-running the whole pipeline - the base checkpoint (walk_forward_full/base/
experiment_1_df.parquet) is read once and each fold's train slice is re-filtered from it, same
as preprocessing/pipeline.py's run_fold() does, but skipping straight to
Utils.drop_na_features() instead of running the full consensus/clustering/cap machinery.

Reads each fold's train_years/effective_train_cutoff straight from the fold_metadata.json
already on disk (research/walk_forward_full/{development,final_test}/fold_*/fold_metadata.json)
rather than recomputing walk-forward boundaries, so this can't drift from what actually ran.

Usage: ~/pyspark-venv/bin/python3 scripts/check_na_drop_fold_diff.py
"""
import json
import glob
from datetime import date

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from superfeatures.preprocessing import Utils

REPO_ROOT = "/home/bvail/temporal-superfeatures"
BASE_PARQUET_PATH = "walk_forward_full/base/experiment_1_df.parquet"  # HDFS-relative, same as PipelineConfig.walk_forward_base_path
DROP_THRESHOLD = 0.3  # matches pipeline.py's run_fold() call
ALWAYS_KEEP_FEATURES = ['prev_month_return', 'prev_month_sector_return']  # exempt from drop_na_features, same as pipeline.py
NON_FEATURE_COLS = {
    'fsym', 'date', 'factset_sector_desc', 'target_date', 'return_year', 'return_month',
    'monthly_return', 'fsym_id', 'fund_report_date',
}
OUTPUT_DIR = f"{REPO_ROOT}/scripts/output"


def _build_spark_session() -> SparkSession:
    import os
    os.environ["SPARK_HOME"] = "/opt/spark"
    os.environ["HADOOP_HOME"] = "/opt/hadoop"
    os.environ["HADOOP_CONF_DIR"] = "/opt/hadoop-3.4.1/etc/hadoop"
    os.environ["YARN_CONF_DIR"] = "/opt/hadoop-3.4.1/etc/hadoop"
    os.environ["PYSPARK_PYTHON"] = "/home/bvail/pyspark-venv/bin/python3"
    os.environ["PYSPARK_DRIVER_PYTHON"] = "/home/bvail/pyspark-venv/bin/python3"

    print("Starting SparkSession (this can take a moment on YARN)...")
    spark = (
        SparkSession.builder
        .master("yarn")
        .appName("SuperFeatures_check_na_drop_fold_diff")
        # Lighter than run_preprocessing.py's 10x2x8g - this job only does chunked null-count
        # scans (no RF permutation importance / mutual information / clustering), shared infra
        # so keep it modest per CLAUDE.md.
        .config("spark.executor.instances", "6")
        .config("spark.executor.cores", "2")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        # Same HDFS block-placement workaround as run_preprocessing.py/run_ga.py - irrelevant
        # here since this script never writes to HDFS, kept only for parity/safety.
        .config("spark.hadoop.dfs.client.block.write.retries", "10")
        .config("spark.hadoop.dfs.replication", "1")
        .getOrCreate()
    )
    print("SparkSession ready.")
    return spark


def _load_folds() -> list[dict]:
    """One entry per fold_metadata.json under walk_forward_full/{development,final_test}/."""
    paths = sorted(glob.glob(f"{REPO_ROOT}/research/walk_forward_full/development/fold_*/fold_metadata.json")) + \
        sorted(glob.glob(f"{REPO_ROOT}/research/walk_forward_full/final_test/fold_*/fold_metadata.json"))
    folds = []
    for p in paths:
        meta = json.load(open(p))
        label = meta['output_dir'].replace('walk_forward_full/', '')  # e.g. "development/fold_01"
        train_start_year = meta['train_years'][0]
        effective_train_cutoff = date.fromisoformat(meta['effective_train_cutoff'])
        folds.append({
            'label': label,
            'train_years': tuple(meta['train_years']),
            'train_start_date': date(train_start_year, 1, 1),
            'effective_train_cutoff': effective_train_cutoff,
            'n_candidate_features': meta['n_candidate_features'],
            'n_selectable_features': meta['n_selectable_features'],
            'n_post_drop_na_features_expected': meta['n_post_drop_na_features'],
        })
    # Chronological order by training window end (ties broken by label so development sorts
    # before final_test for an identical train window, e.g. development/fold_03 and
    # final_test/fold_01 both train on [2001, 2019]).
    folds.sort(key=lambda f: (f['effective_train_cutoff'], f['label']))
    return folds


def main():
    folds = _load_folds()
    print(f"Loaded {len(folds)} folds: {[f['label'] for f in folds]}")

    spark = _build_spark_session()
    try:
        wf_base_df = spark.read.parquet(BASE_PARQUET_PATH).cache()
        total_rows = wf_base_df.count()
        print(f"wf_base_df read, rows={total_rows}, columns={len(wf_base_df.columns)}")

        candidate_feature_columns = [c for c in wf_base_df.columns if c not in NON_FEATURE_COLS]
        selectable_candidates = [c for c in candidate_feature_columns if c not in ALWAYS_KEEP_FEATURES]
        print(f"Derived {len(candidate_feature_columns)} candidate features "
              f"({len(selectable_candidates)} selectable, {len(ALWAYS_KEEP_FEATURES)} always-kept)")

        dropped_by_fold = {}  # label -> set of dropped feature names
        for fold in folds:
            print(f"\n=== {fold['label']} (train {fold['train_years'][0]}-{fold['train_years'][1]}, "
                  f"cutoff {fold['effective_train_cutoff']}) ===")

            if fold['n_candidate_features'] != len(candidate_feature_columns) or \
                    fold['n_selectable_features'] != len(selectable_candidates):
                print(f"  WARNING: candidate/selectable counts don't match fold_metadata.json "
                      f"(expected {fold['n_candidate_features']}/{fold['n_selectable_features']}, "
                      f"derived {len(candidate_feature_columns)}/{len(selectable_candidates)}) - "
                      f"NON_FEATURE_COLS may not match this run.")

            train_df = wf_base_df.filter(
                (F.col('target_date') >= fold['train_start_date']) &
                (F.col('target_date') <= fold['effective_train_cutoff'])
            )
            train_features_view = train_df.select('fsym', 'date', *selectable_candidates)
            _, dropped_na_features, _ = Utils.drop_na_features(train_features_view, drop_threshold=DROP_THRESHOLD)
            dropped_by_fold[fold['label']] = set(dropped_na_features)

            expected = fold['n_post_drop_na_features_expected']
            actual = len(selectable_candidates) - len(dropped_na_features)
            match = "OK" if actual == expected else "MISMATCH"
            print(f"  dropped {len(dropped_na_features)} features, {actual} survive "
                  f"(fold_metadata.json expected {expected}) [{match}]")

        # --- cross-fold membership table: feature x fold, True = dropped in that fold ---
        fold_labels = [f['label'] for f in folds]
        all_dropped_anywhere = sorted(set().union(*dropped_by_fold.values()))
        print(f"\n{len(all_dropped_anywhere)} distinct features dropped by NA threshold in at least one fold")

        import csv
        import os
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        matrix_path = f"{OUTPUT_DIR}/na_drop_membership.csv"
        with open(matrix_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["feature"] + fold_labels)
            for feat in all_dropped_anywhere:
                writer.writerow([feat] + [feat in dropped_by_fold[lbl] for lbl in fold_labels])
        print(f"Wrote per-fold drop membership matrix to {matrix_path}")

        # --- flag non-monotonic features: dropped, then not, then dropped again (or reverse) ---
        toggled = []
        for feat in all_dropped_anywhere:
            trajectory = [feat in dropped_by_fold[lbl] for lbl in fold_labels]
            transitions = sum(1 for a, b in zip(trajectory, trajectory[1:]) if a != b)
            if transitions > 1:
                toggled.append((feat, trajectory))

        summary_path = f"{OUTPUT_DIR}/na_drop_summary.txt"
        with open(summary_path, "w") as f:
            def out(line=""):
                print(line)
                f.write(line + "\n")

            out("=== fold order (chronological) ===")
            for lbl in fold_labels:
                out(f"  {lbl}")

            out("\n=== features dropped in an EARLIER fold but present again in a LATER fold "
                "(or dropped, kept, dropped again - i.e. non-monotonic) ===")
            if not toggled:
                out("  none found - every NA-dropped feature's drop status only ever moves in "
                    "one direction across the expanding window (or never changes).")
            else:
                for feat, trajectory in toggled:
                    marks = "".join("X" if t else "." for t in trajectory)
                    out(f"  {feat:40s} {marks}   (X = dropped that fold)")

            out(f"\nFull per-fold membership matrix: {matrix_path}")

        print(f"\nWrote summary to {summary_path}")
    finally:
        print("Stopping SparkSession...")
        spark.stop()
        print("SparkSession stopped.")


if __name__ == "__main__":
    main()
