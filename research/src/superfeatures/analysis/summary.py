"""
Ported from `research/GA_test.ipynb` cells 27/28/29 — the post-hoc, whole-run summary/report
functions (`build_final_test_summary`, `build_pairwise_comparisons`, `build_winner_composition`),
split out of the former `ga/driver.py` (now `ga/algorithms.py`, which keeps the per-fold search
execution). These three operate on a completed list of `fold_results` dicts after every fold is
done - a distinct, post-hoc-shaped concern from executing one fold's search, hence living in
`analysis/` rather than `ga/`. See docs/RESTRUCTURING_TODO.md / the port plan.
"""

import pandas as pd

from .significance import holm_bonferroni, block_bootstrap_ic_delta
from ..config import GAConfig


def build_final_test_summary(fold_results: list, config: "GAConfig"):
    """
    Ported from cell 27 (Deliverable 1 - supersedes cell 26's simpler version, which wrote the
    same output path; cell 26 itself isn't ported separately). Adds baseline A/B/C
    rmse+mean_ic+ic_ir, baseline B/C's winning expression + leaf count, and the GA winner's
    mean_ic/ic_ir/leaf_count. No "v2" columns - GA_test.ipynb has only one GA lineage (see
    CLAUDE.md's "Methodology additions" section for why items 4/6/7 were adapted to a
    single-variant shape).
    """
    final_test_results = [r for r in fold_results if r["category"] == "final_test"]
    if not final_test_results:
        print("No final-test fold results to build the extended summary table from yet.")
        return pd.DataFrame()
    extended_summary_df = pd.DataFrame([
        {
            "fold_name": r["fold_name"],
            "eval_year": r["eval_year"],
            "elapsed_seconds": r["elapsed_seconds"],
            "winner_expression": r["winning_expression"],
            "winner_leaf_count": r["winner_leaf_count"],
            "winner_true_test_rmse": r["true_test_rmse"],
            "winner_mean_ic": r["winner_mean_ic"],
            "winner_ic_ir": r["winner_ic_ir"],
            "baseline_a_rmse": r["baseline_rmse"],
            "baseline_a_mean_ic": r["baseline_a_mean_ic"],
            "baseline_a_ic_ir": r["baseline_a_ic_ir"],
            "baseline_b_expression": r["baseline_b_individual"],
            "baseline_b_leaf_count": r["baseline_b_leaf_count"],
            "baseline_b_rmse": r["baseline_b_rmse"],
            "baseline_b_mean_ic": r["baseline_b_mean_ic"],
            "baseline_b_ic_ir": r["baseline_b_ic_ir"],
            "baseline_c_expression": r["baseline_c_individual"],
            "baseline_c_leaf_count": r["baseline_c_leaf_count"],
            "baseline_c_rmse": r["baseline_c_rmse"],
            "baseline_c_mean_ic": r["baseline_c_mean_ic"],
            "baseline_c_ic_ir": r["baseline_c_ic_ir"],
        }
        for r in final_test_results
    ]).sort_values("eval_year").reset_index(drop=True)

    print("Extended final-test fold results table:")
    print(extended_summary_df.to_string(index=False))

    extended_summary_path = f"{config.output_dir}/final_test_summary.csv"
    extended_summary_df.to_csv(extended_summary_path, index=False)
    print(f"\nWrote {extended_summary_path}")
    return extended_summary_df


def build_pairwise_comparisons(fold_results: list, config: "GAConfig"):
    """
    Ported from cell 28 (Deliverable 2, items 4 + 5), then REDESIGNED to match
    evaluation_framework.md's "secondary baseline comparisons" spec: winner vs baseline A
    (fixed predictors), B (best raw feature), and (C, if config.run_baseline_c, matched-compute
    random search) - this is the "secondary hypothesis" family, tested per arm (this function
    runs once per temporal-ON or temporal-OFF run) and Holm-Bonferroni corrected across that
    arm's own 2-3 comparisons - see docs/EVALUATION.md for why this stays per-arm rather than
    pooled with the other arm's baseline checks.

    Each comparison uses the one-sided, null-centered, within-fold block bootstrap
    (`block_bootstrap_ic_delta`) on matched (fold, month) IC deltas, using the monthly IC series
    each fold already computed inside run_ga_for_fold (no recomputation) - then a single
    Holm-Bonferroni pass across this arm's comparisons' raw p-values, since "GA winner beats
    baseline X" is a directional claim in every case here.
    """
    final_test_results = [r for r in fold_results if r["category"] == "final_test"]
    _comparisons = [
        ("winner_vs_baseline_a", "winner_monthly_ic", "baseline_a_monthly_ic"),
        ("winner_vs_baseline_b", "winner_monthly_ic", "baseline_b_monthly_ic"),
    ]
    if config.run_baseline_c:
        _comparisons.append(("winner_vs_baseline_c", "winner_monthly_ic", "baseline_c_monthly_ic"))

    if not final_test_results:
        print("No final-test fold results to build the pairwise comparison table from yet.")
        return pd.DataFrame()
    pairwise_rows = []
    raw_pvalues = []
    for comparison_name, key_a, key_b in _comparisons:
        fold_month_deltas = {}
        for r in final_test_results:
            ic_a, ic_b = r.get(key_a), r.get(key_b)
            if not ic_a or not ic_b:
                continue
            series_a, series_b = pd.Series(ic_a), pd.Series(ic_b)
            matched_dates = series_a.index.intersection(series_b.index).sort_values()
            if len(matched_dates) == 0:
                continue
            fold_month_deltas[r["fold_name"]] = (series_a[matched_dates] - series_b[matched_dates]).tolist()

        if not fold_month_deltas:
            print(f"Skipping {comparison_name}: no matched (fold, month) IC pairs available.")
            continue

        boot_result = block_bootstrap_ic_delta(fold_month_deltas)
        insufficient = boot_result.get("insufficient_folds", False)
        # Insufficient-fold comparisons (too few distinct folds for a meaningful bootstrap - see
        # block_bootstrap_ic_delta's min_folds) still get a row (with NaN bounds), but their
        # p-value is excluded from the Holm-Bonferroni pass below - holm_bonferroni can't rank a
        # NaN against real p-values, and folding a near-meaningless exact-looking p-value into the
        # correction would distort every other comparison's adjusted p-value too.
        if not insufficient:
            raw_pvalues.append(boot_result["p_value"])
        else:
            print(f"{comparison_name}: only {len(fold_month_deltas)} fold(s) (below min_folds) - "
                  f"bootstrap lower bound/p-value unreliable, excluded from Holm-Bonferroni correction.")
        pairwise_rows.append({
            "comparison": comparison_name,
            "n_folds": len(fold_month_deltas),
            "observed_delta": boot_result["observed_delta"],
            "lower_bound_95": boot_result["lower_bound_95"],
            "raw_p_value": boot_result["p_value"],
            "insufficient_folds": insufficient,
        })

    if pairwise_rows:
        adjusted_pvalues = iter(holm_bonferroni(raw_pvalues))
        for row in pairwise_rows:
            if row["insufficient_folds"]:
                row["holm_adjusted_p_value"] = float('nan')
                row["significant_at_0.05"] = False
            else:
                adj_p = next(adjusted_pvalues)
                row["holm_adjusted_p_value"] = adj_p
                row["significant_at_0.05"] = adj_p < 0.05

        pairwise_df = pd.DataFrame(pairwise_rows)
        print("Pairwise comparison table (IC deltas, one-sided block-bootstrapped within fold, "
              "Holm-Bonferroni corrected across this arm's baseline comparisons):")
        print(pairwise_df.to_string(index=False))

        pairwise_path = f"{config.output_dir}/pairwise_comparisons.csv"
        pairwise_df.to_csv(pairwise_path, index=False)
        print(f"\nWrote {pairwise_path}")
        return pairwise_df
    else:
        print("No pairwise comparisons could be computed.")
        return pd.DataFrame()


def _build_dev_fold_hyperparameter_comparison(results_by_value: dict, output_dir: str, param_label: str,
                                               output_filename: str, min_folds: int = 3) -> pd.DataFrame:
    """
    Shared significance pass behind run_ga.py's dev-fold-only hyperparameter sweeps
    (--gbt-tree-search, --max-features-search): for every pair of values (fewer, more) among the
    sweep, is the GA winner's rank IC measurably better at the larger value? Reuses the exact same
    one-sided, null-centered, within-fold block bootstrap (`block_bootstrap_ic_delta`) that
    build_pairwise_comparisons uses for winner-vs-baseline - here the "comparator" is always the
    larger value and the "baseline" the smaller one, so a low p-value means "the larger value
    beats the smaller one," not just "these differ." Holm-Bonferroni corrected across all pairs
    in the sweep, same as build_pairwise_comparisons does across its own comparisons.

    `results_by_value` is {value: [fold_result, ...]} - one list of DEVELOPMENT-fold result dicts
    per swept value (these sweeps never touch final-test folds), each dict carrying the same
    `winner_monthly_ic` field run_ga_for_fold already computes for every fold regardless of
    category. Matched on (fold_name, month) the same way build_pairwise_comparisons matches
    (fold, month) pairs between two IC series. `param_label` (e.g. "trees", "max_features") only
    feeds the `comparison` column's naming (e.g. "trees_30_vs_trees_10") and the printed table
    header - it doesn't affect the statistics.

    min_folds default is 3, not block_bootstrap_ic_delta's own default of 5 - deliberately
    loosened, since these sweeps only ever produce 3 development folds and 5 is otherwise
    unreachable here. KNOWN WEAKER GUARANTEE, not equivalent to the 5-fold standard used
    elsewhere (build_pairwise_comparisons, compare_ga_runs.py): with only 3 distinct folds to
    draw from, a large share of bootstrap replicates necessarily recombine the same 3 folds'
    own data rather than approximating genuine between-fold variability, so the resulting
    p-value/CI is more "corroborates the observed direction" than a fully-trustworthy
    significance test. Treat accordingly - this is an explicit, documented trade of some rigor
    for getting a number at all out of a 3-fold sweep, not a silent lowering of the bar.
    """
    values = sorted(results_by_value.keys())
    pairwise_rows = []
    raw_pvalues = []
    for i, baseline_value in enumerate(values):
        for comparator_value in values[i + 1:]:
            comparison_name = f"{param_label}_{comparator_value}_vs_{param_label}_{baseline_value}"
            baseline_by_fold = {r["fold_name"]: r for r in results_by_value[baseline_value]}
            comparator_by_fold = {r["fold_name"]: r for r in results_by_value[comparator_value]}

            fold_month_deltas = {}
            for fold_name in sorted(set(baseline_by_fold) & set(comparator_by_fold)):
                ic_baseline = baseline_by_fold[fold_name].get("winner_monthly_ic")
                ic_comparator = comparator_by_fold[fold_name].get("winner_monthly_ic")
                if not ic_baseline or not ic_comparator:
                    continue
                series_baseline, series_comparator = pd.Series(ic_baseline), pd.Series(ic_comparator)
                matched_dates = series_baseline.index.intersection(series_comparator.index).sort_values()
                if len(matched_dates) == 0:
                    continue
                fold_month_deltas[fold_name] = (series_comparator[matched_dates] - series_baseline[matched_dates]).tolist()

            if not fold_month_deltas:
                print(f"Skipping {comparison_name}: no matched (fold, month) IC pairs available.")
                continue

            boot_result = block_bootstrap_ic_delta(fold_month_deltas, min_folds=min_folds)
            insufficient = boot_result.get("insufficient_folds", False)
            if not insufficient:
                raw_pvalues.append(boot_result["p_value"])
            else:
                print(f"{comparison_name}: only {len(fold_month_deltas)} fold(s) (below min_folds) - "
                      f"bootstrap lower bound/p-value unreliable, excluded from Holm-Bonferroni correction.")
            pairwise_rows.append({
                "comparison": comparison_name,
                "baseline_value": baseline_value,
                "comparator_value": comparator_value,
                "n_folds": len(fold_month_deltas),
                "observed_delta": boot_result["observed_delta"],
                "lower_bound_95": boot_result["lower_bound_95"],
                "raw_p_value": boot_result["p_value"],
                "insufficient_folds": insufficient,
            })

    if not pairwise_rows:
        print(f"No {param_label} comparisons could be computed.")
        return pd.DataFrame()

    adjusted_pvalues = iter(holm_bonferroni(raw_pvalues))
    for row in pairwise_rows:
        if row["insufficient_folds"]:
            row["holm_adjusted_p_value"] = float('nan')
            row["significant_at_0.05"] = False
        else:
            adj_p = next(adjusted_pvalues)
            row["holm_adjusted_p_value"] = adj_p
            row["significant_at_0.05"] = adj_p < 0.05

    comparison_df = pd.DataFrame(pairwise_rows)
    print(f"{param_label} search comparison table (dev-fold winner IC deltas, one-sided "
          "block-bootstrapped within fold, Holm-Bonferroni corrected across all pairs):")
    print(comparison_df.to_string(index=False))

    comparison_path = f"{output_dir}/{output_filename}"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nWrote {comparison_path}")
    return comparison_df


def build_gbt_tree_search_comparison(results_by_tree_count: dict, output_dir: str,
                                      min_folds: int = 3) -> pd.DataFrame:
    """run_ga.py --gbt-tree-search's significance pass - see _build_dev_fold_hyperparameter_comparison."""
    return _build_dev_fold_hyperparameter_comparison(
        results_by_tree_count, output_dir, param_label="trees",
        output_filename="gbt_tree_search_comparison.csv", min_folds=min_folds,
    )


def build_max_features_search_comparison(results_by_max_features: dict, output_dir: str,
                                          min_folds: int = 3) -> pd.DataFrame:
    """run_ga.py --max-features-search's significance pass - see _build_dev_fold_hyperparameter_comparison."""
    return _build_dev_fold_hyperparameter_comparison(
        results_by_max_features, output_dir, param_label="max_features",
        output_filename="max_features_search_comparison.csv", min_folds=min_folds,
    )


def build_winner_composition(fold_results: list, config: "GAConfig"):
    """
    Ported from cell 29 (Deliverable 3, item 7). Per final-test fold: fraction of the GA
    winner's leaves that are temporal (_lag1/_delta1/_growth1/_mean3/_std3, classify_leaf) vs
    raw, whether it used ANY temporal leaf, and that fold's own winner_mean_ic/true_test_rmse
    (no "delta vs v2" column - GA_test.ipynb has no separate v2 run to diff against).
    """
    final_test_results = [r for r in fold_results if r["category"] == "final_test"]
    if not final_test_results:
        print("No final-test fold results to build the winner-composition table from yet.")
        return pd.DataFrame()
    composition_rows = [
        {
            "fold_name": r["fold_name"],
            "eval_year": r["eval_year"],
            "winner_leaf_count": r["winner_leaf_count"],
            "winner_temporal_fraction": r["winner_temporal_fraction"],
            "winner_uses_any_temporal": r["winner_uses_any_temporal"],
            "winner_leaf_classification": r["winner_leaf_classification"],
            "winner_mean_ic": r["winner_mean_ic"],
            "winner_true_test_rmse": r["true_test_rmse"],
        }
        for r in final_test_results
    ]
    composition_df = pd.DataFrame(composition_rows).sort_values("eval_year").reset_index(drop=True)

    aggregate_row = {
        "fold_name": "AGGREGATE (mean)",
        "eval_year": None,
        "winner_leaf_count": composition_df["winner_leaf_count"].mean(),
        "winner_temporal_fraction": composition_df["winner_temporal_fraction"].mean(),
        "winner_uses_any_temporal": composition_df["winner_uses_any_temporal"].mean(),
        "winner_leaf_classification": None,
        "winner_mean_ic": composition_df["winner_mean_ic"].mean(),
        "winner_true_test_rmse": composition_df["winner_true_test_rmse"].mean(),
    }
    composition_df = pd.concat([composition_df, pd.DataFrame([aggregate_row])], ignore_index=True)

    print("Winner-composition table:")
    print(composition_df.to_string(index=False))

    composition_path = f"{config.output_dir}/winner_composition.csv"
    composition_df.to_csv(composition_path, index=False)
    print(f"\nWrote {composition_path}")
    return composition_df
