"""
Cross-run stability metrics: pure-Python aggregation over N already-written fold_result.json
dicts for the SAME fold, run under different seeds (see ga/algorithms.py's run_ga_for_fold
`seed` param). No Spark session, no orchestration - producing the N runs (a shell loop or small
ad hoc script calling run_ga_for_fold repeatedly with different seeds) is deliberately left to
the caller; this module only computes the comparison once those runs already exist.
"""
from itertools import combinations
from statistics import pstdev


def _jaccard(a: set, b: set) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def compute_stability_metrics(fold_results: list) -> dict:
    """
    fold_results: N already-loaded fold_result.json dicts for the same fold (different seeds).

    - champion_stability: fraction of runs whose winning_expression (already canonicalized in
      fold_result.json - see run_ga_for_fold) matches the modal (most common) winner across the
      N runs.
    - top_k_overlap: mean pairwise Jaccard similarity of each run's canonicalized
      final_population_top_k sets (no pairs, i.e. a single run, trivially overlaps itself: 1.0).
    - true_test_rmse_std/_range, winner_mean_ic_std/_range: spread of the winner's own true
      held-out RMSE / mean IC across the N runs.
    """
    if not fold_results:
        raise ValueError("fold_results must be a non-empty list of fold_result.json dicts.")

    winners = [r["winning_expression"] for r in fold_results]
    modal_winner = max(set(winners), key=winners.count)
    champion_stability = winners.count(modal_winner) / len(winners)

    top_k_sets = [set(r.get("final_population_top_k") or []) for r in fold_results]
    pairs = list(combinations(range(len(top_k_sets)), 2))
    top_k_overlap = (
        sum(_jaccard(top_k_sets[i], top_k_sets[j]) for i, j in pairs) / len(pairs)
        if pairs else 1.0
    )

    rmses = [r["true_test_rmse"] for r in fold_results if r.get("true_test_rmse") is not None]
    ics = [r["winner_mean_ic"] for r in fold_results if r.get("winner_mean_ic") is not None]

    return {
        "n_runs": len(fold_results),
        "modal_winner": modal_winner,
        "champion_stability": champion_stability,
        "top_k_overlap": top_k_overlap,
        "true_test_rmse_std": pstdev(rmses) if len(rmses) >= 2 else float('nan'),
        "true_test_rmse_range": (max(rmses) - min(rmses)) if rmses else float('nan'),
        "winner_mean_ic_std": pstdev(ics) if len(ics) >= 2 else float('nan'),
        "winner_mean_ic_range": (max(ics) - min(ics)) if ics else float('nan'),
    }
