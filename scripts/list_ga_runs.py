"""
Lists every GA run that ever happened - every timestamped directory under ga_runs/ (see
run_ga.py's __main__ block: each invocation gets its own permanent ga_runs/<family>_<stamp>/
directory instead of overwriting a fixed name), read from local disk alone - no Spark, no HDFS,
no YARN, so this runs in well under a second even with hundreds of runs.

Each run directory carries everything needed to describe it:
  run_config.json    the full GAConfig (dataclasses.asdict), written before any fold runs
  run_metadata.json  application_id/run_started_at (written up front) + base_seed/
                      run_finished_at (added once the run completes) + family
  final_test_summary.csv   present only once every final-test fold has been scored - the
                      completion signal this script uses, same as the dashboard's classify()

Usage: python3 scripts/list_ga_runs.py [--root ga_runs] [--family ga_fast] [--json]
"""
import argparse
import csv
import json
import os
import statistics
from datetime import datetime, timezone

DEFAULT_ROOT = "ga_runs"


def _read_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _mean_metrics(summary_path: str):
    """Mean winner_true_test_rmse/winner_mean_ic/winner_ic_ir across every final-test fold row -
    the same three headline numbers final_test_summary.csv's own columns carry per fold."""
    rmse, ic, icir = [], [], []
    try:
        with open(summary_path, newline="") as f:
            for row in csv.DictReader(f):
                for target, key in ((rmse, "winner_true_test_rmse"),
                                     (ic, "winner_mean_ic"), (icir, "winner_ic_ir")):
                    try:
                        target.append(float(row[key]))
                    except (KeyError, ValueError):
                        pass
    except OSError:
        return None
    if not rmse and not ic:
        return None
    return {
        "mean_true_test_rmse": statistics.mean(rmse) if rmse else None,
        "mean_ic": statistics.mean(ic) if ic else None,
        "mean_ic_ir": statistics.mean(icir) if icir else None,
        "n_folds": len(rmse),
    }


def describe_run(run_dir: str) -> dict:
    config = _read_json(os.path.join(run_dir, "run_config.json")) or {}
    metadata = _read_json(os.path.join(run_dir, "run_metadata.json")) or {}
    summary_path = os.path.join(run_dir, "final_test_summary.csv")
    metrics = _mean_metrics(summary_path) if os.path.exists(summary_path) else None

    if metrics is not None:
        status = "complete"
    elif metadata.get("run_finished_at"):
        # every fold ran (run_metadata.json's final write happened) but no final-test folds
        # were discovered/scored - a genuinely empty result, not a crash
        status = "finished, no final-test folds"
    elif metadata.get("run_started_at"):
        status = "running or died (no terminal record - cross-check application_id in YARN)"
    else:
        status = "unknown (no run_metadata.json)"

    return {
        "run_dir": run_dir,
        "family": metadata.get("family") or os.path.basename(run_dir),
        "scale": "fast" if "fast" in (config.get("walk_forward_namespace") or "") else "full",
        "temporal_operators": config.get("enable_temporal_operators"),
        "seed": metadata.get("base_seed", config.get("random_seed")),
        "population_size": config.get("target_population_size"),
        "generations": config.get("generations"),
        "run_baseline_c": config.get("run_baseline_c"),
        "application_id": metadata.get("application_id"),
        "started_at": metadata.get("run_started_at"),
        "finished_at": metadata.get("run_finished_at"),
        "status": status,
        "metrics": metrics,
    }


def discover_runs(root: str, family: str = None) -> list:
    if not os.path.isdir(root):
        return []
    runs = []
    for name in sorted(os.listdir(root)):
        run_dir = os.path.join(root, name)
        if not os.path.isdir(run_dir):
            continue
        info = describe_run(run_dir)
        if family and info["family"] != family:
            continue
        runs.append(info)
    # newest first - started_at is an ISO string, sorts correctly as text; runs with no
    # started_at at all (no run_metadata.json) sort last rather than crashing the comparison
    runs.sort(key=lambda r: r["started_at"] or "", reverse=True)
    return runs


def _fmt(value, width):
    return str(value if value is not None else "-")[:width].ljust(width)


def print_table(runs: list):
    if not runs:
        print("No runs found.")
        return
    header = ("FAMILY", "SCALE", "TEMPORAL", "SEED", "STATUS", "STARTED",
              "RMSE", "IC", "IC-IR", "APP ID")
    widths = (28, 5, 8, 5, 40, 20, 8, 8, 8, 20)
    print("  ".join(_fmt(h, w) for h, w in zip(header, widths)))
    for run in runs:
        metrics = run["metrics"] or {}
        started = (run["started_at"] or "")[:19]
        row = (
            run["family"], run["scale"],
            "on" if run["temporal_operators"] else "off",
            run["seed"], run["status"], started,
            f"{metrics.get('mean_true_test_rmse'):.4f}" if metrics.get("mean_true_test_rmse") is not None else "-",
            f"{metrics.get('mean_ic'):+.4f}" if metrics.get("mean_ic") is not None else "-",
            f"{metrics.get('mean_ic_ir'):+.3f}" if metrics.get("mean_ic_ir") is not None else "-",
            run["application_id"] or "-",
        )
        print("  ".join(_fmt(v, w) for v, w in zip(row, widths)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT,
                         help=f"ga_runs/ root to scan (default: {DEFAULT_ROOT}, relative to cwd "
                              f"- run this from research/, same as run_ga.py itself)")
    parser.add_argument("--family", default=None,
                         help="only list runs of one family, e.g. ga_fast_seed7")
    parser.add_argument("--json", action="store_true", help="machine-readable output instead of a table")
    args = parser.parse_args()

    runs = discover_runs(args.root, args.family)
    if args.json:
        print(json.dumps(runs, indent=2))
    else:
        print_table(runs)
        print(f"\n{len(runs)} run(s) under {args.root}/")
