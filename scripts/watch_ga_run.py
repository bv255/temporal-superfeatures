"""
Live-watches one GA run (research/ga_runs/<family>_<timestamp>/, or its family symlink, e.g.
"ga_fast_seed7") without a web server, browser, or port-forwarding - just a terminal you leave
open. Polls the currently in-progress fold's checkpoint.json on HDFS (written every generation
by GeneticAlgorithm1.run(), see ga/checkpoint.py) and prints its live fitness/diversity as it
moves, plus every fold's status (queued/running/complete).

No Spark session is started - ga.checkpoint reads HDFS directly via pyarrow's libhdfs, and
discover_folds()/GAConfig need no live cluster - so this starts in under a second and is safe to
run alongside the actual run_ga.py job without competing for cluster resources.

Usage (run from research/, same cwd convention as run_ga.py/run_preprocessing.py):
  python3 scripts/watch_ga_run.py ga_fast_seed7            # family symlink, watches the latest
  python3 scripts/watch_ga_run.py ga_runs/ga_fast_seed7_20260818-153000   # a specific run
  python3 scripts/watch_ga_run.py ga_fast --interval 5      # poll every 5s instead of 10
  python3 scripts/watch_ga_run.py ga_fast --once            # one snapshot, no loop

If the run looks stuck or dead, this prints application_id so you can check/kill it by hand:
  yarn application -status <application_id>
  yarn application -kill <application_id>
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault("HADOOP_HOME", "/opt/hadoop")

from superfeatures.config import GAConfig
from superfeatures.ga.algorithms import discover_folds
from superfeatures.ga.checkpoint import read_checkpoint_raw

STAGNATION_WINDOW = 10  # matches engine.py's own convergence-check window (GAConfig.stagnation_ceiling's neighbor)


def _read_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _load_run(run_arg: str):
    if not os.path.isdir(run_arg):
        print(f"error: '{run_arg}' is not a directory (checked relative to cwd - run this from "
              f"research/, and pass either a family name like 'ga_fast' or a full ga_runs/... path)")
        sys.exit(1)
    config_dict = _read_json(os.path.join(run_arg, "run_config.json"))
    if config_dict is None:
        print(f"error: no run_config.json under '{run_arg}' - is this a real run_ga.py output directory?")
        sys.exit(1)
    config = GAConfig(**config_dict)
    folds = discover_folds(config)
    if not folds:
        print(f"warning: discover_folds() found no folds for walk_forward_namespace="
              f"'{config.walk_forward_namespace}' - has run_preprocessing.py been run for this scale?")
    return config, folds


def _fold_output_dir(run_dir: str, fold: dict) -> str:
    return f"{run_dir}/{fold['category']}/{fold['fold_name']}"


def _admitted_rate_trailing_mean(admitted_rate: list, window: int = STAGNATION_WINDOW) -> float:
    recent = [value for value in (admitted_rate or [])[-window:] if value is not None]
    return sum(recent) / len(recent) if recent else 0.0


def _health_flags(checkpoint: dict, config: GAConfig) -> list:
    """
    Mirrors engine.py's OWN convergence signal (admitted_rate trailing average vs
    config.admitted_rate_threshold, plus a fitness plateau) rather than borrowing a generic
    diversity-share heuristic from elsewhere - this is the actual rule GeneticAlgorithm1._should_stop
    uses to decide "converged", so a flag here means the run is close to (or already past) its
    own real stopping condition, not an arbitrary guess at one.
    """
    flags = []
    best = [v for v in (checkpoint.get("best_fitnesses") or []) if v is not None]
    if len(best) >= STAGNATION_WINDOW and len(set(best[-STAGNATION_WINDOW:])) == 1:
        flags.append(f"best fitness unchanged for the last {STAGNATION_WINDOW} generations")

    trailing = _admitted_rate_trailing_mean(checkpoint.get("admitted_rate") or [])
    if trailing >= config.admitted_rate_threshold:
        flags.append(f"admitted_rate trailing average {trailing:.0%} >= threshold "
                     f"{config.admitted_rate_threshold:.0%} (close to this run's own convergence rule)")

    diversity = checkpoint.get("diversity") or []
    if diversity and config.target_population_size:
        share = diversity[-1] / config.target_population_size
        if share <= 0.25:
            flags.append(f"population diversity down to {diversity[-1]}/{config.target_population_size} ({share:.0%})")
    return flags


def _fmt_expr(individual, width: int = 90) -> str:
    text = str(individual) if individual is not None else "-"
    return text if len(text) <= width else text[:width - 3] + "..."


def snapshot(run_dir: str, config: GAConfig, folds: list, run_metadata: dict) -> bool:
    """Prints one status snapshot. Returns True if the whole run is finished (terminal)."""
    now = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"\n=== {now} UTC  —  {run_dir} ===")
    if run_metadata:
        print(f"application_id={run_metadata.get('application_id') or '-'}  "
              f"started={run_metadata.get('run_started_at') or '-'}  "
              f"finished={run_metadata.get('run_finished_at') or '(not yet)'}")
    else:
        print("no run_metadata.json found - this run predates application_id tracking, or hasn't "
              "started writing yet")

    current_fold = None
    for fold in folds:
        fold_dir = _fold_output_dir(run_dir, fold)
        result = _read_json(f"{fold_dir}/fold_result.json")
        if result is not None:
            metric = result.get("winner_true_test_rmse", result.get("validation_rmse"))
            line = f"  [done]    {fold['category']}/{fold['fold_name']}  generations_run={result.get('generations_run')}"
            if metric is not None:
                line += f"  rmse={metric:.5f}"
            print(line)
        elif current_fold is None:
            current_fold = fold
            print(f"  [current] {fold['category']}/{fold['fold_name']}")
        else:
            print(f"  [queued]  {fold['category']}/{fold['fold_name']}")

    if current_fold is not None:
        checkpoint = read_checkpoint_raw(f"{_fold_output_dir(run_dir, current_fold)}/checkpoint.json")
        if checkpoint is None:
            print("            (no checkpoint written yet - still building this fold's panel, "
                  "or generation 1 hasn't completed)")
        else:
            best = checkpoint.get("best_fitnesses") or []
            avg = checkpoint.get("average_fitnesses") or []
            diversity = checkpoint.get("diversity") or []
            print(f"            generation={checkpoint.get('generation')}  "
                  f"best_fitness={best[-1] if best else '-'}  "
                  f"avg_fitness={avg[-1] if avg else '-'}  "
                  f"diversity={diversity[-1] if diversity else '-'}/{config.target_population_size}")
            print(f"            best_individual={_fmt_expr(checkpoint.get('best_individual'))}")
            for flag in _health_flags(checkpoint, config):
                print(f"            HEALTH: {flag}")

    finished = bool(run_metadata and run_metadata.get("run_finished_at"))
    if finished:
        print("\nRun has finished (run_metadata.json has run_finished_at). Final snapshot above.")
    elif current_fold is None:
        print("\nEvery discovered fold has a fold_result.json, but run_metadata.json has no "
              "run_finished_at yet - the summary-building step (final_test_summary.csv etc.) "
              "may still be in progress.")
    return finished or current_fold is None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="a family symlink (e.g. 'ga_fast') or a ga_runs/... path")
    parser.add_argument("--interval", type=float, default=10.0, help="seconds between polls (default 10)")
    parser.add_argument("--once", action="store_true", help="print one snapshot and exit, no loop")
    args = parser.parse_args()

    config, folds = _load_run(args.run)
    try:
        while True:
            run_metadata = _read_json(os.path.join(args.run, "run_metadata.json")) or {}
            done = snapshot(args.run, config, folds, run_metadata)
            if args.once or done:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped watching (the run itself keeps going - this was read-only).")
