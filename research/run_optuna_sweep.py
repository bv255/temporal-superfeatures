"""
Optuna sweep over GAConfig's GA-mechanics and temporal-operator hyperparameters, development
folds only (see run_ga.py's --gbt-tree-search/--max-features-search for the existing one-
dimension-at-a-time sweeps this supersedes for the params below - those stay as-is for anyone
who wants a quick single-axis check, but don't explore interactions).

Swept jointly (the "common GA hyperparameters", shared by both temporal arms - see docs'
Hyperparameter Selection section):
  tournament_size        int   2-8
  mutation_method         cat   flat / increasing / decreasing
  min_mutation            float 0.05-0.3
  max_mutation            float min_mutation-0.6 (constrained >= min_mutation)
  max_features            int   4-7   (max_nesting is set equal to this, not swept independently -
                                        GAConfig's defaults have always had them equal, so treating
                                        that as deliberate rather than adding an 8th free dimension)
  gbt_max_iter            int   15-100
  temporal_wrap_rate      float 0.05-0.4
  temporal_unwrap_rate    float 0.05-(0.4, capped so wrap_rate+unwrap_rate <= 1 - grammar.py's
                                        mutate() draws one random number against their sum)

Held fixed (not swept): crossover_mutation, elitest_mutation, temporal_lag_periods,
temporal_window_sizes (all at whatever GAConfig's current defaults are), fit_backend="local",
num_threads* (run_ga.py's own script defaults). target_population_size/generations are NOT
reduced for the sweep - each trial runs at whichever base config's own scale (100/500 for the
default walk_forward_full, or 15/15 if --fast is passed) - same scale as a real run_ga.py
invocation, not a cut-down calibration scale.

Objective (joint across both temporal arms, per the Hyperparameter Selection writeup): for a
proposed common configuration theta, the SAME theta is evaluated under both the Temporal-ON and
Temporal-OFF arms, using matched development folds and seeds (OPTUNA_SEEDS - 2 seeds, one GA run
per (arm, seed) covering every development fold). J(theta) = 0.5 * (IC_ON(theta) + IC_OFF(theta)),
where IC_ON/IC_OFF are each the mean validation Rank IC across that arm's matched development
runs (both seeds, every development fold, pooled before averaging). Each trial's IC_ON/IC_OFF are
also recorded as trial.user_attrs for later inspection (study.trials_dataframe()), not just their
average. Development folds never touch baseline A/B/C (see run_ga_for_fold's docstring - those are
final-test-only), so nothing needs to be disabled there.

Execution: every (arm, seed) run is dispatched as its own run_ga.py --execution cluster job (see
CLAUDE.md's "Distributing whole GA runs across YARN nodes" section) rather than run in-process -
this sweep used to call run_ga_for_fold directly against a local SparkSession, but now that whole
GA runs can land on dedicated YARN containers, that in-process path leaves the cluster idle and
serializes everything through bialobog's own cores instead. Each trial submits up to 4 cluster
jobs at once (2 arms x 2 seeds); a shared ThreadPoolExecutor caps how many of THOSE jobs (across
every trial, not just one) are ever in flight globally at --max-concurrent-cluster (default 6) -
same deliberate throttle run_ga_sweep.py's own --max-concurrent uses, since this cluster is shared
with other students. To actually reach that cap in practice, --parallel-trials (default 2) runs
that many Optuna trials concurrently, each in its own thread against the shared study/storage -
Optuna's own documented threading recipe (study.optimize() called concurrently from multiple
threads sharing one storage-backed study). A cluster job that fails (non-SUCCEEDED YARN state, or
SUCCEEDED but missing fold_result.json - see cluster_submit.py's docstring on why YARN's own
verdict isn't the authoritative signal) fails just that one trial (study.optimize(catch=(Exception,))
- other trials/workers keep going.

Each (arm, seed) job's own output_dir is pre-decided by this script (not left to run_ga.py's usual
auto-timestamped ga_runs/<family>_<stamp> naming) and passed via --resume, so this script knows
exactly where to `hdfs dfs -get` the finished results back from without having to guess or list
HDFS - ga_runs/optuna_<study_name>_trial<NNNN>_<on|off>_seed<N>, deterministic and collision-free
across trials/arms/seeds for the life of a study.

No pruning: the old single-process version could call trial.should_prune() between folds since it
ran them one at a time in-process. A cluster job runs a trial's whole fold set unattended on YARN
once submitted - there's no cheap way to abort it mid-flight from here - so pruning is dropped
rather than kept as a no-op. A trial's cost is now 4 whole cluster jobs, win or lose.

Storage is a local sqlite file (optuna_sweep.db) so the study survives killing this script
partway through - already-completed trials stay recorded; reload with
`optuna.load_study(study_name=..., storage="sqlite:///optuna_sweep.db")` any time to inspect
study.best_params/study.best_value or resume adding trials via study.optimize(...) again.

Run from research/ (same cwd convention as run_ga_sweep.py/pull_cluster_output.py - relative
paths like "run_ga.py"/"scripts/pull_cluster_output.py" assume it).

Usage: ~/pyspark-venv/bin/python3 run_optuna_sweep.py --n-trials 40
       ~/pyspark-venv/bin/python3 run_optuna_sweep.py --fast --n-trials 40
"""
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
from statistics import mean

import optuna

from superfeatures.config import FULL_GA_CONFIG, FAST_GA_CONFIG
from superfeatures.ga import discover_folds

from run_ga_sweep import _submit, _final_state, POLL_SECONDS


OPTUNA_SEEDS = (11, 12)  # matched development-fold seeds used for hyperparameter selection only -
                          # deliberately distinct from the pre-registered 10-seed (100-109) H1 test
                          # set reserved for the final temporal-on/off comparison.


def _suggest_hyperparams(trial: optuna.Trial) -> dict:
    min_mutation = trial.suggest_float("min_mutation", 0.05, 0.3)
    max_mutation = trial.suggest_float("max_mutation", min_mutation, 0.6)

    temporal_wrap_rate = trial.suggest_float("temporal_wrap_rate", 0.05, 0.4)
    temporal_unwrap_rate = trial.suggest_float(
        "temporal_unwrap_rate", 0.05, min(0.4, 1.0 - temporal_wrap_rate)
    )

    max_features = trial.suggest_int("max_features", 4, 7)

    return {
        "tournament_size": trial.suggest_int("tournament_size", 2, 8),
        "mutation_method": trial.suggest_categorical(
            "mutation_method", ["flat", "increasing", "decreasing"]
        ),
        "min_mutation": min_mutation,
        "max_mutation": max_mutation,
        "max_features": max_features,
        "gbt_max_iter": trial.suggest_int("gbt_max_iter", 15, 100),
        "temporal_wrap_rate": temporal_wrap_rate,
        "temporal_unwrap_rate": temporal_unwrap_rate,
    }


def _run_cluster_job(seed: int, enable_temporal: bool, output_dir: str, params: dict, fast: bool,
                      dev_fold_names: list) -> list:
    """
    Submits one run_ga.py --execution cluster --dev-only job (this (arm, seed)'s matched
    development-fold run for the trial's suggested hyperparameters), polls it to a final YARN
    state, pulls its output back from HDFS, and returns its per-fold validation_rank_ic values.
    Raises on submission failure, a non-SUCCEEDED final state, or a missing/None
    validation_rank_ic - the caller (the trial's objective) lets that fail just this one trial.
    """
    passthrough = []
    if fast:
        passthrough.append("--fast")
    if not enable_temporal:
        passthrough.append("--no-temporal-operators")
    passthrough += ["--dev-only", "--resume", output_dir]
    passthrough += ["--tournament-size", str(params["tournament_size"])]
    passthrough += ["--mutation-method", params["mutation_method"]]
    passthrough += ["--min-mutation", str(params["min_mutation"])]
    passthrough += ["--max-mutation", str(params["max_mutation"])]
    passthrough += ["--max-features", str(params["max_features"])]
    passthrough += ["--gbt-max-iter", str(params["gbt_max_iter"])]
    passthrough += ["--temporal-wrap-rate", str(params["temporal_wrap_rate"])]
    passthrough += ["--temporal-unwrap-rate", str(params["temporal_unwrap_rate"])]

    app_id = _submit(seed, passthrough)
    if app_id is None:
        raise RuntimeError(f"submission failed for seed={seed}, enable_temporal={enable_temporal}, "
                            f"output_dir={output_dir} - see stderr above.")

    state = None
    while state is None:
        time.sleep(POLL_SECONDS)
        state = _final_state(app_id)
    print(f"seed={seed} enable_temporal={enable_temporal} ({app_id}) finished: {state}.")
    if state != "SUCCEEDED":
        raise RuntimeError(f"cluster job {app_id} (seed={seed}, enable_temporal={enable_temporal}, "
                            f"output_dir={output_dir}) finished with YARN state {state}, not "
                            f"SUCCEEDED.")

    pull = subprocess.run([sys.executable, "scripts/pull_cluster_output.py", output_dir],
                           capture_output=True, text=True)
    print(pull.stdout)
    if pull.returncode != 0:
        print(pull.stderr, file=sys.stderr)
        raise RuntimeError(f"failed to pull {output_dir} back from HDFS (exit {pull.returncode}).")

    rank_ics = []
    for fold_name in dev_fold_names:
        result_path = f"{output_dir}/development/{fold_name}/fold_result.json"
        if not os.path.isfile(result_path):
            raise RuntimeError(f"{result_path} missing after a SUCCEEDED cluster job - per "
                                f"cluster_submit.py's own docstring, YARN's Final-State isn't the "
                                f"authoritative signal for a fold's own success.")
        with open(result_path) as f:
            result = json.load(f)
        rank_ic = result.get("validation_rank_ic")
        if rank_ic is None:
            raise RuntimeError(f"{result_path} has no validation_rank_ic - was fitness_metric "
                                f"'rank_ic' actually applied for this run?")
        rank_ics.append(rank_ic)
    return rank_ics


def make_objective(dev_fold_names: list, executor: concurrent.futures.ThreadPoolExecutor,
                    study_name: str, fast: bool, seeds: tuple):
    def objective(trial: optuna.Trial) -> float:
        params = _suggest_hyperparams(trial)

        futures = {}
        for enable_temporal in (True, False):
            for seed in seeds:
                arm = "on" if enable_temporal else "off"
                output_dir = f"ga_runs/optuna_{study_name}_trial{trial.number:04d}_{arm}_seed{seed}"
                futures[(enable_temporal, seed)] = executor.submit(
                    _run_cluster_job, seed, enable_temporal, output_dir, params, fast, dev_fold_names
                )

        results = {key: future.result() for key, future in futures.items()}
        ic_on = mean(ic for seed in seeds for ic in results[(True, seed)])
        ic_off = mean(ic for seed in seeds for ic in results[(False, seed)])
        trial.set_user_attr("ic_on", ic_on)
        trial.set_user_attr("ic_off", ic_off)
        return 0.5 * (ic_on + ic_off)

    return objective


def main(fast: bool, n_trials: int, study_name: str, storage: str, max_concurrent_cluster: int,
         parallel_trials: int, seeds: tuple = OPTUNA_SEEDS) -> None:
    base_config = FAST_GA_CONFIG if fast else FULL_GA_CONFIG
    all_folds = discover_folds(base_config)
    dev_folds = [f for f in all_folds if f["category"] == "development"]
    dev_fold_names = [f["fold_name"] for f in dev_folds]
    print(f"Optuna sweep: {len(dev_folds)} development fold(s), {len(seeds)} seed(s) "
          f"{seeds} x 2 arms (Temporal-ON/Temporal-OFF) per trial "
          f"({len(seeds) * 2} cluster job(s)/trial, {len(dev_folds)} fold(s) each), "
          f"namespace={base_config.walk_forward_namespace}, up to {max_concurrent_cluster} cluster "
          f"job(s) in flight at once across {parallel_trials} parallel trial(s).")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
    )

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_cluster)
    objective = make_objective(dev_fold_names, executor, study_name, fast, seeds)

    trials_per_worker = [n_trials // parallel_trials] * parallel_trials
    for i in range(n_trials % parallel_trials):
        trials_per_worker[i] += 1

    def worker(n: int) -> None:
        if n > 0:
            study.optimize(objective, n_trials=n, catch=(Exception,))

    threads = [threading.Thread(target=worker, args=(n,)) for n in trials_per_worker]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    executor.shutdown(wait=True)

    print(f"\nDone. {len(study.trials)} total trial(s) in study '{study_name}'.")
    print(f"Best value (J = 0.5 * (IC_ON + IC_OFF)): {study.best_value}")
    print(f"Best params: {study.best_params}")
    print(f"Best trial IC_ON/IC_OFF: {study.best_trial.user_attrs.get('ic_on')}/"
          f"{study.best_trial.user_attrs.get('ic_off')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true",
                         help="sweep against walk_forward_fast (the --fast preprocessing output) "
                              "instead of the default walk_forward_full")
    parser.add_argument("--n-trials", type=int, default=40,
                         help="number of Optuna trials to run this invocation (default 40, a "
                              "first-pass calibration budget - see the module docstring). Re-run "
                              "the same command to add more trials to the same study.")
    parser.add_argument("--study-name", default=None,
                         help="Optuna study name (default: derived from --fast so full/fast "
                              "sweeps never collide in the same storage file)")
    parser.add_argument("--storage", default="sqlite:///optuna_sweep.db",
                         help="Optuna storage URL (default: local sqlite file, survives killing "
                              "this script partway through - see module docstring)")
    parser.add_argument("--max-concurrent-cluster", type=int, default=6,
                         help="max number of run_ga.py --execution cluster jobs allowed in flight "
                              "globally at once, across every parallel trial combined - this "
                              "cluster is shared with other students, matching run_ga_sweep.py's "
                              "own --max-concurrent throttle (default 6)")
    parser.add_argument("--parallel-trials", type=int, default=2,
                         help="number of Optuna trials evaluated concurrently, each in its own "
                              "thread against the shared study/storage (Optuna's own documented "
                              "threading recipe for parallel optimization). Each trial submits up "
                              "to (2 x number of --seeds) cluster jobs at once, so the default of "
                              "2 keeps --max-concurrent-cluster's default of 6 slots busy without "
                              "over-submitting when running the default 2 seeds (default 2)")
    parser.add_argument("--seeds", default=None, metavar="N,N,...",
                         help="comma-separated development-fold seed(s) for the joint objective "
                              "(default: OPTUNA_SEEDS = 11,12). Pass a single seed (e.g. --seeds "
                              "11) to halve each trial's cluster-job count (2 jobs/trial instead "
                              "of 4 - 1 arm-pair instead of 2) - useful when cluster capacity is "
                              "tight. Distinct from the pre-registered 10-seed 100-109 final H1 "
                              "evaluation set - these are dev-fold hyperparameter-selection seeds "
                              "only, see OPTUNA_SEEDS's own comment.")
    args = parser.parse_args()

    seeds = tuple(int(s) for s in args.seeds.split(",")) if args.seeds else OPTUNA_SEEDS
    study_name = args.study_name or ("ga_temporal_sweep_fast" if args.fast else "ga_temporal_sweep_full")
    main(fast=args.fast, n_trials=args.n_trials, study_name=study_name, storage=args.storage,
         max_concurrent_cluster=args.max_concurrent_cluster, parallel_trials=args.parallel_trials,
         seeds=seeds)
