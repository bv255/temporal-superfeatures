"""
Package-backed entry point for the GA search, parameterized by `superfeatures.config.GAConfig`.
Reads `research/run_preprocessing.py`'s output (`walk_forward_full/` or `walk_forward_fast/`,
matching whichever of `--fast` you pass here) - run that first.

Two independent flags:
  --fast                  reduced-scale search (population 15/generations 15 instead of
                           100/500) against the `--fast` preprocessing output
  --no-temporal-operators  disables the GA's live lag/delta/growth/mean/std subtree operators
                           (GAConfig.enable_temporal_operators, see
                           TEMPORAL_SUBTREE_OPERATORS_PROMPT.md) for this run

Output directory is built from both: `ga/`, `ga_no_temporal/`, `ga_fast/`,
`ga_fast_no_temporal/` - so all 4 combinations land in distinct directories instead of
overwriting each other, and can be compared side by side afterward (see CLAUDE.md's "Running
the pipeline" section).

Usage: ~/pyspark-venv/bin/python3 run_ga.py
       ~/pyspark-venv/bin/python3 run_ga.py --no-temporal-operators
       ~/pyspark-venv/bin/python3 run_ga.py --fast
       ~/pyspark-venv/bin/python3 run_ga.py --fast --no-temporal-operators
       ~/pyspark-venv/bin/python3 run_ga.py --fit-backend local --seed 10
       ~/pyspark-venv/bin/python3 run_ga.py --seed 10 --execution cluster
                                # ^ same run, but placed on a dedicated YARN container instead
                                #   of executing here - see --execution's own help text below

If a run dies partway through (SSH drop, OOM, etc.) - or you just want to add the final-test
folds you hadn't gotten to yet - rerun the SAME command with the SAME flags:
       ~/pyspark-venv/bin/python3 run_ga.py --seed 7
`_find_matching_run` (below) scans ga_runs/ for an existing run whose run_config.json is
IDENTICAL to this invocation's config (everything except output_dir itself, the one field
that's allowed to differ) and, if found, reuses that exact directory instead of minting a fresh
one. Once reused: any fold whose fold_result.json already carries a fingerprint matching this
config + fold data is skipped outright (no Spark work at all); any fold that has a checkpoint
but no fold_result.json yet resumes from its last saved generation; anything with neither runs
fresh. Change ANYTHING in the config (mutation rate, population size, whatever - not just the
CLI flags) and it's treated as a different run - no match, a brand-new directory, no
interference with the old one. The one thing this can't distinguish: running the EXACT same
config a second time on purpose (e.g. to check Spark's own run-to-run variance) looks identical
to "resume the crashed one" - it will always merge into the most recent matching run rather than
starting an independent second attempt.
Use --resume to manually point at a specific existing ga_runs/ directory instead of relying on
auto-discovery - mainly useful for a run whose run_config.json predates this fingerprint-based
matching (see --resume's own help text), or to disambiguate when you don't want the most recent
match auto-discovery would pick.
"""
import argparse
import dataclasses
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from pyspark.sql import SparkSession

from superfeatures.config import GAConfig, FULL_GA_CONFIG, FAST_GA_CONFIG
from superfeatures.ga import (
    build_final_test_summary,
    build_gbt_tree_search_comparison,
    build_max_features_search_comparison,
    build_pairwise_comparisons,
    build_winner_composition,
    discover_folds,
    run_ga_for_fold,
)
from superfeatures.ga.checkpoint import compute_fingerprint


def _build_spark_session(driver_memory: str = "8g", master: str = "yarn") -> SparkSession:
    """GA_test.ipynb cell 0's SparkSession config - a separate appName/process from
    PreProcessing's own session (run_preprocessing.py's _build_spark_session), since the two
    notebooks were always run as independent kernels.

    Always master="yarn" now (see SUPERFEATURES_CLUSTER_CHILD-gated sizing below) - the
    `--execution cluster` path (cluster_submit.py) used to run this SparkSession as
    master="local[*]" once the whole process had already been placed on a dedicated YARN
    container via `spark-submit --deploy-mode cluster`, on the theory that a container that's
    already placed has no reason to request separate executor containers. **That theory was
    wrong in a way that silently killed every full-scale cluster run**: `local[*]` never
    instantiates Spark's YarnClusterSchedulerBackend, so the SparkContext it creates never fires
    the sparkContextInitialized callback ApplicationMaster.runDriver() is waiting on - and that
    callback firing is what makes the ApplicationMaster actually call registerApplicationMaster()
    against the ResourceManager. Setting spark.yarn.am.waitTime very high (cluster_submit.py used
    to do this) only stops the ApplicationMaster's own internal self-timeout from firing while it
    waits for that callback - it does NOT make the callback fire, so the AM never registers at
    all. Confirmed directly (2026-08-24): 4 real full-scale seeds (application_1787288171816_0079-
    0082) all died with "ApplicationMaster ... timed out" at 645-777s in, matching
    yarn.am.liveness-monitor.expiry-interval-ms=600000 on this cluster (`yarn-site.xml`) almost
    exactly - the RM's OWN independent liveness monitor, unrelated to spark.yarn.am.waitTime, kills
    any AM that never registers, regardless of whether real work is happening inside it (one of the
    4, seed 103, had already completed fold_01 with a real true-test RMSE printed when it was
    killed - that work was never persisted, since the HDFS push only happens in run_ga.py's
    `finally` block on a *process* that's still alive to reach it). This never showed up during
    the earlier `--fast --dev-only` validation because that whole run finished in ~2-3 minutes,
    comfortably under the 10-minute expiry window - nothing to do with correctness, just luck of
    not running long enough to hit it. Fix: request one real (tiny) executor under
    SUPERFEATURES_CLUSTER_CHILD too, so Spark's normal YarnClusterSchedulerBackend registers the
    AM promptly and keeps heartbeating for the run's whole lifetime, exactly like every other
    Spark-on-YARN cluster-mode job already does - proven, no hand-rolled AMRMClient/heartbeat code
    needed. GAPreprocessing's per-fold HDFS reads and everything else already work identically
    against master="yarn" (that's what --execution local has always used), so this only changes
    executor *count*, not any application code path.

    The notebook could get away without setting these env vars / an explicit .master("yarn")
    because it inherited them from whatever shell launched Jupyter Lab. A standalone script
    invocation has no such inherited shell state, so - same as run_preprocessing.py's own
    _build_spark_session - they're set explicitly here. Missing this is a real bug the port
    surfaced: without HADOOP_CONF_DIR, Spark never loads core-site.xml, so fs.defaultFS silently
    falls back to the local filesystem instead of hdfs://bialobog:8020, and every relative-path
    read of a fold's train/eval CSV (written to HDFS by run_preprocessing.py) fails with
    PATH_NOT_FOUND against a local `file:/...` path instead - on the very first fold touched,
    not something specific to any one fold.

    spark.executor.instances lowered from 20 to 8 (bialobog/`--execution local` sizing only, see
    the SUPERFEATURES_CLUSTER_CHILD branch below for the cluster-child's own, much smaller
    sizing) - real Spark UI stage metrics pulled from a live run at the original 20-executor
    sizing showed zero memoryBytesSpilled/diskBytesSpilled, only ~25% of executorRunTime spent as
    actual executorCpuTime, and ~83% of executorRunTime spent in executorDeserializeTime - the
    signature of many small per-individual GBT-fit tasks bottlenecked on fixed per-task
    scheduling/deserialization overhead, not on executor count, cores, or memory. 20 executors
    weren't buying meaningfully more throughput per run on this shared cluster; 8 roughly halves
    one run's YARN footprint (~21 -> ~10 vCores) so more full-scale runs (e.g. a multi-seed sweep)
    can execute concurrently instead of queuing for the same wide allocation one run at a time.
    Not rigorously re-benchmarked end-to-end at 8 - if a future run's own Spark UI stage metrics
    ever show real spill or high executorCpuTime%, that's the signal this was too aggressive for
    that workload shape and should be raised back up, not a hardcoded permanent value.
    """
    is_cluster_child = os.environ.get("SUPERFEATURES_CLUSTER_CHILD") == "1"

    os.environ["SPARK_HOME"] = "/opt/spark"
    os.environ["HADOOP_HOME"] = "/opt/hadoop"
    os.environ["HADOOP_CONF_DIR"] = "/opt/hadoop-3.4.1/etc/hadoop"
    os.environ["YARN_CONF_DIR"] = "/opt/hadoop-3.4.1/etc/hadoop"
    # Hardcoding bialobog's own venv path only makes sense for a true bialobog-hosted run. A
    # cluster-child (is_cluster_child) is itself already running as whatever interpreter
    # spark-submit's --archives/PYSPARK_PYTHON launched it as (the shipped venv archive, unpacked
    # fresh per-container) - overwriting PYSPARK_PYTHON here with a bialobog-only path would break
    # that (and now also break its executor, which needs the identical treatment - see
    # pyspark_python below), so leave it alone in that case.
    if master == "yarn" and not is_cluster_child:
        os.environ["PYSPARK_PYTHON"] = "/home/bvail/pyspark-venv/bin/python3"
        os.environ["PYSPARK_DRIVER_PYTHON"] = "/home/bvail/pyspark-venv/bin/python3"

    print(f"Starting SparkSession (master={master}; this can take a moment on YARN)...")
    # sys.executable is an absolute path into *this* container's own NM local-dir (e.g.
    # .../appcache/application_X/container_X_01_000001/environment/bin/python3) - correct for
    # this process itself, but meaningless if handed to a *different* executor container, whose
    # own copy of the same shipped archive unpacks under its own different absolute local-dir
    # path. The portable spelling every container agrees on is the relative one YARN's
    # `--archives ...#environment` naming already guarantees ("./environment/bin/python3" from
    # each container's own cwd, driver or executor alike) - use that instead of sys.executable
    # whenever this config might reach an executor (is_cluster_child, now that it requests a real
    # one - see below). Not a concern for the bialobog case: there sys.executable is already the
    # one true bialobog venv path every executor needs too.
    pyspark_python = "./environment/bin/python3" if is_cluster_child else sys.executable
    builder = (
        SparkSession.builder
        .master(master)
        .appName("GeneticAlgorithm_2025")
        .config("spark.driver.memory", driver_memory)
        .config("spark.eventLog.enabled", "false")
        .config("spark.pyspark.python", pyspark_python)
        .config("spark.pyspark.driver.python", pyspark_python)
    )
    if master == "yarn":
        if is_cluster_child:
            # Deliberately tiny - this executor's only job is to give the ApplicationMaster a
            # real YARN-registered SparkContext to report (see this function's docstring), not to
            # do meaningful work. Under fit_backend="local" the actual per-individual GA compute
            # never touches Spark tasks at all (pandas/XGBoost in-process on the driver's own
            # ThreadPoolExecutor - see caching.LocalDataCache/operators/*_local.py); only each
            # fold's one-time HDFS read + gen-0 toPandas cache-fill run as real (tiny, fast) Spark
            # jobs, and those are exactly as correct on 1 executor as on 8.
            builder = (
                builder
                .config("spark.executor.instances", "1")
                .config("spark.executor.cores", "1")
                .config("spark.executor.memory", "1g")
                .config("spark.executor.memoryOverhead", "300m")
            )
        else:
            builder = (
                builder
                .config("spark.executor.instances", "8")
                .config("spark.executor.cores", "1")
                .config("spark.executor.memory", "1g")
                .config("spark.executor.memoryOverhead", "300m")
            )
    spark = (
        builder
        # Workaround for a cluster-capacity issue, not a pipeline bug: this driver runs ON
        # bialobog, which is ALSO an HDFS datanode - HDFS's default block placement policy
        # always tries the writer's own local node first, so every block write tries
        # 192.168.2.1 (bialobog) first, guaranteed, and it's ~99.8% full. No rack topology is
        # configured (single flat rack), so the other 2 of 3 replicas are drawn close to
        # randomly from the remaining nodes, 3 of which (skrzat3/4/5) are also nearly full -
        # enough to exhaust the default dfs.client.block.write.retries=3 and abort the whole
        # SparkContext before a single fold even runs. replication=1 means a block only needs
        # ONE good node instead of 3 simultaneously good ones; more retries gives extra chances
        # to skip past the bad picks. Fine for this pipeline's output (fully regenerable,
        # doesn't need 3x durability) but does trade away HDFS's normal fault tolerance for it.
        .config("spark.hadoop.dfs.client.block.write.retries", "10")
        .config("spark.hadoop.dfs.replication", "1")
        .getOrCreate()
    )
    print("SparkSession ready.")
    return spark


def _update_latest_link(link_path: str, target_path: str) -> None:
    """
    Point `link_path` (e.g. "ga_fast") at `target_path` (e.g. "ga_runs/ga_fast_20260818-153000")
    as a relative symlink, so every doc/script that already reads the plain family name
    (CLAUDE.md's "Running the pipeline", compare_ga_runs.py, this file's own docstring) keeps
    working unchanged and transparently sees the most recent run of that family - `open()`,
    `os.path.isdir`, pandas, and a file browser all follow a symlink without knowing it's one.

    Never touches `link_path` if something real is already there: `ga/`, `ga_fast/`, etc.
    predate this scheme and are real directories holding real prior run output on this host, not
    symlinks - overwriting one to make room for a link would delete that data. In that case this
    prints a warning and leaves it alone; the new run's own output still lands correctly under
    `target_path`, only the family-name convenience link is skipped until the old directory is
    manually moved out of the way.
    """
    if os.path.islink(link_path):
        os.unlink(link_path)
    elif os.path.exists(link_path):
        print(f"WARNING: '{link_path}' already exists and is a real directory (not a symlink "
              f"this script manages) - leaving it untouched rather than deleting it. This run's "
              f"output is still at '{target_path}'; '{link_path}' will not point to it until "
              f"the existing directory is moved aside.")
        return
    os.symlink(os.path.relpath(target_path, os.path.dirname(link_path) or "."), link_path)
    print(f"'{link_path}' -> '{target_path}'")


def _find_matching_run(config: GAConfig, ga_runs_dir: str = "ga_runs") -> str:
    """
    Scans ga_runs_dir for an existing run whose run_config.json is identical to `config` -
    same comparison ga/checkpoint.py's compute_fingerprint uses per-fold: everything except
    output_dir (the one field two runs of an otherwise-identical config are expected to differ
    on - it's "where," not "what"). Returns the most-recently-modified match's directory path,
    or None if nothing matches.

    This is what makes "just rerun the same command" automatically continue/skip-complete an
    interrupted run of the identical config, instead of always minting a brand-new directory
    that starts every fold over. The accepted tradeoff (see run_ga.py's module docstring):
    running the exact same config a second time on purpose isn't distinguishable from "resume
    the crashed one" - it always merges into the most recent match.
    """
    if not os.path.isdir(ga_runs_dir):
        return None
    current = dataclasses.asdict(config)
    current.pop("output_dir", None)
    candidates = []
    for name in os.listdir(ga_runs_dir):
        run_dir = os.path.join(ga_runs_dir, name)
        run_config_path = os.path.join(run_dir, "run_config.json")
        if not os.path.isfile(run_config_path):
            continue
        try:
            with open(run_config_path) as f:
                stored = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        stored.pop("output_dir", None)
        # Tolerate a run_config.json written during the brief window checkpoint_namespace
        # existed as a GAConfig field (since removed - see config.py) - it's absent from
        # `current` (a fresh asdict), so leaving it in `stored` would make every comparison
        # against such a file fail even when everything else matches.
        stored.pop("checkpoint_namespace", None)
        if stored == current:
            candidates.append(run_dir)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _load_or_run_fold(fold: dict, config: GAConfig, spark):
    """
    Skips a fold entirely (no Spark work at all, not even the generation-0 baseline pass) if its
    fold_result.json already exists AND carries a fingerprint matching this exact config + fold
    data (same compute_fingerprint the per-generation HDFS checkpoint already uses - see
    algorithms.py's "fingerprint" key in its result dict). Any mismatch - a real config/data
    change, or a legacy fold_result.json with no "fingerprint" key at all - always falls through
    to a real run_ga_for_fold call rather than trusting stale output, which is what makes this
    safe to do unconditionally instead of needing an opt-in flag.
    """
    fold_output_dir = f"{config.output_dir}/{fold['category']}/{fold['fold_name']}"
    result_path = f"{fold_output_dir}/fold_result.json"
    if os.path.isfile(result_path):
        with open(result_path) as f:
            existing_result = json.load(f)
        if existing_result.get("fingerprint") == compute_fingerprint(fold, config):
            print(f"Fold {fold['fold_name']} ({fold['category']}): already complete with a "
                  f"matching config/data fingerprint - skipping (loaded from {result_path}).")
            return existing_result
    return run_ga_for_fold(fold, config, spark)


def _print_baseline_c_preflight(folds: list, config: GAConfig) -> None:
    """Cell 24: flag baseline C's estimated added cost BEFORE the fold loop runs (a fit count,
    not a wall-clock estimate - see the notebook cell's own docstring for why)."""
    final_test_count = len([f for f in folds if f["category"] == "final_test"])
    if config.run_baseline_c:
        max_added_fits_per_fold = (config.generations - 1) * config.target_population_size
        print(f"config.run_baseline_c=True: baseline C will run for each of the {final_test_count} "
              f"final-test folds discovered, adding up to ~{max_added_fits_per_fold} extra "
              f"composite-individual GBT fits per fold (fewer if a fold's real run terminates early, "
              f"or its evolvable-feature count caps population_size below {config.target_population_size}) - "
              f"up to ~{max_added_fits_per_fold * final_test_count} extra fits total across all "
              f"final-test folds. This reuses the real run's warmed caches, so it is NOT a second "
              f"full GA run's worth of compute. Set --no-baseline-c to skip it entirely.")
    else:
        print("config.run_baseline_c=False: baseline C will be skipped for every fold.")


GBT_TREE_SEARCH_COUNTS = (10, 30, 50)


def run_gbt_tree_search(base_config: GAConfig, tree_counts: tuple = GBT_TREE_SEARCH_COUNTS) -> None:
    """
    --gbt-tree-search: a hardcoded hyperparameter sweep over GAConfig.gbt_max_iter (GBTRegressor's
    number of boosting trees), development folds only - this sweep is a search over the fitness
    function's own model complexity, not a real GA search result, so it never touches final-test
    folds (final-test folds are only ever evaluated once, for the headline result). For each
    candidate tree count, runs every development fold through the normal run_ga_for_fold path via
    a fresh GAConfig (gbt_max_iter overridden, its own output_dir under
    '{base_config.output_dir}_gbt_search/trees_N/') - so each tree count's runs/checkpoints get
    the same per-fold fingerprinted skip/resume behavior a normal run_ga.py invocation gets:
    rerunning this command resumes any tree count's incomplete dev folds rather than restarting
    the whole sweep. Once every tree count has run every dev fold, build_gbt_tree_search_comparison
    compares each pair's winner rank-IC via the same one-sided block-bootstrap/Holm-Bonferroni
    machinery build_pairwise_comparisons uses for winner-vs-baseline comparisons elsewhere in this
    pipeline - see that function's docstring for why "more trees" is always the comparator arm.
    """
    spark = _build_spark_session()
    search_output_dir = f"{base_config.output_dir}_gbt_search"
    try:
        all_folds = discover_folds(base_config)
        dev_folds = [f for f in all_folds if f["category"] == "development"]
        print(f"--gbt-tree-search: sweeping gbt_max_iter over {tree_counts}, {len(dev_folds)} "
              f"development fold(s) each, under '{search_output_dir}/'.")

        results_by_tree_count = {}
        for n_trees in tree_counts:
            tree_config = dataclasses.replace(
                base_config, gbt_max_iter=n_trees, output_dir=f"{search_output_dir}/trees_{n_trees}",
            )
            os.makedirs(tree_config.output_dir, exist_ok=True)
            with open(f"{tree_config.output_dir}/run_config.json", "w") as f:
                json.dump(dataclasses.asdict(tree_config), f, indent=2)

            print(f"\n########## gbt_max_iter={n_trees} ##########")
            results_by_tree_count[n_trees] = [
                _load_or_run_fold(fold, tree_config, spark) for fold in dev_folds
            ]

        build_gbt_tree_search_comparison(results_by_tree_count, search_output_dir)
    finally:
        print("Stopping SparkSession...")
        spark.stop()
        print("SparkSession stopped.")


MAX_FEATURES_SEARCH_VALUES = (3, 5, 8)


def run_max_features_search(base_config: GAConfig, max_features_values: tuple = MAX_FEATURES_SEARCH_VALUES) -> None:
    """
    --max-features-search: a hardcoded hyperparameter sweep over GAConfig.max_features (the cap
    on how many leaf features a single evolved expression/individual can combine via crossover -
    ExpressionGrammar's "maximum individual size"), development folds only - same shape as
    run_gbt_tree_search above, just a different config field and search space. Never touches
    final-test folds, for the same reason: this is a search over the GA's own representation
    capacity, not a real GA search result. For each candidate value, runs every development fold
    through the normal run_ga_for_fold path via a fresh GAConfig (max_features overridden, its own
    output_dir under '{base_config.output_dir}_max_features_search/max_features_N/') - so each
    value's runs/checkpoints get the same per-fold fingerprinted skip/resume behavior a normal
    run_ga.py invocation gets. Once every value has run every dev fold,
    build_max_features_search_comparison compares each pair's winner rank-IC via the same
    one-sided block-bootstrap/Holm-Bonferroni machinery run_gbt_tree_search uses.

    Unlike gbt_max_iter, max_features doesn't change each individual's own fit cost directly -
    initialize_population() still starts every individual as a single raw feature regardless of
    this setting - it changes the ceiling crossover can grow the population toward over
    generations, so runtime impact is less predictable up front than the tree-count sweep's.
    """
    spark = _build_spark_session()
    search_output_dir = f"{base_config.output_dir}_max_features_search"
    try:
        all_folds = discover_folds(base_config)
        dev_folds = [f for f in all_folds if f["category"] == "development"]
        print(f"--max-features-search: sweeping max_features over {max_features_values}, "
              f"{len(dev_folds)} development fold(s) each, under '{search_output_dir}/'.")

        results_by_max_features = {}
        for n_features in max_features_values:
            features_config = dataclasses.replace(
                base_config, max_features=n_features, output_dir=f"{search_output_dir}/max_features_{n_features}",
            )
            os.makedirs(features_config.output_dir, exist_ok=True)
            with open(f"{features_config.output_dir}/run_config.json", "w") as f:
                json.dump(dataclasses.asdict(features_config), f, indent=2)

            print(f"\n########## max_features={n_features} ##########")
            results_by_max_features[n_features] = [
                _load_or_run_fold(fold, features_config, spark) for fold in dev_folds
            ]

        build_max_features_search_comparison(results_by_max_features, search_output_dir)
    finally:
        print("Stopping SparkSession...")
        spark.stop()
        print("SparkSession stopped.")


def main(config: GAConfig, latest_link: str = None, skip_dev_folds: bool = False,
         skip_final_test_folds: bool = False, driver_memory: str = "8g", master: str = "yarn"):
    spark = _build_spark_session(driver_memory=driver_memory, master=master)
    run_started_at = datetime.now(timezone.utc).isoformat()
    try:
        os.makedirs(config.output_dir, exist_ok=True)
        if latest_link is not None:
            _update_latest_link(latest_link, config.output_dir)

        # So compare_ga_runs.py can confirm grammar-limit fields (e.g. max_features) actually
        # match between a temporal-on/temporal-off pair instead of assuming it from the fact
        # they share one config.py - two runs of this script are two separate GAConfig
        # instances, and only --no-temporal-operators/--fast are guaranteed to have been varied
        # deliberately between them.
        with open(f"{config.output_dir}/run_config.json", "w") as f:
            json.dump(dataclasses.asdict(config), f, indent=2)

        # Written immediately (not just at the end, like run_metadata.json used to be) so a live
        # viewer polling this directory can tell "running" from "died" and show elapsed time
        # while folds are still in progress, instead of only learning application_id/start time
        # once every fold is already done. Overwritten again below once the run actually
        # finishes, adding run_finished_at - same file, two writes. `family` is the
        # un-timestamped name (e.g. "ga_fast_seed7") every run of this exact configuration
        # shares - scripts/list_ga_runs.py groups on it rather than re-parsing the timestamp out
        # of output_dir's own name.
        run_metadata = {
            "application_id": spark.sparkContext.applicationId,
            "run_started_at": run_started_at,
            "family": latest_link,
        }
        with open(f"{config.output_dir}/run_metadata.json", "w") as f:
            json.dump(run_metadata, f, indent=2)

        all_folds = discover_folds(config)
        dev_folds = [f for f in all_folds if f["category"] == "development"]
        final_test_folds = [f for f in all_folds if f["category"] == "final_test"]
        print(f"Discovered {len(dev_folds)} development folds and {len(final_test_folds)} final-test folds.")
        _print_baseline_c_preflight(all_folds, config)

        fold_results = []
        if skip_dev_folds:
            print(f"\n########## Development folds (skipped by default - pass --with-dev-folds "
                  f"to include them, {len(dev_folds)} fold(s) not run) ##########")
        else:
            print("\n########## Development folds ##########")
            for fold in dev_folds:
                fold_results.append(_load_or_run_fold(fold, config, spark))

        if skip_final_test_folds:
            print(f"\n########## Final-test folds (skipped - --dev-only was passed, "
                  f"{len(final_test_folds)} fold(s) not run) ##########")
        else:
            print("\n########## Final-test folds ##########")
            for fold in final_test_folds:
                fold_results.append(_load_or_run_fold(fold, config, spark))

        print("\nAll folds done.")

        if skip_final_test_folds:
            print("Skipping summary builders (final-test-scoped - see load_final_test_fold_"
                  "results docstrings - and --dev-only ran none).")
        else:
            build_final_test_summary(fold_results, config)
            build_pairwise_comparisons(fold_results, config)
            build_winner_composition(fold_results, config)

        # Run-level provenance - updates the same run_metadata.json written up front above,
        # adding run_finished_at/base_seed now that the whole multi-fold run has actually
        # completed (application_id/run_started_at were already written before the fold loop
        # started, for a live viewer's sake - this isn't the first write to this path). base_seed
        # is config.random_seed since this script never passes an explicit per-call seed
        # override to run_ga_for_fold.
        run_metadata["base_seed"] = config.random_seed
        run_metadata["run_finished_at"] = datetime.now(timezone.utc).isoformat()
        with open(f"{config.output_dir}/run_metadata.json", "w") as f:
            json.dump(run_metadata, f, indent=2)
    finally:
        if os.environ.get("SUPERFEATURES_CLUSTER_CHILD") == "1":
            # config.output_dir is a relative local-disk path, written inside this YARN
            # container's own ephemeral NodeManager scratch space (not
            # ~/temporal-superfeatures/research/ - that only exists on bialobog, see CLAUDE.md's
            # "/home/bvail is NOT shared with the YARN executors") - it's deleted once this
            # container exits. Push whatever's been written so far (even on a partial/crashed
            # run - already-completed folds' fold_result.json shouldn't be lost) to HDFS, same
            # root ga/checkpoint.py's _HDFS_CHECKPOINT_ROOT already uses for per-generation
            # checkpoints, so pulling this path back down also picks up any checkpoints that
            # happen to still be there. research/scripts/pull_cluster_output.py does the pull.
            # Full path, not bare "hdfs" - not on PATH inside the container (confirmed the hard
            # way: bare "hdfs" raised FileNotFoundError here, which - since this is a finally
            # block - clobbered whatever real exception the try block was already handling and
            # skipped spark.stop() below entirely). Wrapped in try/except for the same reason: a
            # push failure must never prevent cleanup or mask the run's real outcome.
            #
            # Pushed file-by-file, at each file's exact relative path, rather than `-put -f
            # config.output_dir <hdfs_dir>` as one call: ga/checkpoint.py's per-generation
            # checkpoint writes already create e.g. .../development/fold_01/ on HDFS *during* the
            # run, so by the time this runs the destination directory already exists - `-put`
            # then treats it as "copy INTO the existing dir" (cp semantics) and nests the whole
            # output_dir one level too deep (confirmed the hard way:
            # .../ga_fast_seed999_.../ga_fast_seed999_.../development/... instead of
            # .../ga_fast_seed999_.../development/...). Pushing each file at its own exact
            # destination path sidesteps that ambiguity entirely - `-put -f` unambiguously means
            # "this file goes exactly here" regardless of what already exists alongside it.
            print(f"Pushing {config.output_dir} to HDFS (cluster child)...")
            hdfs_bin = os.path.join(os.environ.get("HADOOP_HOME", "/opt/hadoop"), "bin", "hdfs")
            try:
                failures = []
                for dirpath, _, filenames in os.walk(config.output_dir):
                    for name in filenames:
                        local_file = os.path.join(dirpath, name)
                        hdfs_file = f"/user/bvail/ga-runs/{local_file}"
                        push_result = subprocess.run(
                            [hdfs_bin, "dfs", "-put", "-f", local_file, hdfs_file],
                            capture_output=True, text=True,
                        )
                        if push_result.returncode != 0:
                            failures.append((local_file, push_result.stderr))
                if failures:
                    print(f"WARNING: HDFS push failed for {len(failures)} file(s) - some results "
                          f"are only in this now-ephemeral container's local disk and will be "
                          f"lost: {failures}")
                else:
                    print(f"Pushed to hdfs:///user/bvail/ga-runs/{config.output_dir} - pull down with "
                          f"research/scripts/pull_cluster_output.py {config.output_dir}"
                          + (f" --family {latest_link}" if latest_link else ""))
            except OSError as e:
                print(f"WARNING: HDFS push errored ({e}) - results are only in this now-ephemeral "
                      f"container's local disk and will be lost.")
        print("Stopping SparkSession...")
        spark.stop()
        print("SparkSession stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true",
                         help="reduced-scale search (population 15/generations 15) against the "
                              "--fast preprocessing output - writes to ga_fast/ instead of ga/")
    parser.add_argument("--no-baseline-c", action="store_true",
                         help="skip baseline C (matched-budget random search) for every fold")
    parser.add_argument("--fitness-metric", choices=["rmse", "rank_ic"], default=None,
                         help="what GeneticAlgorithm1.evaluate_fitness_static scores each "
                              "individual on during the search (GAConfig.fitness_metric). "
                              "'rank_ic' (this script's default as of 2026-08-24 - see CLAUDE.md's "
                              "--fitness-metric note; GAConfig's own dataclass default stays "
                              "'rmse' for other callers) - fitness is mean monthly Spearman rank "
                              "IC over the validation period. 'rmse' - fitness is -RMSE on the "
                              "fold's validation rows instead (the original, pre-2026-08-24 "
                              "behavior). Either way the GBTRegressor is still fit to predict "
                              "next-month returns - only the GA's own selection signal changes, "
                              "not what the model is trained on. No output_dir suffix either way.")
    parser.add_argument("--fit-backend", choices=["spark", "local"], default=None,
                         help="which implementation GeneticAlgorithm1 uses to fit/score each "
                              "individual's GBT model during the search (GAConfig.fit_backend). "
                              "'local' (this script's default - GAConfig's own dataclass default "
                              "stays 'spark' for other callers) - xgboost.XGBRegressor per "
                              "individual, in-process pandas/numpy, no per-individual Spark "
                              "scheduling overhead (added after profiling showed that overhead "
                              "dominates wall-clock time for data this small) and releases the "
                              "GIL during fitting so the existing ThreadPoolExecutor gets real "
                              "multi-core parallelism. 'spark' - pyspark.ml.regression.GBTRegressor "
                              "per individual, on the cluster instead (the original, much slower "
                              "for this workload - see CLAUDE.md). NOT numerically reproducible "
                              "against each other even at matched seeds - separate methodology "
                              "arms. No output_dir suffix either way.")
    parser.add_argument("--no-temporal-operators", action="store_true",
                         help="disable the GA's live lag/delta/growth/mean/std subtree operators "
                              "for this run, and suffix output_dir with '_no_temporal' so results "
                              "don't collide with a temporal-enabled run")
    parser.add_argument("--seed", type=int, default=None,
                         help="override GAConfig.random_seed (the base seed every fold's own "
                              "seed is derived from - see GAConfig.random_seed's docstring). "
                              "Suffixes output_dir with '_seedN' so a seed-sweep's runs land in "
                              "distinct directories instead of overwriting each other - the "
                              "default seed (42, unpassed) keeps the plain, unsuffixed name.")
    parser.add_argument("--resume", metavar="OUTPUT_DIR", default=None,
                         help="manually target a specific existing ga_runs/ directory instead of "
                              "letting _find_matching_run auto-discover one - see the module "
                              "docstring above for how auto-discovery/skip/resume works by "
                              "default (no flag needed) for any run whose run_config.json this "
                              "script itself wrote. Mainly useful for: a run whose "
                              "run_config.json predates the 'fingerprint' key in fold_result.json "
                              "(older runs still auto-discover by directory/config match, just "
                              "without per-fold fingerprinted skipping) or disambiguating when "
                              "you don't want the most-recently-modified match auto-discovery "
                              "would pick.")
    parser.add_argument("--gbt-tree-search", action="store_true",
                         help="run a hardcoded hyperparameter sweep over GAConfig.gbt_max_iter "
                              f"({GBT_TREE_SEARCH_COUNTS}) against DEVELOPMENT folds only, then "
                              "compare each pair's winner rank-IC via the same one-sided block-"
                              "bootstrap/Holm-Bonferroni significance test used for winner-vs-"
                              "baseline comparisons. Writes to "
                              "'{output_dir}_gbt_search/trees_N/' plus a top-level "
                              "gbt_tree_search_comparison.csv - composes with --fast/"
                              "--no-temporal-operators/--seed (which set the base output_dir), "
                              "but not with --resume (the sweep manages its own per-tree-count "
                              "resumability the same way a normal run does).")
    parser.add_argument("--max-features-search", action="store_true",
                         help="run a hardcoded hyperparameter sweep over GAConfig.max_features "
                              f"({MAX_FEATURES_SEARCH_VALUES}) against DEVELOPMENT folds only, "
                              "then compare each pair's winner rank-IC via the same one-sided "
                              "block-bootstrap/Holm-Bonferroni significance test --gbt-tree-search "
                              "uses. Writes to '{output_dir}_max_features_search/max_features_N/' "
                              "plus a top-level max_features_search_comparison.csv - composes "
                              "with --fast/--no-temporal-operators/--seed, but not with --resume "
                              "or --gbt-tree-search.")
    parser.add_argument("--min-mutation", type=float, default=None,
                         help="override GAConfig.min_mutation (lower bound of the mutation-rate "
                              "range mutation_config draws from)")
    parser.add_argument("--max-mutation", type=float, default=None,
                         help="override GAConfig.max_mutation (upper bound of the mutation-rate "
                              "range mutation_config draws from)")
    parser.add_argument("--temporal-wrap-rate", type=float, default=None,
                         help="override GAConfig.temporal_wrap_rate (per-mutation-event "
                              "probability of structurally wrapping a temporal operator - see "
                              "genome/grammar.py's mutate())")
    parser.add_argument("--temporal-unwrap-rate", type=float, default=None,
                         help="override GAConfig.temporal_unwrap_rate (per-mutation-event "
                              "probability of structurally unwrapping a temporal operator)")
    parser.add_argument("--max-features", type=int, default=None,
                         help="override GAConfig.max_features (cap on how many leaf features a "
                              "single evolved expression can combine via crossover) - also sets "
                              "GAConfig.max_nesting to the same value (no separate --max-nesting "
                              "flag; matches run_optuna_sweep.py's convention of keeping them "
                              "equal rather than treating max_nesting as an independent axis)")
    parser.add_argument("--tournament-size", type=int, default=None,
                         help="override GAConfig.tournament_size (selection pressure - how many "
                              "individuals compete per tournament-selection draw)")
    parser.add_argument("--mutation-method", choices=["flat", "increasing", "decreasing"], default=None,
                         help="override GAConfig.mutation_method (how the mutation rate moves "
                              "across generations between min_mutation/max_mutation)")
    parser.add_argument("--gbt-max-iter", type=int, default=None,
                         help="override GAConfig.gbt_max_iter (GBTRegressor's boosting-iteration/"
                              "tree count used to score every individual's fitness)")
    parser.add_argument("--num-threads", type=int, default=None,
                         help="override GAConfig.num_threads (the ThreadPoolExecutor size "
                              "evaluate_population uses - under fit_backend=local this maps 1:1 "
                              "to real concurrent CPU core demand, since every XGBRegressor fit "
                              "uses n_jobs=1). Lower this if running multiple GA processes "
                              "concurrently on the same host oversubscribes its cores - e.g. two "
                              "processes at the default 8 each means 16 threads competing for "
                              "bialobog's 4 physical cores. Under --execution cluster, this "
                              "flips the other way - it also sizes the requested YARN container's "
                              "vcores (spark.yarn.am.cores), so raise it to use a whole dedicated "
                              "node's cores instead of leaving it at bialobog-safe defaults. Part "
                              "of GAConfig, so changing it changes the run's fingerprint (a "
                              "differently-threaded run won't resume an existing checkpoint - see "
                              "run_ga.py's module docstring).")
    parser.add_argument("--driver-memory", default=None,
                         help="override the SparkSession's spark.driver.memory (default 8g, "
                              "lowered from an original 14g - inherited from the old spark "
                              "fit-backend's heavier driver-side aggregation, oversized for "
                              "fit_backend=local, where Spark only does a lightweight one-time "
                              "per-fold HDFS read - once it became clear a `--execution cluster` "
                              "AM's footprint directly eats into root.bvail's queue-level AM-share "
                              "cap, so a smaller default lets more seeds run concurrently under "
                              "the same cap). Not part of "
                              "GAConfig, so this does NOT affect run fingerprinting/resume. "
                              "Lower this (e.g. '4g') to ease memory pressure when running "
                              "multiple GA processes concurrently on the same host.")
    parser.add_argument("--execution", choices=["local", "cluster"], default="local",
                         help="where this run's whole process executes (see "
                              "~/.claude/plans/clever-scribbling-mccarthy.md). 'local' (default, "
                              "unchanged behavior) - runs here, on this host, against a "
                              "master=\"yarn\" SparkSession (spark.executor.instances=8). "
                              "'cluster' - packages the current code + the pre-packed venv "
                              "(research/scripts/pack_venv.sh) and resubmits this EXACT command "
                              "as its own YARN application via `spark-submit --deploy-mode "
                              "cluster` (see cluster_submit.py), so the whole run - not just its "
                              "Spark reads - lands on one dedicated YARN container elsewhere in "
                              "the cluster instead of contending for this host's cores. Inside "
                              "that container the run is identical to a 'local' run except its "
                              "SparkSession requests a single tiny executor instead of 8 (still "
                              "master=\"yarn\" - see _build_spark_session's docstring for why a "
                              "real, if minimal, executor is required rather than 0). Not part of "
                              "GAConfig - does NOT affect fingerprinting/resume/output_dir, same "
                              "as --driver-memory below. Prints the submitted YARN application "
                              "id and returns immediately (does not block for the run's actual "
                              "duration) rather than running the GA itself when first invoked; "
                              "the resubmitted child process (SUPERFEATURES_CLUSTER_CHILD=1 in "
                              "its environment) is what actually runs it.")
    parser.add_argument("--with-dev-folds", action="store_true",
                         help="also run development folds (default: final-test folds only). "
                              "Development-fold results aren't currently read by anything in this "
                              "pipeline (compare_ga_runs.py is scoped to final-test folds only - "
                              "see its load_final_test_fold_results docstring - and the "
                              "block_length-calibration-from-dev-data idea in "
                              "evaluation_framework.md was never implemented), so unless you pass "
                              "this flag, final-test-only skips real compute that nothing "
                              "downstream consumes. --gbt-tree-search/--max-features-search "
                              "ignore this flag - they always use development folds regardless.")
    parser.add_argument("--dev-only", action="store_true",
                         help="run ONLY development folds, skipping final-test folds entirely "
                              "(the reverse of the default) - overrides --with-dev-folds "
                              "(development folds run either way). Useful for exercising the "
                              "pipeline without touching final-test data at all - e.g. validating "
                              "--execution cluster - since development-fold results aren't read "
                              "by anything downstream (see --with-dev-folds's help text above), "
                              "the final-test-scoped summary builders (build_final_test_summary/"
                              "build_pairwise_comparisons/build_winner_composition) are skipped "
                              "too rather than run against zero final-test results.")
    args = parser.parse_args()
    if args.execution == "cluster" and os.environ.get("SUPERFEATURES_CLUSTER_CHILD") != "1":
        # Not yet running inside the submitted cluster job - package + resubmit this exact
        # command as its own YARN application and exit, instead of running the GA here. The
        # resubmitted child gets the identical argv (including --execution cluster, which is
        # harmless there - SUPERFEATURES_CLUSTER_CHILD=1 in its environment short-circuits this
        # branch and it falls through to the normal run path below, master="yarn" with the tiny
        # cluster-child executor sizing - see _build_spark_session's docstring).
        import cluster_submit
        app_id = cluster_submit.submit_cluster_run(
            sys.argv[1:],
            num_threads=args.num_threads if args.num_threads is not None else 8,
            driver_memory=args.driver_memory if args.driver_memory is not None else "8g",
            walk_forward_namespace="walk_forward_fast" if args.fast else "walk_forward_full",
        )
        print(f"Submitted YARN application {app_id} - it runs independently from here on.")
        print(f"Monitor with: yarn application -status {app_id}")
        print(f"Logs (after it finishes, or with -am for the AM's live log): "
              f"yarn logs -applicationId {app_id}")
        sys.exit(0)
    _dev_only_sweep_flags = {"--gbt-tree-search": args.gbt_tree_search,
                              "--max-features-search": args.max_features_search}
    _active_sweep_flags = [name for name, on in _dev_only_sweep_flags.items() if on]
    if len(_active_sweep_flags) > 1:
        parser.error(f"{' and '.join(_active_sweep_flags)} are mutually exclusive - pass only one "
                      f"hyperparameter sweep at a time.")
    config = FAST_GA_CONFIG if args.fast else FULL_GA_CONFIG
    # Defaults for this script specifically (not GAConfig's own dataclass defaults, which stay
    # "spark"/"rmse" for other callers): local fit backend and rank_ic fitness, no output_dir
    # suffix either way - see --fit-backend/--fitness-metric below for how to override back.
    config = GAConfig(**{**config.__dict__, "fit_backend": "local", "fitness_metric": "rank_ic"})
    overrides = {}
    if args.fit_backend is not None:
        overrides["fit_backend"] = args.fit_backend
    if args.no_baseline_c:
        overrides["run_baseline_c"] = False
    if args.no_temporal_operators:
        overrides["enable_temporal_operators"] = False
        overrides["output_dir"] = config.output_dir + "_no_temporal"
    if args.fitness_metric is not None:
        overrides["fitness_metric"] = args.fitness_metric
    if args.seed is not None:
        overrides["random_seed"] = args.seed
        overrides["output_dir"] = overrides.get("output_dir", config.output_dir) + f"_seed{args.seed}"
    # GA-mechanics overrides (e.g. a hyperparameter combination found by an Optuna sweep) - no
    # output_dir suffix, same as --no-baseline-c above: _find_matching_run's fingerprinting
    # already ensures a run with any of these set never collides with/resumes a mismatched run,
    # it just won't get its own distinct family-symlink name the way --seed/--fitness-metric do.
    if args.min_mutation is not None:
        overrides["min_mutation"] = args.min_mutation
    if args.max_mutation is not None:
        overrides["max_mutation"] = args.max_mutation
    if args.temporal_wrap_rate is not None:
        overrides["temporal_wrap_rate"] = args.temporal_wrap_rate
    if args.temporal_unwrap_rate is not None:
        overrides["temporal_unwrap_rate"] = args.temporal_unwrap_rate
    if args.max_features is not None:
        overrides["max_features"] = args.max_features
        # max_nesting tracks max_features, not swept/overridden independently - same convention
        # run_optuna_sweep.py's _suggest_config follows (GAConfig defaults have always had them
        # equal). There's no separate --max-nesting flag, so without this a --max-features
        # override would silently leave max_nesting at GAConfig's own default instead.
        overrides["max_nesting"] = args.max_features
    if args.tournament_size is not None:
        overrides["tournament_size"] = args.tournament_size
    if args.mutation_method is not None:
        overrides["mutation_method"] = args.mutation_method
    if args.gbt_max_iter is not None:
        overrides["gbt_max_iter"] = args.gbt_max_iter
    if args.num_threads is not None:
        overrides["num_threads"] = args.num_threads
    if overrides:
        config = GAConfig(**{**config.__dict__, **overrides})

    if args.gbt_tree_search:
        run_gbt_tree_search(config)
        sys.exit(0)
    if args.max_features_search:
        run_max_features_search(config)
        sys.exit(0)

    # `family` (the plain flag-derived name, e.g. "ga_fast_seed7") becomes a symlink pointing at
    # whichever ga_runs/ directory this invocation ends up using - see _update_latest_link - so
    # every existing doc/notebook that reads e.g. "ga_fast/final_test_summary.csv" keeps working
    # against whichever run is current, while scripts/list_ga_runs.py can enumerate every run of
    # every family from ga_runs/ alone.
    family = config.output_dir
    if args.resume is not None:
        matched_dir = args.resume
    else:
        matched_dir = _find_matching_run(config)
    if matched_dir is not None:
        print(f"Found an existing run with an identical config at '{matched_dir}' - reusing it "
              f"(already-complete folds skip, in-progress folds resume from checkpoint, "
              f"never-started folds run fresh).")
        config = GAConfig(**{**config.__dict__, "output_dir": matched_dir})
    else:
        run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        config = GAConfig(**{**config.__dict__, "output_dir": f"ga_runs/{family}_{run_stamp}"})
    main(config, latest_link=family,
         skip_dev_folds=not (args.with_dev_folds or args.dev_only),
         skip_final_test_folds=args.dev_only,
         driver_memory=args.driver_memory if args.driver_memory is not None else "8g",
         # Always "yarn" now - see _build_spark_session's docstring for why the cluster-child
         # branch here used to pass "local[*]" and why that silently killed every full-scale
         # cluster run via YARN's own AM-liveness timeout. _build_spark_session itself still reads
         # SUPERFEATURES_CLUSTER_CHILD to size the cluster-child's executor request very small.
         master="yarn")
