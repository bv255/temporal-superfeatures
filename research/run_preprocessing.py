"""
Package-backed entry point for the PreProcessing pipeline. Two independent axes, both on
`superfeatures.config.PipelineConfig`:
  - scale: full (default) vs. `--fast` - a reduced stock universe/feature set/
    ConsensusFeatureSelector.TERMINAL_CAP, ported from PreProcessing_test.ipynb's own
    disabled-by-default "FAST TEST PIPELINE" cell-8 block (see CLAUDE.md) rather than invented
    fresh - for quick end-to-end iteration without paying full-scale wall-clock cost.
  - (retired) precomputed temporal feature columns - `PipelineConfig.add_temporal_features`
    defaults to False now; see config.py's module docstring.

Mirrors the original notebook's cells section-for-section:
  cell 0            -> _build_spark_session() (env vars + the explicitly-sized SparkSession;
                        Utils.__init__ reuses this same session via its own getOrCreate() call,
                        same as the notebook - see _build_spark_session's docstring)
  cell 8            -> _build_base_fundamentals(): Utils() construction through
                        add_return_lag_features, including the fast-mode universe/feature caps
  cell 9            -> _build_and_checkpoint_experiment_1_df()
  cell 11           -> superfeatures.evaluation.compute_fold_boundaries()
  cells 15/17       -> _run_folds(): calls superfeatures.preprocessing.run_fold() per fold
  cell 19           -> dead/commented-out in the notebook, not ported
  cell 20           -> spark.stop() at the end of main()

Usage: ~/pyspark-venv/bin/python3 run_preprocessing.py            # full scale
       ~/pyspark-venv/bin/python3 run_preprocessing.py --fast     # reduced scale
"""
import argparse
import os
import random

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from superfeatures.config import PipelineConfig, FULL_CONFIG, FAST_CONFIG
from superfeatures.preprocessing import Utils, run_fold, ConsensusFeatureSelector
from superfeatures.evaluation import compute_fold_boundaries


def _build_spark_session(fast_mode: bool = False) -> SparkSession:
    """
    Cell 0's env vars + explicitly-sized SparkSession - built here, BEFORE Utils() is
    constructed, so that Utils.__init__'s own `SparkSession.builder...getOrCreate()` call
    reuses this session (getOrCreate() returns the already-active session for a matching
    appName rather than rebuilding it) instead of falling back to the YARN cluster's global
    defaults (4 x 2 x 4g) - see CLAUDE.md's "PreProcessing's SparkSession is explicitly sized...
    rather than relying on the cluster's global defaults." Utils.__init__ registers its own temp
    views (identical table list to cell 0's), so this function doesn't duplicate that step.

    fast_mode scales the executor request down (3 x 1 core x 2g instead of 6 x 2 x 8g) - the
    original 10x2x8g sizing was tuned for full scale's whole-universe/full-date-range join, which
    --fast never touches (Finance-sector-only, capped to fast_stocks_per_sector companies and
    fast_n_features columns - see PipelineConfig/_apply_fast_mode_caps). Keeps --fast runs from
    requesting far more of this shared cluster's YARN capacity than the reduced workload needs,
    and lets one land even when a full-scale run (this pipeline's or someone else's) already has
    most executor slots occupied. Driver memory is untouched by this - the driver runs locally on
    the gateway host in client mode, so it isn't YARN-allocated capacity either way.

    Full scale's own executor_instances was cut from 10 to 6 on 2026-08-27, in response to a
    spell of widespread node instability on this cluster (most of skrzat2/3/4/7/8/9/10/11/12/13
    cycling through UNHEALTHY from local-disk exhaustion under a path unrelated to this project's
    own data - see the "Environment" section's shared-infra note). Fewer executors doesn't fix
    that (it's node/cluster-side, not something a job's resource request controls), but it does
    reduce this job's own exposure: fewer nodes touched simultaneously means fewer chances a
    marginal node tips into UNHEALTHY (or drops out entirely) mid-shuffle under this job's own
    write load specifically, and fewer of this job's own tasks are disrupted if one does.
    Executor memory is deliberately left at 8g, not reduced alongside instance count - less
    memory would push MORE data into shuffle spill on local disk, working against the exact
    problem this change responds to, not for it.
    """
    os.environ["SPARK_HOME"] = "/opt/spark"
    os.environ["HADOOP_HOME"] = "/opt/hadoop"
    os.environ["HADOOP_CONF_DIR"] = "/opt/hadoop-3.4.1/etc/hadoop"
    os.environ["YARN_CONF_DIR"] = "/opt/hadoop-3.4.1/etc/hadoop"
    os.environ["PYSPARK_PYTHON"] = "/home/bvail/pyspark-venv/bin/python3"
    os.environ["PYSPARK_DRIVER_PYTHON"] = "/home/bvail/pyspark-venv/bin/python3"

    executor_instances = "5" if fast_mode else "6"
    executor_cores = "1" if fast_mode else "2"
    executor_memory = "2g" if fast_mode else "8g"

    print(f"Starting SparkSession (this can take a moment on YARN)... "
          f"[fast_mode={fast_mode}: {executor_instances} executors x {executor_cores} core(s) x {executor_memory}]")
    spark = (
        SparkSession.builder
        .master("yarn")
        .appName("SuperFeatures_2024")
        .config("spark.executor.instances", executor_instances)
        .config("spark.executor.cores", executor_cores)
        .config("spark.executor.memory", executor_memory)
        .config("spark.driver.memory", "8g")
        # Workaround for a cluster-capacity issue, not a pipeline bug: this driver runs ON
        # bialobog, which is ALSO an HDFS datanode - HDFS's default block placement policy
        # always tries the writer's own local node first, so every block write tries
        # 192.168.2.1 (bialobog) first, guaranteed, and it's ~99.8% full. No rack topology is
        # configured (single flat rack), so the other 2 of 3 replicas are drawn close to
        # randomly from the remaining nodes, 3 of which (skrzat3/4/5) are also nearly full -
        # enough to exhaust the default dfs.client.block.write.retries=3 and abort the whole
        # SparkContext before any work runs. replication=1 means a block only needs ONE good
        # node instead of 3 simultaneously good ones; more retries gives extra chances to skip
        # past the bad picks. Fine for this pipeline's output (fully regenerable from the raw
        # source tables, doesn't need 3x durability) but does trade away HDFS's normal fault
        # tolerance for it. Same rationale/config in run_ga.py's _build_spark_session.
        .config("spark.hadoop.dfs.client.block.write.retries", "10")
        .config("spark.hadoop.dfs.replication", "1")
        .getOrCreate()
    )
    print("SparkSession ready.")
    return spark


# GA_2025_final.ipynb and IndustryApplication.ipynb hardcode references to these specific
# features (demo cells, plotting, and the "super feature" expressions) - fast mode guarantees
# they're included as candidates so they aren't at the mercy of the arbitrary column order the
# fundamentals SQL join returns. They can still get dropped further downstream by
# drop_na_features/remove_static_features if the reduced stock sample genuinely lacks enough
# data for them - this only guarantees they're considered, not that they survive.
_FAST_REQUIRED_FEATURES = [
    'ff_capex_sales', 'ff_chg_cash_cf', 'ff_compr_inc_for_curn_adj', 'ff_dfd_tax',
    'ff_dfd_tax_assets_lt', 'ff_div_yld', 'ff_earn_yld', 'ff_eff_int_rate', 'ff_eps_reported',
    'ff_fcf_yld', 'ff_fin_uses_cf', 'ff_fy_length_days', 'ff_net_debt', 'ff_pbk', 'ff_pcf',
    'ff_psales_dil', 'ff_roic', 'ff_stk_purch_cf',
]
_FAST_ID_COLS = ['fsym', 'factset_entity_id', 'factset_sector_desc', 'fsym_id', 'date', 'fund_report_date']


def _apply_fast_mode_caps(df_fundamentals, df_prices, config: PipelineConfig):
    """
    Fast mode's two universe/feature caps, ported verbatim from PreProcessing_test.ipynb's
    disabled "FAST TEST PIPELINE" cell-8 block:
      1. Caps df_fundamentals down to config.fast_n_features feature columns (on top of the
         id/date/sector columns downstream code expects), always including any of
         _FAST_REQUIRED_FEATURES present, backfilled with whatever else is available.
      2. Caps the universe to config.fast_stocks_per_sector stocks PER SECTOR (not a flat
         overall limit, so no sector is left with only 1-2 companies pooled together), sampled
         RANDOMLY (seeded via config.fast_sample_seed) rather than a deterministic top-N, via a
         collect_list-then-random.sample idiom (unlike ConsensusFeatureSelector, which no longer
         samples companies at all - see its class docstring - this cap is the point of --fast
         mode, not an approximation to remove).
    """
    available_cols = [c for c in df_fundamentals.columns if c not in _FAST_ID_COLS]
    required_present = [c for c in _FAST_REQUIRED_FEATURES if c in available_cols]
    fill_cols = [c for c in available_cols if c not in required_present]
    feature_cols = required_present + fill_cols[:max(0, config.fast_n_features - len(required_present))]
    df_fundamentals = df_fundamentals.select(*_FAST_ID_COLS, *feature_cols)
    print(f"[FAST MODE] df_fundamentals capped to {len(df_fundamentals.columns)} columns "
          f"({len(_FAST_ID_COLS)} id/date/sector + {len(feature_cols)} features, "
          f"including {len(required_present)}/{len(_FAST_REQUIRED_FEATURES)} required features)")

    random.seed(config.fast_sample_seed)
    fsym_by_sector = (
        df_fundamentals.select('fsym', 'factset_sector_desc').distinct()
        .groupBy('factset_sector_desc')
        .agg(F.collect_list('fsym').alias('fsyms'))
        .collect()
    )
    fsym_list = []
    for row in fsym_by_sector:
        fsyms = row['fsyms']
        n = min(len(fsyms), config.fast_stocks_per_sector)
        fsym_list.extend(random.sample(fsyms, n))
    df_fundamentals = df_fundamentals.filter(df_fundamentals.fsym.isin(fsym_list))
    df_prices = df_prices.filter(df_prices.fsym_id.isin(fsym_list))
    print(f"[FAST MODE] Randomly capped to {len(fsym_list)} stocks "
          f"(up to {config.fast_stocks_per_sector} per sector, seed={config.fast_sample_seed})")

    return df_fundamentals, df_prices


def _build_base_fundamentals(config: PipelineConfig):
    """Cell 8: Utils() construction through add_return_lag_features. Returns
    (utils, base_fundamentals_df, monthly_returns, candidate_feature_columns)."""
    print("=== Data processing pipeline: starting ===")
    utils = Utils(
        num_partitions=config.num_partitions,
        start_date=config.start_date,
        market_value_threshold=config.market_value_threshold,
        sector_filter=config.sector_filter,
    )
    print(f"Pipeline scoped to sector_filter={utils.sector_filter!r} (pushed into the fundamentals/price SQL WHERE clauses)")

    df_fundamentals = utils.get_fundamental_data()
    df_prices = utils.get_price_data()

    if config.fast_mode:
        df_fundamentals, df_prices = _apply_fast_mode_caps(df_fundamentals, df_prices, config)

    df_fundamentals.cache()
    df_prices.cache()

    print("Aligning fundamentals and price data on overlapping date ranges...")
    aligned_fundamentals, aligned_prices = utils.align_data(df_fundamentals, df_prices)
    df_fundamentals.unpersist()
    df_prices.unpersist()
    del df_fundamentals, df_prices

    print("Smoothing price outliers (causal)...")
    smoothed_prices = utils.replace_price_outliers(aligned_prices, "adj_price", std=10)
    aligned_prices.unpersist()
    del aligned_prices

    print("Dropping non-numeric columns from fundamentals...")
    cleaned_fundamentals, _ = utils.drop_non_numeric_columns(
        aligned_fundamentals, keep_columns=['date', 'fsym', 'factset_sector_desc', 'fsym_id', 'fund_report_date']
    )
    aligned_fundamentals.unpersist()
    del aligned_fundamentals

    # drop_na_features/remove_static_features are deferred to per-fold, train-only computation
    # inside run_fold() - see preprocessing/pipeline.py's module docstring / CLAUDE.md's
    # walk-forward section for why. This cell only produces the full candidate feature set.
    candidate_feature_columns = [
        c for c in cleaned_fundamentals.columns
        if c not in ['date', 'fsym', 'factset_sector_desc', 'fsym_id', 'fund_report_date']
    ]

    print("Filling infinite and missing values (causal, backward-only carry-forward)...")
    base_fundamentals_df = utils.interpolate_missing_values(cleaned_fundamentals, candidate_feature_columns)
    del cleaned_fundamentals

    if config.add_temporal_features:
        print("Adding temporal feature columns (lag/delta/growth/mean/std)...")
        base_fundamentals_df, temporal_feature_columns = utils.add_temporal_features(
            base_fundamentals_df, candidate_feature_columns,
            lag_periods=config.temporal_lag_periods, window_sizes=config.temporal_window_sizes,
        )
        candidate_feature_columns = candidate_feature_columns + temporal_feature_columns
        print(f"add_temporal_features: candidate_feature_columns now {len(candidate_feature_columns)} "
              f"({len(temporal_feature_columns)} temporal + "
              f"{len(candidate_feature_columns) - len(temporal_feature_columns)} original)")
    else:
        print("Temporal feature augmentation skipped (config.add_temporal_features=False).")

    base_fundamentals_df.cache()
    print(f"base_fundamentals_df cached and materialized, rows={base_fundamentals_df.count()}")
    utils.count_total_infinity_values(base_fundamentals_df)

    print("Calculating monthly returns...")
    monthly_returns = utils.calculate_monthly_returns(smoothed_prices).cache()
    print(f"monthly_returns cached and materialized, rows={monthly_returns.count()}")

    print("Adding return-lag feature columns (prev_month_return, prev_month_sector_return)...")
    monthly_returns.unpersist()
    monthly_returns = utils.add_return_lag_features(monthly_returns).cache()
    print(f"monthly_returns (with lag features) cached and materialized, rows={monthly_returns.count()}")

    del smoothed_prices
    print("=== Data processing pipeline: done ===")
    return utils, base_fundamentals_df, monthly_returns, candidate_feature_columns


def _build_and_checkpoint_experiment_1_df(utils, base_fundamentals_df, monthly_returns,
                                           candidate_feature_columns, config: PipelineConfig):
    """Cell 9: joins fundamentals to next-month returns (as-of), writes the HDFS parquet
    checkpoint every fold reads back fresh. Returns the (possibly extended)
    candidate_feature_columns list - prev_month_return/prev_month_sector_return are folded in
    here, matching the notebook."""
    print("=== Building the full-range base dataset (experiment_1_df) + walk-forward checkpoint: starting ===")
    candidate_feature_columns = candidate_feature_columns + ['prev_month_return', 'prev_month_sector_return']

    experiment_1_df = utils.feature_selection_dataset(base_fundamentals_df, monthly_returns).cache()
    print(f"experiment_1_df cached and materialized, rows={experiment_1_df.count()}")

    print(f"Writing walk-forward base checkpoint to {config.walk_forward_base_path} ...")
    experiment_1_df.write.mode("overwrite").parquet(config.walk_forward_base_path)
    print("Walk-forward base checkpoint written.")

    experiment_1_df.unpersist()
    base_fundamentals_df.unpersist()
    monthly_returns.unpersist()
    print("=== Building the full-range base dataset: done ===")
    return candidate_feature_columns


def _run_folds(spark, utils, config: PipelineConfig, candidate_feature_columns, only_folds=None):
    """Cells 11/15/17: read the checkpoint back fresh, compute fold boundaries, run every
    development fold then every final-test fold.

    only_folds: optional list of "category/fold_NN" strings (e.g. "final_test/fold_03") - when
    set, restricts the run to just those folds instead of the full dev+final-test sweep. See
    --only-folds's own help text for why this exists."""
    print("=== Walk-forward fold definitions: starting ===")
    wf_base_df = spark.read.parquet(config.walk_forward_base_path).cache()
    print(f"wf_base_df read from checkpoint, rows={wf_base_df.count()}")
    boundaries = compute_fold_boundaries(wf_base_df, config)
    dev_folds = boundaries['dev_folds']
    final_test_folds = boundaries['final_test_folds']
    print(f"{len(dev_folds)} development folds, {len(final_test_folds)} final-test folds "
          f"({boundaries['n_years']} distinct target years: {boundaries['all_years']})")

    if only_folds:
        wanted = {f"{config.walk_forward_namespace}/{spec}" for spec in only_folds}
        dev_folds = [f for f in dev_folds if f['output_dir'] in wanted]
        final_test_folds = [f for f in final_test_folds if f['output_dir'] in wanted]
        matched = {f['output_dir'] for f in dev_folds} | {f['output_dir'] for f in final_test_folds}
        unmatched = wanted - matched
        if unmatched:
            print(f"WARNING --only-folds: {unmatched} did not match any computed fold boundary - typo, "
                  f"or a fold number this dataset's actual year range doesn't produce?")
        print(f"--only-folds filter applied: running {len(dev_folds)} development fold(s), "
              f"{len(final_test_folds)} final-test fold(s)")
    print("=== Walk-forward fold definitions: done ===")

    terminal_cap = config.fast_terminal_cap if config.fast_mode else ConsensusFeatureSelector.TERMINAL_CAP

    print(f"=== Development folds: starting ({len(dev_folds)} folds) ===")
    for fold in dev_folds:
        run_fold(
            base_df=wf_base_df,
            train_years=fold['train_years'],
            eval_years=fold['eval_years'],
            eval_label=fold['eval_label'],
            output_dir=fold['output_dir'],
            candidate_feature_columns=candidate_feature_columns,
            utils=utils,
            spark=spark,
            embargo_months=config.embargo_months,
            random_seed=config.random_seed,
            terminal_cap=terminal_cap,
        )
    print("=== Development folds: done ===")

    if config.fast_mode:
        # --fast only ever feeds run_ga.py's dev-fold-only hyperparameter sweeps
        # (--gbt-tree-search/--max-features-search) - final-test folds are never read by anything
        # in --fast mode, so skip building them entirely rather than spending real compute on
        # output nothing consumes. target_final_test_folds is NOT touched anywhere in this file
        # (still the same value compute_fold_boundaries used above to work out where the dev
        # window has to stop) - only skipping the build loop below, not the year reservation
        # itself, so the dev folds' upper boundary stays anchored exactly where it already was.
        print(f"=== Final-test folds: skipped (--fast mode, {len(final_test_folds)} fold(s) "
              f"would have been built) ===")
    else:
        print(f"=== Final-test folds: starting ({len(final_test_folds)} folds) ===")
        for fold in final_test_folds:
            run_fold(
                base_df=wf_base_df,
                train_years=fold['train_years'],
                eval_years=fold['eval_years'],
                eval_label=fold['eval_label'],
                output_dir=fold['output_dir'],
                candidate_feature_columns=candidate_feature_columns,
                utils=utils,
                spark=spark,
                embargo_months=config.embargo_months,
                random_seed=config.random_seed,
                inner_val_years=fold['inner_val_years'],
                terminal_cap=terminal_cap,
            )
        print("=== Final-test folds: done ===")


def main(config: PipelineConfig, only_folds=None):
    spark = _build_spark_session(fast_mode=config.fast_mode)
    try:
        if only_folds:
            # Skip the expensive global setup (get_fundamental_data/get_price_data/causal
            # cleaning/checkpoint write - the ~40-70min part that has nothing to do with any
            # individual fold) and reuse the checkpoint a PRIOR run of this same config already
            # wrote to HDFS. candidate_feature_columns doesn't need to be recomputed either - the
            # checkpoint's own schema already encodes it (every column except the fixed
            # id/target set feature_selection_dataset always outputs - see that method's own
            # final .select() for the authoritative list this mirrors).
            print(f"=== --only-folds {only_folds}: reusing existing checkpoint at "
                  f"{config.walk_forward_base_path}, skipping global setup ===")
            utils = Utils(
                num_partitions=config.num_partitions,
                start_date=config.start_date,
                market_value_threshold=config.market_value_threshold,
                sector_filter=config.sector_filter,
            )
            wf_base_df = spark.read.parquet(config.walk_forward_base_path)
            non_feature_columns = ['fsym', 'date', 'factset_sector_desc', 'target_date',
                                    'return_year', 'return_month', 'monthly_return']
            candidate_feature_columns = [c for c in wf_base_df.columns if c not in non_feature_columns]
            print(f"Recovered {len(candidate_feature_columns)} candidate feature columns from "
                  f"the checkpoint's own schema (no rebuild needed).")
            _run_folds(spark, utils, config, candidate_feature_columns, only_folds=only_folds)
        else:
            utils, base_fundamentals_df, monthly_returns, candidate_feature_columns = _build_base_fundamentals(config)
            candidate_feature_columns = _build_and_checkpoint_experiment_1_df(
                utils, base_fundamentals_df, monthly_returns, candidate_feature_columns, config
            )
            _run_folds(spark, utils, config, candidate_feature_columns)
    finally:
        print("Stopping SparkSession...")
        spark.stop()
        print("SparkSession stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true",
                         help="reduced-scale run (capped universe/feature set/TERMINAL_CAP) - "
                              "writes to the walk_forward_fast/ namespace instead of walk_forward_full/")
    parser.add_argument("--only-folds", nargs="+", default=None, metavar="CATEGORY/FOLD_NN",
                         help="Only (re)run these specific folds, e.g. "
                              "--only-folds final_test/fold_03 final_test/fold_04 final_test/fold_05 "
                              "- reuses the existing HDFS checkpoint (walk_forward_{full,fast}/base/"
                              "experiment_1_df.parquet) instead of rebuilding it, so a run that died "
                              "partway through the fold loop can finish the remaining folds without "
                              "redoing the expensive global setup. Requires that checkpoint to "
                              "already exist from a prior run of this same --fast/full scale - if it "
                              "doesn't, this will fail reading it; run without --only-folds first.")
    args = parser.parse_args()
    main(FAST_CONFIG if args.fast else FULL_CONFIG, only_folds=args.only_folds)
