"""
End-to-end comparison test for GeneticAlgorithm1's fit_backend="local" path (xgboost.XGBRegressor,
in-process pandas/numpy) against the default fit_backend="spark" path (pyspark.ml GBTRegressor) -
see GAConfig.fit_backend's docstring and docs/paper_tables.md's local-backend-methodology-fork
notes.

This test does NOT assert numerical equality between backends - XGBoost's GBT and Spark's
GBTRegressor are different implementations (different split-gain regularization/RNG internals)
and are not expected to agree even at matched seeds/hyperparameters. This test instead asserts
both backends are individually
*sane* and *consistent* with each other in the ways that matter for trusting the local backend's
wiring: same reachable leaf vocabulary, non-decreasing best-fitness trajectory (elitism holds
under either backend, since it's a property of GeneticAlgorithm1.run's loop, not of the fit
implementation), and same-order-of-magnitude final fitness.

Local SparkSession only - the local backend still needs Spark for the once-per-fold data load
(see operators/arithmetic_local.py's/temporal_local.py's module docstrings).
"""
import os
import random
from datetime import date

import pytest
from pyspark.sql import Row, SparkSession
from pyspark.sql.functions import min as spark_min

from superfeatures.ga.engine import GeneticAlgorithm1
from superfeatures.data.panel import GAPreprocessing
from superfeatures.reporting.history import ResultsTracker

FEATURES = ["feat_a", "feat_b", "feat_c", "feat_d"]
SELECTED_FEATURES = FEATURES + ["prev_month_return", "prev_month_sector_return"]


@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("test_ga_engine_local_backend")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()


def _fold_frame(spark, n_rows=30, seed=7):
    rnd = random.Random(seed)
    rows = []
    for i in range(n_rows):
        month = 1 + (i % 12)
        base = rnd.uniform(-1, 1)
        rows.append(Row(
            fsym=f"F{i}",
            date=date(2020, month, 1),
            target_date=date(2020, month, 1),
            prev_month_return=rnd.uniform(-0.05, 0.05),
            prev_month_sector_return=rnd.uniform(-0.05, 0.05),
            monthly_return=0.3 * base + rnd.uniform(-0.02, 0.02),
            feat_a=base + rnd.uniform(-0.1, 0.1),
            feat_b=-base + rnd.uniform(-0.1, 0.1),
            feat_c=rnd.uniform(-1, 1),
            feat_d=base * 2 + rnd.uniform(-0.2, 0.2),
        ))
    return spark.createDataFrame(rows)


def _bare_gapreprocessing(spark, train_df, eval_df):
    gp = GAPreprocessing.__new__(GAPreprocessing)
    gp.spark = spark
    gp.sector_column = "factset_sector_desc"
    gp.num_partitions = 2
    gp.selected_features = list(SELECTED_FEATURES)
    gp.train_df = train_df
    gp.eval_df = eval_df
    gp.has_inner_validation = False
    gp.true_eval_df = eval_df
    # Set the same way GAPreprocessing.__init__ does (see data/panel.py) - needed since
    # generate_training_testing_data/get_true_test_frame's lookback-buffer trim reads these.
    gp.eval_start_date = eval_df.agg(spark_min("target_date")).first()[0]
    gp.true_eval_start_date = gp.eval_start_date
    return gp


def _run_ga(spark, train_df, eval_df, tmp_path, fit_backend, seed=42):
    gp = _bare_gapreprocessing(spark, train_df, eval_df)
    tracker = ResultsTracker()

    random.seed(seed)
    ga = GeneticAlgorithm1(
        gapreprocessing=gp,
        sector="Finance",
        results_tracker=tracker,
        spark=spark,
        tournament_size=2,
        mutation_config=("flat", True, False, 0.4, 0.1),
        generations=3,
        population_size=4,
        num_threads=1,
        fit_backend=fit_backend,
    )

    ga.initialize_population()

    os.makedirs(tmp_path, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        ga.run()
    finally:
        os.chdir(cwd)

    return ga, tracker


def test_local_and_spark_backends_are_both_sane_and_consistent(spark, tmp_path):
    train_df = _fold_frame(spark, n_rows=30, seed=7).cache()
    eval_df = _fold_frame(spark, n_rows=10, seed=99).cache()
    train_df.count()
    eval_df.count()

    spark_ga, spark_tracker = _run_ga(spark, train_df, eval_df, tmp_path / "spark", fit_backend="spark")
    local_ga, local_tracker = _run_ga(spark, train_df, eval_df, tmp_path / "local", fit_backend="local")

    spark_best, _, _ = spark_tracker.get_fitnesses()
    local_best, _, _ = local_tracker.get_fitnesses()

    # Same closed leaf vocabulary reachable - a basic wiring sanity check, not a numerical one.
    assert sorted(spark_ga.features) == sorted(local_ga.features)

    # Elitism holds under either backend - non-decreasing best fitness is a property of
    # GeneticAlgorithm1.run's loop, not of the fit implementation.
    assert all(b2 >= b1 - 1e-9 for b1, b2 in zip(spark_best, spark_best[1:]))
    assert all(b2 >= b1 - 1e-9 for b1, b2 in zip(local_best, local_best[1:]))

    # Both backends reach a valid winning individual from the population's own leaf vocabulary.
    assert spark_ga.best_individual is not None
    assert local_ga.best_individual is not None

    # Same rough neighborhood, not equal - RMSE fitness is negative (GA maximizes -RMSE), so
    # "same sign, not orders of magnitude apart" is the right plausibility bar here.
    assert (spark_best[-1] < 0) == (local_best[-1] < 0)
    assert abs(spark_best[-1]) < 10 * (abs(local_best[-1]) + 1e-9)
    assert abs(local_best[-1]) < 10 * (abs(spark_best[-1]) + 1e-9)


def test_local_backend_rank_ic_fitness_metric_runs_end_to_end(spark, tmp_path):
    # fitness_metric="rank_ic" is the metric actually used for the paper's real runs - make sure
    # the local backend's rank_ic branch (_monthly_ic_from_dataframe on a pandas predictions
    # frame, no .toPandas() round-trip needed) works end-to-end too, not just the rmse default.
    train_df = _fold_frame(spark, n_rows=30, seed=7).cache()
    eval_df = _fold_frame(spark, n_rows=10, seed=99).cache()
    train_df.count()
    eval_df.count()

    ga, tracker = _run_ga(spark, train_df, eval_df, tmp_path / "local_rank_ic", fit_backend="local")
    ga.fitness_metric = "rank_ic"  # cheap override; GAConfig plumbing itself is exercised in algorithms.py
    # Re-run a single fresh individual's fitness evaluation directly with fitness_metric="rank_ic"
    # to confirm the rank_ic branch of the local path doesn't raise and returns a finite float.
    individual = ga.features[0]
    fitness = GeneticAlgorithm1.evaluate_fitness_static(
        individual, "Finance", ga.gapreprocessing, ga.training_cache, ga.testing_cache,
        ga.prediction_cache, gen=0, grammar=ga.grammar, fold_seed=ga.fold_seed,
        gbt_max_iter=ga.gbt_max_iter, fitness_metric="rank_ic", fit_backend="local",
    )
    assert isinstance(fitness, float)
    assert fitness == fitness  # not NaN
