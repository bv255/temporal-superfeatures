"""
Ported from `research/GA_test.ipynb` cell 23 — the per-fold search execution (`build_grammar`,
`discover_folds`, `evaluate_baseline_for_fold`, `_leaf_features`, `_evaluate_on_true_test`,
`_run_random_search_baseline_c`, `run_ga_for_fold`) - the algorithms actually run for one fold
(the real GA plus baselines A/B/C), binding a grammar to `ga.engine.GeneticAlgorithm1`. See
docs/RESTRUCTURING_TODO.md / the port plan. Extracted mechanically from the notebook cell
source, then edited to replace the notebook driver-cell module-level constants
(`GA_OUTPUT_DIR`, `TARGET_POPULATION_SIZE`, `RUN_BASELINE_C`, and `run_ga_for_fold`'s inline
`TOURNAMENT_SIZE`/mutation-config locals) with an explicit `GAConfig` parameter - the same
technique `config.py`'s module docstring describes for `PipelineConfig`, applied to the GA
driver's own `test`/`test_v2` axes (confirmed by diffing GA_test.ipynb's cell 23 against
GA_test_v2.ipynb's directly - see GAConfig's own docstring for the exact 3-way diff, including
a real latent bug this fixes: `discover_folds`'s `fold_dir_hdfs` was hardcoded to a literal
`"walk_forward_test/..."` string independent of whatever `base_local_dir` was passed). Every
other line is untouched.

The post-hoc, whole-run summary functions this module's cell originally also defined
(`build_final_test_summary`/`build_pairwise_comparisons`/`build_winner_composition`, cells
27/28/29) now live in `analysis/summary.py` instead - they operate on a completed list of
`fold_results` after every fold is done, a distinct concern from executing one fold's search.
"""

import os
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from xgboost import XGBRegressor
import numpy as np
import gc

from ..data.panel import GAPreprocessing
from ..caching import DataCache, LocalDataCache
from ..reporting.history import ResultsTracker
from .engine import GeneticAlgorithm1
from ..genome.grammar import ExpressionGrammar
from .seeding import derive_seed
from .checkpoint import compute_fingerprint, load_checkpoint

from ..evaluation.metrics import _count_leaf_features, compute_monthly_ic, summarize_ic, _random_individual
from ..config import GAConfig
from ..operators.temporal import temporal_ops as _temporal_ops

import glob
import json
import random
import shutil
import time
from datetime import datetime, timezone


def build_grammar(config: "GAConfig") -> ExpressionGrammar:
    """
    Builds the ExpressionGrammar (representation/mutation/crossover + temporal wrap/unwrap,
    see grammar.py) from a GAConfig - the integration point TEMPORAL_SUBTREE_OPERATORS_PROMPT.md
    section 7 calls for, keeping engine.py itself config-agnostic.
    """
    ops = _temporal_ops(config.temporal_lag_periods, config.temporal_window_sizes) if config.enable_temporal_operators else []
    return ExpressionGrammar(
        temporal_ops=ops,
        max_features=config.max_features,
        max_nesting=config.max_nesting,
        max_temporal_depth=config.max_temporal_depth,
        temporal_wrap_rate=config.temporal_wrap_rate,
        temporal_unwrap_rate=config.temporal_unwrap_rate,
        report_date_col=config.temporal_report_date_col,
        entity_col="fsym_id",
    )


def discover_folds(config: "GAConfig"):
    """
    Enumerate walk-forward folds from fold_metadata.json (local disk) rather than re-deriving
    fold boundaries - this is the authoritative source PreProcessing_test.ipynb's run_fold()
    already writes per fold. Returns a list of dicts, one per fold.
    """
    folds = []
    for category in ["development", "final_test"]:
        for metadata_path in sorted(glob.glob(f"{config.walk_forward_namespace}/{category}/fold_*/fold_metadata.json")):
            fold_dir_local = os.path.dirname(metadata_path)
            fold_name = os.path.basename(fold_dir_local)
            with open(metadata_path) as f:
                metadata = json.load(f)
            folds.append({
                "category": category,
                "fold_name": fold_name,
                "fold_dir_local": fold_dir_local,
                "fold_dir_hdfs": config.fold_dir_hdfs(category, fold_name),
                "eval_label": metadata["eval_label"],
                "eval_year": metadata["eval_year"],
                "metadata": metadata,
            })
    return folds


def evaluate_baseline_for_fold(gapreprocessing: GAPreprocessing, save_path: str = None, fold_seed: int = None) -> float:
    """
    Baseline RMSE using ONLY prev_month_return/prev_month_sector_return (no super-feature) -
    fits directly off gapreprocessing.train_df/true_eval_df instead of reusing cell 19's
    evaluate_and_save_baseline(), which hardcodes a specific feature name ('ff_pbk_tang') to
    pull cached alignment data from a training_cache/testing_cache. That's not safe here: each
    fold has its own independently-selected feature set (PreProcessing_test's per-fold
    ConsensusFeatureSelector), so that feature may not exist in a given fold's data or population
    at all. This avoids any per-feature cache dependency entirely. Same GBT config (maxIter=10)
    as the rest of the codebase.

    Uses true_eval_df (not eval_df) deliberately: for final-test folds eval_df is the inner
    validation split (used to drive the GA's own selection), while true_eval_df is always the
    eval_label file's own content (test.csv for final-test folds) - the baseline here needs to
    be comparable to the GA's true held-out RMSE, not its validation-time fitness. For
    development folds the two are the same object, so this is a no-op change there.
    """
    select_cols = [
        col('fsym').alias('fsym_id'), col('target_date').alias('sector_return_date'),
        'prev_month_return', 'prev_month_sector_return', col('monthly_return').alias('label'),
    ]
    train = gapreprocessing.train_df.select(*select_cols).dropna()
    eval_ = gapreprocessing.true_eval_df.select(*select_cols).dropna()

    assembler = VectorAssembler(inputCols=["prev_month_return", "prev_month_sector_return"], outputCol="features", handleInvalid="skip")
    gbt_seed = derive_seed(fold_seed, "evaluate_baseline_for_fold")
    gbt = GBTRegressor(featuresCol="features", labelCol="label", maxIter=10, seed=gbt_seed)
    pipeline = Pipeline(stages=[assembler, gbt])

    model = pipeline.fit(train)
    predictions = model.transform(eval_)

    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)

    if save_path:
        predictions_pd = predictions.select("fsym_id", "sector_return_date", "label", "prediction").toPandas()
        predictions_pd.to_csv(save_path, index=False)

    del train, eval_, predictions, model
    return rmse


def evaluate_baseline_for_fold_local(gapreprocessing: GAPreprocessing, save_path: str = None, fold_seed: int = None, gbt_max_iter: int = 10) -> float:
    """
    Pandas/XGBoost sibling of `evaluate_baseline_for_fold`, for `GAConfig.fit_backend="local"` -
    same "prev_month_return/prev_month_sector_return only, no super-feature" baseline A, fit
    directly off `gapreprocessing.train_df`/`true_eval_df` (never routes through
    `grammar.evaluate`/a leaf cache, same as the Spark version). See
    `evaluate_fitness_static`'s docstring for the fit_backend rationale.
    """
    select_cols = ["fsym", "target_date", "prev_month_return", "prev_month_sector_return", "monthly_return"]
    train = gapreprocessing.train_df.select(*select_cols).toPandas().rename(
        columns={"fsym": "fsym_id", "target_date": "sector_return_date", "monthly_return": "label"}
    ).dropna()
    eval_ = gapreprocessing.true_eval_df.select(*select_cols).toPandas().rename(
        columns={"fsym": "fsym_id", "target_date": "sector_return_date", "monthly_return": "label"}
    ).dropna()

    feature_cols = ["prev_month_return", "prev_month_sector_return"]
    gbt_seed = derive_seed(fold_seed, "evaluate_baseline_for_fold")
    model = XGBRegressor(n_estimators=gbt_max_iter, max_depth=5, learning_rate=0.1, random_state=gbt_seed, n_jobs=1)
    model.fit(train[feature_cols].to_numpy(), train["label"].to_numpy())
    predictions_pd = eval_[["fsym_id", "sector_return_date"]].assign(
        label=eval_["label"].to_numpy(), prediction=model.predict(eval_[feature_cols].to_numpy()),
    )
    rmse = float(np.sqrt(np.mean((predictions_pd["label"] - predictions_pd["prediction"]) ** 2)))

    if save_path:
        predictions_pd.to_csv(save_path, index=False)

    del train, eval_, predictions_pd, model
    return rmse


def _leaf_features(individual, grammar: ExpressionGrammar = None) -> list:
    """
    Flatten a (possibly nested) super-feature expression tuple down to its unique base feature
    names, dropping operator tokens ('+','-','*','/') and unwrapping any temporal atom down to
    its real base feature(s) (grammar.flatten_to_leaves - without this, a temporal atom's own
    op-name string, e.g. 'lag1', would be mistaken for a real feature column and passed to
    gapreprocessing.get_true_test_frame(), which has no such column). This is only used to know
    which base features need fresh true-test cache entries built for
    evaluate_and_save_predictions() below - defaults to a temporal_ops=[] grammar (identical to
    the original flat-only flatten) when no grammar is supplied.
    """
    grammar = grammar if grammar is not None else ExpressionGrammar(temporal_ops=[])
    return sorted(set(grammar.flatten_to_leaves(individual)))


def _unique_leaf_or_atom_tokens(individual, grammar: ExpressionGrammar) -> list:
    """
    One-level, atom-aware, de-duped view of an individual's leaf/atom tokens - unlike
    _leaf_features (which unwraps atoms down to real base feature names, for cache lookups), a
    temporal atom is kept and classified as itself here (e.g. ('lag1', 'ff_roic') classifies as
    'lag', not as its inner base feature) - see TEMPORAL_SUBTREE_OPERATORS_PROMPT.md section 9.
    """
    tokens = [t for t in grammar.flatten_expression(individual) if grammar.is_leaf_token(t)]
    seen = set()
    unique = []
    for t in tokens:
        key = t if isinstance(t, str) else repr(t)
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


def _evaluate_on_true_test(ga: "GeneticAlgorithm1", gapreprocessing: GAPreprocessing, individual, sector: str, save_path: str) -> float:
    """
    Shared true-held-out-test evaluation path for any individual (the GA winner, baseline B, or
    baseline C's best find) - builds a small fresh DataCache scoped to just this individual's
    leaf features from get_true_test_frame() (test.csv, never touched during the search), reuses
    ga.training_cache as-is (train_df never changes), and calls cell 19's
    evaluate_and_save_predictions unmodified. Returns a plain (positive) RMSE. Factored out of
    the winner-only inline block the fold driver used to have, since baseline B/C now need the
    exact same evaluation path.
    """
    leaf_features = _leaf_features(individual, ga.grammar)
    is_local = ga.fit_backend == "local"
    true_testing_cache = LocalDataCache(maxsize=max(len(leaf_features), 1)) if is_local else DataCache(maxsize=max(len(leaf_features), 1))
    for feat in leaf_features:
        true_test_frame = gapreprocessing.get_true_test_frame(feat)
        if is_local:
            true_test_frame = true_test_frame.toPandas()
        true_testing_cache.store(feat, true_test_frame)

    rmse = -ga.evaluate_and_save_predictions(
        individual, sector=sector, gapreprocessing=gapreprocessing,
        save_path=save_path, training_cache=ga.training_cache, testing_cache=true_testing_cache,
        grammar=ga.grammar, fold_seed=ga.fold_seed, gbt_max_iter=ga.gbt_max_iter,
        fit_backend=ga.fit_backend,
    )

    if not is_local:
        for df in true_testing_cache.cache.values():
            df.unpersist()

    return rmse


def _run_random_search_baseline_c(ga: "GeneticAlgorithm1", generations_run: int, effective_population_size: int, config: "GAConfig"):
    """
    Item 3 (ga_methodology_additions_prompt.md) - matched-evaluation-budget random search.
    Replaces tournament_selection + crossover_deep + mutation with uniform random resampling
    each generation (_random_individual, cell 22), and disables early termination so the full
    generations_run budget always runs - generations_run is the REAL run's actual generation
    count (early termination may have cut it short), not the configured max, so total fitness
    evaluations match exactly what the real search did for this fold.

    Reuses ga's already-fully-warmed caches: every leaf in ga.features was cached during the
    real run's own gen==0 pass. evaluate_fitness_static's gen==0 branch always fetches fresh from
    GAPreprocessing and ignores the cache regardless of what's already stored there, so every
    evaluate_population call below deliberately passes gen=1 (any nonzero value) to force the
    cache-read path - this is what makes the reuse actually skip recomputation rather than
    silently redoing every leaf's gen-0 fetch on top of what the real run already did.

    A fresh GeneticAlgorithm1 is constructed only to reuse its evaluate_population/__init__
    plumbing (cell 19 stays untouched) - its own randomly-sampled self.features/caches from
    __init__ are immediately overridden with the real run's own, so baseline C searches the
    EXACT SAME closed leaf vocabulary ga did for this fold. Constructing a fresh instance without
    this override would re-random.sample a DIFFERENT feature subset whenever a fold has more
    evolvable features than TARGET_POPULATION_SIZE, which would make the comparison unfair - same
    override-after-construct pattern PreProcessing_test.ipynb already uses for
    ConsensusFeatureSelector.TERMINAL_CAP.

    Returns (best_individual, best_fitness) - the single best individual found across every
    round (no elitism machinery needed for a from-scratch random search).
    """
    # mutation_config is passed explicitly (never relied on the class default) because
    # GeneticAlgorithm1.__init__'s default value - ('flat', True, False, 0.4), only 4 elements -
    # is itself broken: __init__ unconditionally does self.min_mutation = mutation_config[4],
    # which needs a 5th element. Every other call site in this codebase (the real run in
    # run_ga_for_fold above) always passes an explicit 5-tuple, so that broken default had never
    # actually been hit until baseline C - the first construction site that has no use for
    # mutation_config at all (baseline C never calls get_mutation_rates/update_population/mutate,
    # replacing that whole step with uniform random resampling below) and so never had a reason
    # to pass one. The actual values here are inert - only the shape (5 elements) matters.
    # Derived from ga.fold_seed (not reused directly) so baseline C's own randomness - both its
    # GeneticAlgorithm1 internals (unused here beyond evaluate_population plumbing) and its
    # per-generation random-expression sampler below - is independently reproducible without
    # sharing an RNG stream with the real run.
    baseline_c_seed = derive_seed(ga.fold_seed, "baseline_c")
    ga_c = GeneticAlgorithm1(
        ga.gapreprocessing, sector=ga.sector, results_tracker=ResultsTracker(), spark=ga.spark,
        mutation_config=('flat', True, False, 0.4, 0.1),
        population_size=effective_population_size, generations=generations_run, num_threads=config.num_threads_baseline_c,
        grammar=ga.grammar, seed=baseline_c_seed, gbt_max_iter=config.gbt_max_iter,
        fitness_metric=config.fitness_metric, fit_backend=config.fit_backend,
    )
    ga_c.features = ga.features
    ga_c.training_cache = ga.training_cache
    ga_c.testing_cache = ga.testing_cache
    ga_c.prediction_cache = ga.prediction_cache

    sampler_rng = random.Random(derive_seed(baseline_c_seed, "sampler"))

    # baseline C samples temporal atoms at the same rate generation 0 does (config.seed_fraction)
    # whenever ga_c.grammar actually has a temporal vocabulary - reuses grammar.wrap() (the same
    # structural-wrap function initialize_population()/mutate() call), so a temporal atom baseline
    # C draws is distributionally the same shape the real search could have produced. On a
    # temporal-off run ga_c.grammar.temporal_ops is empty, so _random_individual's wrap attempt is
    # automatically a no-op there - no separate on/off branch needed here. seed_fraction (not
    # temporal_wrap_rate) is the right rate to reuse: every baseline C individual is a fresh
    # from-scratch draw each generation, never a mutated survivor, so "fraction of freshly-built
    # individuals pre-wrapped" (generation 0's own semantics) is the faithful analogy - not
    # temporal_wrap_rate's per-mutation-event rate, which presumes an individual being repeatedly
    # mutated across generations, a concept baseline C doesn't have.
    best_individual, best_fitness = None, float('-inf')
    for _ in range(generations_run):
        ga_c.population = [
            _random_individual(ga_c.features, max_features=config.max_features, rng=sampler_rng,
                                grammar=ga_c.grammar, temporal_rate=config.seed_fraction)
            for _ in range(effective_population_size)
        ]
        evaluated = ga_c.evaluate_population(gen=1)
        gen_best_individual, gen_best_fitness = max(evaluated, key=lambda x: x[1])
        if gen_best_fitness > best_fitness:
            best_individual, best_fitness = gen_best_individual, gen_best_fitness

    gc.collect()
    return best_individual, best_fitness


def run_ga_for_fold(fold: dict, config: "GAConfig", spark, seed: int = None) -> dict:
    """
    Run the (unmodified) GeneticAlgorithm1 once against a single fold. Fresh GAPreprocessing,
    ResultsTracker, and GeneticAlgorithm1 per fold - no cache/population/state is shared across
    folds. The GA winner's own predictions/leaf-composition/rank-IC (items 1/6/7 of
    ga_methodology_additions_prompt.md) are now computed for EVERY fold, development included -
    originally so a comparison script could derive Delta (GA_v2_v3_handover.md Sec 5) from
    development-fold IC variance. That consumer never materialized: compare_ga_runs.py (the
    REDESIGNED, currently-maintained comparison script - compare_metrics.ipynb is its frozen
    notebook ancestor, not something run directly anymore) scopes itself to final-test folds only
    (see its load_final_test_fold_results docstring: "development folds are for debugging/
    iteration, not reported"), and evaluation_framework.md's block_length-calibration-from-dev-
    data idea (the other originally-intended dev-fold use) is still an unimplemented placeholder
    too (see analysis/significance.py's block_bootstrap_ic_delta docstring). So this data is
    computed but currently has no reader - kept anyway since it's cheap relative to the fold's own
    GA search and a real future use (either of the above) would need it recomputed per fold
    otherwise. --gbt-tree-search (run_ga.py) is the one thing that DOES consume it today, for a
    different purpose (comparing GBT tree counts across dev folds, deliberately final-test-free).
    Baseline B (item 2) and baseline C (item 3, gated by RUN_BASELINE_C) stay final-test-only, per
    the handover doc's Sec 2 scope ("evaluated once per final-test fold") - see CLAUDE.md's
    "Methodology additions" section for the single-variant adaptation notes (this notebook has no
    separate v2 run to diff against).

    seed: base seed for this call - defaults to config.random_seed when omitted (the normal
    run_ga.py path). This fold's actual seed is derived from it via derive_seed(base_seed,
    fold['fold_name']) - independent per fold regardless of processing order, so folds don't
    share one RNG stream within a process - and recorded in the result dict as fold_seed. Callers
    that want to rerun the SAME fold under different seeds (cross-run stability, see
    analysis/stability.py) pass an explicit seed here per call.
    """
    fold_output_dir = f"{config.output_dir}/{fold['category']}/{fold['fold_name']}"
    os.makedirs(fold_output_dir, exist_ok=True)

    print(f"\n===== Fold {fold['fold_name']} ({fold['category']}, eval_year={fold['eval_year']}) =====")
    fold_started_at = datetime.now(timezone.utc).isoformat()
    start_time = time.time()

    base_seed = seed if seed is not None else config.random_seed
    fold_seed = derive_seed(base_seed, fold['fold_name'])

    gapreprocessing = GAPreprocessing(
        spark, fold['fold_dir_hdfs'], fold['fold_dir_local'], fold['eval_label']
    )

    # population_size is capped by how many evolvable features this fold actually has -
    # GeneticAlgorithm1.initialize_population() does random.sample(features, population_size),
    # which raises ValueError if population_size exceeds the available feature count. Computed
    # here (driver-side) rather than inside the class, so GeneticAlgorithm1 itself needs no
    # changes.
    available_features = gapreprocessing.get_feature_list()
    effective_population_size = min(config.target_population_size, len(available_features))
    if effective_population_size < config.target_population_size:
        print(f"run_ga_for_fold: {fold['fold_name']} has only {len(available_features)} evolvable "
              f"features - capping population_size to {effective_population_size} (target was {config.target_population_size})")

    # tournament_selection (cell 19) does random.sample(evaluated_population, tournament_size),
    # which raises "ValueError: Sample larger than population or is negative" if tournament_size
    # exceeds population_size - hit in practice on final_test/fold_01, which only has 3 evolvable
    # features (below the tournament_size=4 default). Capped the same way population_size is
    # above, floor of 1 (a tournament of 1 is a trivial/no-op selection, but still valid).
    effective_tournament_size = min(config.tournament_size, effective_population_size) if effective_population_size > 0 else 1
    if effective_tournament_size < config.tournament_size:
        print(f"run_ga_for_fold: {fold['fold_name']}'s population_size ({effective_population_size}) is below "
              f"tournament_size ({config.tournament_size}) - capping tournament_size to {effective_tournament_size}")

    tracker = ResultsTracker()

    ga = GeneticAlgorithm1(
        gapreprocessing,
        sector=config.sector,
        results_tracker=tracker,
        spark=spark,
        tournament_size=effective_tournament_size,
        mutation_config=config.mutation_config,
        generations=config.generations,
        population_size=effective_population_size,
        num_threads=config.num_threads,
        grammar=build_grammar(config),
        seed=fold_seed,
        seed_fraction=config.seed_fraction,
        admitted_rate_threshold=config.admitted_rate_threshold,
        stagnation_ceiling=config.stagnation_ceiling,
        gbt_max_iter=config.gbt_max_iter,
        fitness_metric=config.fitness_metric,
        fit_backend=config.fit_backend,
    )

    ga.initialize_population()

    # Item 2 (baseline B): capture generation 0's evaluated population - one individual per leaf
    # feature - strictly before any tournament-selection/crossover step, by calling
    # evaluate_population directly. This costs nothing extra: it's exactly what ga.run()'s own
    # first iteration does internally, so every individual lands in ga.prediction_cache here, and
    # when ga.run() evaluates the same (unchanged) gen-0 population itself moments later, every
    # individual is a cache hit and is skipped rather than retrained. ga.best_individual after
    # ga.run() is NOT safe to use for this - elitism only guarantees the best-known individual
    # survives unmutated once found, not that it's from generation 0.
    evaluated_population_gen0 = ga.evaluate_population(gen=0)
    baseline_b_individual = max(evaluated_population_gen0, key=lambda x: x[1])[0]

    # Lightweight per-generation checkpointing (see ga/checkpoint.py / GeneticAlgorithm1.run's
    # checkpoint_path/checkpoint_fingerprint params). The fingerprint gates resume: if this
    # fold's selected_features.txt/fold_metadata.json or the GAConfig differ from what produced
    # an existing checkpoint.json, it's ignored and overwritten by this fresh run instead of
    # being resumed from - a run never silently replays old population/history against changed
    # data or config. restore_from_checkpoint is called AFTER the baseline-B evaluate_population
    # above (not before) - that pass is what warms training_cache/testing_cache for this run's
    # real generation-0 population, a prerequisite restore_from_checkpoint's own docstring
    # explains in more depth.
    checkpoint_fingerprint = compute_fingerprint(fold, config)
    checkpoint_path = f"{fold_output_dir}/checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_path, checkpoint_fingerprint)
    start_generation = 0
    if checkpoint is not None:
        start_generation = ga.restore_from_checkpoint(checkpoint)
        print(f"run_ga_for_fold: resuming {fold['fold_name']} from generation {start_generation} "
              f"(checkpoint fingerprint matched)")

    ga.run(start_generation=start_generation, checkpoint_path=checkpoint_path, checkpoint_fingerprint=checkpoint_fingerprint)

    elapsed_time = time.time() - start_time
    generations_run = len(tracker.best_fitnesses)

    # ga.run() unconditionally writes "fitness_progression.png" at a bare relative path (cell 19
    # is left untouched, so its path isn't fold-aware) - move it into this fold's output dir
    # right after it's produced, rather than editing cell 19 to parameterize the path.
    if os.path.exists("fitness_progression.png"):
        shutil.move("fitness_progression.png", f"{fold_output_dir}/fitness_progression.png")

    tracker.plot_mutation_progression(save_path=f"{fold_output_dir}/mutation_progression.png")
    tracker.plot_diversity_progression(save_path=f"{fold_output_dir}/diversity_progression.png")
    tracker.plot_admitted_rate_progression(save_path=f"{fold_output_dir}/admitted_rate_progression.png")

    # Fitness is -RMSE when config.fitness_metric=="rmse" (default, original behavior) or mean
    # validation rank IC directly when "rank_ic" (see GAConfig.fitness_metric /
    # evaluate_fitness_static's docstring) - GA maximizes fitness either way, so elitism (with
    # elitest_mutation=False, the current config) keeps the single best-ever individual in the
    # population unmutated once found regardless of which metric is active, and best_fitnesses is
    # non-decreasing with its last entry the best fitness seen all run either way. This is scored
    # against gapreprocessing.eval_df, which for final-test folds is now the inner validation.csv
    # split (see GAPreprocessing.has_inner_validation) - it's the GA's own internal search signal,
    # not a claim about true held-out performance, hence the "validation_" naming below. Exactly
    # one of validation_rmse/validation_rank_ic is populated (matching whichever metric actually
    # drove this fold's search) and the other stays None - reporting a "validation_rmse" number
    # when the search was actually scored on rank IC would be actively misleading, not just
    # differently-labeled.
    best_fitness = tracker.best_fitnesses[-1] if tracker.best_fitnesses else None
    validation_rmse = -best_fitness if best_fitness is not None and config.fitness_metric == "rmse" else None
    validation_rank_ic = best_fitness if best_fitness is not None and config.fitness_metric == "rank_ic" else None

    # Baseline RMSE/IC stay None outside final-test folds (baselines are final-test-only, per
    # GA_v2_v3_handover.md Sec 2). The GA winner's own fields just below are populated for every
    # fold - see the docstring above.
    predictions_path = None
    baseline_rmse = None
    true_test_rmse = None
    winner_leaf_count = None
    winner_leaf_classification = None
    winner_temporal_fraction = None
    winner_uses_any_temporal = None
    winner_monthly_ic = None
    winner_mean_ic = None
    winner_ic_ir = None
    baseline_a_predictions_path = None
    baseline_a_monthly_ic = None
    baseline_a_mean_ic = None
    baseline_a_ic_ir = None
    baseline_b_predictions_path = None
    baseline_b_rmse = None
    baseline_b_leaf_count = None
    baseline_b_monthly_ic = None
    baseline_b_mean_ic = None
    baseline_b_ic_ir = None
    baseline_c_individual = None
    baseline_c_predictions_path = None
    baseline_c_rmse = None
    baseline_c_leaf_count = None
    baseline_c_monthly_ic = None
    baseline_c_mean_ic = None
    baseline_c_ic_ir = None

    # ---- GA winner ---- (every fold, not just final-test - see docstring)
    predictions_path = f"{fold_output_dir}/predictions.csv"
    true_test_rmse = _evaluate_on_true_test(ga, gapreprocessing, ga.best_individual, config.sector, predictions_path)
    print(f"Fold {fold['fold_name']}: true held-out test RMSE (never seen during the search) = {true_test_rmse}")

    winner_leaf_count = ga.grammar.count_features_through_atoms(ga.best_individual)
    winner_leaf_classification = [
        ga.grammar.classify_leaf_or_atom(t) for t in _unique_leaf_or_atom_tokens(ga.best_individual, ga.grammar)
    ]
    winner_uses_any_temporal = any(c != 'raw' for c in winner_leaf_classification)
    # len([...]) instead of sum(1 for ...) - plain sum() is shadowed by
    # pyspark.sql.functions.sum in this kernel (see cell 22's _count_leaf_features note).
    winner_temporal_fraction = (
        len([c for c in winner_leaf_classification if c != 'raw']) / len(winner_leaf_classification)
        if winner_leaf_classification else float('nan')
    )
    winner_monthly_ic = compute_monthly_ic(predictions_path)
    winner_mean_ic, winner_ic_ir = summarize_ic(winner_monthly_ic)

    if fold['category'] == 'final_test':
        # ---- Baseline A (trivial: prev_month_return/prev_month_sector_return only) ----
        baseline_a_predictions_path = f"{fold_output_dir}/baseline_predictions.csv"
        baseline_a_fn = evaluate_baseline_for_fold_local if config.fit_backend == "local" else evaluate_baseline_for_fold
        baseline_rmse = baseline_a_fn(gapreprocessing, save_path=baseline_a_predictions_path, fold_seed=fold_seed)
        print(f"Fold {fold['fold_name']}: baseline A RMSE (prev_month_return/prev_month_sector_return only, "
              f"same true held-out test data) = {baseline_rmse}")
        baseline_a_monthly_ic = compute_monthly_ic(baseline_a_predictions_path)
        baseline_a_mean_ic, baseline_a_ic_ir = summarize_ic(baseline_a_monthly_ic)

        # ---- Baseline B (best single feature at generation 0) ----
        baseline_b_predictions_path = f"{fold_output_dir}/baseline_b_predictions.csv"
        baseline_b_rmse = _evaluate_on_true_test(ga, gapreprocessing, baseline_b_individual, config.sector, baseline_b_predictions_path)
        baseline_b_leaf_count = _count_leaf_features(baseline_b_individual)
        print(f"Fold {fold['fold_name']}: baseline B (best gen-0 single feature = {baseline_b_individual!r}) "
              f"true held-out test RMSE = {baseline_b_rmse}")
        baseline_b_monthly_ic = compute_monthly_ic(baseline_b_predictions_path)
        baseline_b_mean_ic, baseline_b_ic_ir = summarize_ic(baseline_b_monthly_ic)

        # ---- Baseline C (matched-budget random search) ----
        if config.run_baseline_c:
            baseline_c_individual, _ = _run_random_search_baseline_c(ga, generations_run, effective_population_size, config)
            baseline_c_predictions_path = f"{fold_output_dir}/baseline_c_predictions.csv"
            baseline_c_rmse = _evaluate_on_true_test(ga, gapreprocessing, baseline_c_individual, config.sector, baseline_c_predictions_path)
            baseline_c_leaf_count = _count_leaf_features(baseline_c_individual)
            print(f"Fold {fold['fold_name']}: baseline C (random search winner = {baseline_c_individual!r}) "
                  f"true held-out test RMSE = {baseline_c_rmse}")
            baseline_c_monthly_ic = compute_monthly_ic(baseline_c_predictions_path)
            baseline_c_mean_ic, baseline_c_ic_ir = summarize_ic(baseline_c_monthly_ic)

    fold_finished_at = datetime.now(timezone.utc).isoformat()

    result = {
        "category": fold['category'],
        "fold_name": fold['fold_name'],
        "eval_label": fold['eval_label'],
        "eval_year": fold['eval_year'],
        # Canonicalized (str(grammar.canonicalize(...)), not the raw ga.best_individual) so two
        # spellings of the same value (e.g. stacked lags vs. their collapsed form) compare equal
        # across seeded reruns of the same fold - see analysis/stability.py's champion_stability.
        "winning_expression": str(ga.grammar.canonicalize(ga.best_individual)),
        "validation_rmse": validation_rmse,
        "validation_rank_ic": validation_rank_ic,
        "fitness_metric": config.fitness_metric,
        "true_test_rmse": true_test_rmse,
        "baseline_rmse": baseline_rmse,
        "predictions_path": predictions_path,
        "generations_run": generations_run,
        "effective_population_size": effective_population_size,
        "elapsed_seconds": elapsed_time,
        "best_fitnesses": tracker.best_fitnesses,
        "average_fitnesses": tracker.average_fitnesses,
        "worst_fitnesses": tracker.worst_fitnesses,
        "diversity_history": tracker.diversity,
        # ---- Provenance ----
        # Same value as this fold's checkpoint.json (ga/checkpoint.py's compute_fingerprint) -
        # lets a future run_ga.py invocation recognize "this fold's fold_result.json was produced
        # under an identical config + fold data" and skip recomputing it entirely, rather than
        # just checking whether the file exists (see run_ga.py's _load_or_run_fold).
        "fingerprint": checkpoint_fingerprint,
        "fold_seed": fold_seed,
        "fold_started_at": fold_started_at,
        "fold_finished_at": fold_finished_at,
        "generation_timings": ga.time_taken,
        # ---- Cross-run stability support (analysis/stability.py) ----
        "final_population_top_k": ga.final_ranked_population,
        # ---- Methodology additions (ga_methodology_additions_prompt.md) ----
        "winner_leaf_count": winner_leaf_count,
        "winner_leaf_classification": winner_leaf_classification,
        "winner_temporal_fraction": winner_temporal_fraction,
        "winner_uses_any_temporal": winner_uses_any_temporal,
        "winner_monthly_ic": winner_monthly_ic.to_dict() if winner_monthly_ic is not None else None,
        "winner_mean_ic": winner_mean_ic,
        "winner_ic_ir": winner_ic_ir,
        "baseline_a_predictions_path": baseline_a_predictions_path,
        "baseline_a_monthly_ic": baseline_a_monthly_ic.to_dict() if baseline_a_monthly_ic is not None else None,
        "baseline_a_mean_ic": baseline_a_mean_ic,
        "baseline_a_ic_ir": baseline_a_ic_ir,
        "baseline_b_individual": baseline_b_individual if fold['category'] == 'final_test' else None,
        "baseline_b_predictions_path": baseline_b_predictions_path,
        "baseline_b_rmse": baseline_b_rmse,
        "baseline_b_leaf_count": baseline_b_leaf_count,
        "baseline_b_monthly_ic": baseline_b_monthly_ic.to_dict() if baseline_b_monthly_ic is not None else None,
        "baseline_b_mean_ic": baseline_b_mean_ic,
        "baseline_b_ic_ir": baseline_b_ic_ir,
        "baseline_c_individual": baseline_c_individual,
        "baseline_c_predictions_path": baseline_c_predictions_path,
        "baseline_c_rmse": baseline_c_rmse,
        "baseline_c_leaf_count": baseline_c_leaf_count,
        "baseline_c_monthly_ic": baseline_c_monthly_ic.to_dict() if baseline_c_monthly_ic is not None else None,
        "baseline_c_mean_ic": baseline_c_mean_ic,
        "baseline_c_ic_ir": baseline_c_ic_ir,
    }

    with open(f"{fold_output_dir}/fold_result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Fold {fold['fold_name']} done: fitness_metric={config.fitness_metric}, "
          f"validation_rmse={validation_rmse}, validation_rank_ic={validation_rank_ic}, "
          f"true_test_rmse={true_test_rmse}, generations={generations_run}, elapsed={elapsed_time:.1f}s")

    # Per-fold cleanup - no cache/state carried into the next fold. Baseline C reuses ga's own
    # caches (see _run_random_search_baseline_c), so unpersisting ga's caches here covers it too
    # - nothing baseline-C-specific needs separate cleanup.
    spark.catalog.clearCache()
    for cache_name in ['training_cache', 'testing_cache', 'prediction_cache']:
        if hasattr(ga, cache_name):
            for df in getattr(ga, cache_name).cache.values():
                try:
                    df.unpersist()
                except Exception:
                    pass
    gapreprocessing.train_df.unpersist()
    gapreprocessing.eval_df.unpersist()
    if gapreprocessing.true_eval_df is not gapreprocessing.eval_df:
        gapreprocessing.true_eval_df.unpersist()
    gc.collect()

    return result
