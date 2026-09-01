"""
`GeneticAlgorithm1`, the search core. Originally ported mechanically from
`legacy/notebooks/GA_test.ipynb` cell 19 - see `docs/RESTRUCTURING_TODO.md`, the port plan in
`docs/RESEARCH_STRUCTURE.md`, and `docs/GA_structure.md` (the fidelity checklist this class
reproduces - flat odd-length-tuple representation, no operator precedence surviving crossover,
the dead `cache_size` constructor param, the closed leaf vocabulary, tournament-only selection,
`-RMSE` fitness, the `1e-6` division guard, elitism-without-true-elitism). The notebook lineage is
now legacy (moved to `legacy/notebooks/`, no longer maintained or treated as a frozen parity
reference) - this package is the actively maintained implementation, and its class body is no
longer required to mirror the notebook cell verbatim; see `fit_backend` (added to `GAConfig` and
threaded through `evaluate_fitness_static` below) for the first deliberate divergence.

One further wrinkle: the class body also uses several names that come from cell 21 in the
notebook (`sqrt`/`mean`/`pow` from `pyspark.sql.functions`, `StorageLevel`, `os`/`psutil`/`time`)
- a DRIVER cell that runs AFTER this class's own cell (19) but BEFORE any of its methods are
actually called, since a notebook kernel shares one global namespace across all cells regardless
of definition order. Reproduced verbatim below (`from pyspark.sql.functions import col, pow,
sqrt, mean` intentionally shadows the `pow`/`col` builtins/prior imports exactly as cell 21 does
in the live kernel - Spark's Column-aware `pow` is what the RMSE calculation actually needs).
`monitor_all_memory` additionally uses `driver_host`/`port`/`log_executor_memory_rest`, which
come from a DIFFERENT cell (2) that scrapes the live Spark UI's REST API - genuinely
cluster-diagnostic-only code, not called anywhere in the documented driver flow
(`docs/GA_structure.md` section 8), so it is NOT reproduced here; `monitor_all_memory` is ported
as-is and will raise NameError if ever called, identically to a fresh notebook kernel that ran
cell 19 but not cell 2.
"""

from typing import List, Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark import StorageLevel
from pyspark.sql.functions import col, pow, sqrt, mean
import os
import psutil
import time

from ..data.panel import GAPreprocessing
from ..caching import DataCache, LocalDataCache, PredictionCache
from ..reporting.history import ResultsTracker
from ..genome.grammar import ExpressionGrammar
from ..operators.temporal import apply_temporal
from ..operators.arithmetic import combine_dataframes, winsorize_feature
from ..operators.temporal_local import apply_temporal_local
from ..operators.arithmetic_local import combine_dataframes_local, winsorize_feature_local
from ..evaluation.metrics import _monthly_ic_from_dataframe
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from .seeding import derive_seed
from .checkpoint import save_checkpoint
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.sql import Row
import concurrent.futures
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, IntegerType, DateType
from itertools import islice
import threading
from concurrent.futures import ThreadPoolExecutor
from pyspark.sql.functions import max as ps_max, min as ps_min
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from pyspark import SparkContext, SparkConf
import gc
import random
from functools import partial
from pyspark.sql.functions import col
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline



def _lists_to_tuples(obj):
    """
    Recursively reconstruct tuples from the nested lists a checkpoint round-trips them through
    as (json.dump auto-converts tuples to arrays on the way out; there is no symmetric auto-
    convert-back on the way in). Used both for individuals (grammar methods check
    isinstance(x, tuple) explicitly - a plain list would break is_atom/flatten_expression/etc.)
    and for random.Random.getstate()'s nested-tuple structure.
    """
    if isinstance(obj, list):
        return tuple(_lists_to_tuples(x) for x in obj)
    return obj


class GeneticAlgorithm1:
    def __init__(self, gapreprocessing: GAPreprocessing, sector: str, results_tracker: ResultsTracker, spark: SparkSession, tournament_size: int = 3, mutation_config: Tuple[str,bool,bool,float,float] = ('flat',True,False,0.4), generations: int = 5, population_size: int = 10, num_threads: int = 1, cache_size: int=30, grammar: ExpressionGrammar = None, seed: int = None, seed_fraction: float = 0.0, top_k: int = 5, admitted_rate_threshold: float = 0.5, stagnation_ceiling: int = 50, gbt_max_iter: int = 10, fitness_metric: str = "rmse", fit_backend: str = "spark"):
        """
        Initialize the Genetic Algorithm.

        Args:
            gapreprocessing (GAPreprocessing): The preprocessing object containing data and methods.
            sector (str): The sector to which the features belong.
            results_tracker (ResultsTracker): The tracker for monitoring results such as fitness over generations.
            tournament_size (int): The size of the tournament for selection. Default is 3.
            mutation_rate (float): The probability of mutation occurring in an individual. Default is 0.1 (10%).
            generations (int): The number of generations to run. Default is 5.
            grammar (ExpressionGrammar): Representation/mutation/crossover logic (see grammar.py).
                Defaults to a temporal_ops=[] grammar, which reproduces this class's original
                (pre-temporal-operator) behavior exactly.
            seed (int): Seeds self.rng (a private random.Random instance) so every random draw
                this class makes - population init, selection, crossover/mutation coin flips,
                and (via self.fold_seed below) the deterministic per-individual GBT seed derived
                in evaluate_fitness_static - is reproducible. None (default) makes self.rng the
                bare `random` module itself (not an independent random.Random() instance) -
                module-level random.sample/choice/random/getstate/etc. are just bound methods of
                one global default Random instance, so this preserves this class's *exact*
                original behavior (reading and advancing the shared global RNG state, the same
                one a test's own random.seed(...) reset controls) when no seed is given, rather
                than silently switching to an independent, differently-seeded stream.
            seed_fraction (float): Fraction of generation-0 individuals initialize_population()
                pre-wraps in a random temporal operator (gafactor's seeded-fraction bootstrap -
                mutation-only introduction of temporal atoms adopts too slowly otherwise).
                Default 0.0 preserves this class's original zero-seeding behavior; only relevant
                when grammar.temporal_ops is non-empty.
            admitted_rate_threshold (float): early-termination "converged" signal - see run()'s
                early-termination block. update_population()'s dedup policy (accept()) forces a
                mutation retry on any crossover offspring/elite copy whose canonical form
                duplicates one already accepted this generation, which keeps raw population
                diversity (unique canonical individuals) pinned near population_size almost by
                construction - even a population dominated by one building block still looks
                fully diverse by that count, since the forced retry mutates some other leaf/
                operator into something syntactically novel. admitted_rate (the fraction of a
                generation's individuals that needed >=1 such retry) is the actual "has the
                search run out of new things to try" signal under this policy; termination
                requires its 10-generation average to reach this threshold, not raw diversity.
            stagnation_ceiling (int): unconditional fitness-only backstop - terminate once the
                best fitness has been unchanged for this many generations, regardless of
                admitted_rate, so a fold that stays behaviorally diverse (low admitted_rate)
                without ever improving still has a ceiling instead of grinding to `generations`.
        """
        self.fit_backend = fit_backend
        self.grammar = grammar if grammar is not None else ExpressionGrammar(temporal_ops=[])
        self.gapreprocessing = gapreprocessing
        self.sector = sector
        self.results_tracker = results_tracker
        self.rng = random.Random(seed) if seed is not None else random
        self.fold_seed = seed
        self.seed_fraction = seed_fraction
        self.features = self.rng.sample(self.gapreprocessing.get_feature_list(), population_size)
        self.population = []
        if tournament_size > population_size:
            raise ValueError(
                f"tournament_size ({tournament_size}) cannot exceed population_size ({population_size})"
            )
        self.tournament_size = tournament_size
        self.mutation_method = mutation_config[0]
        self.crossover_mutation = mutation_config[1]
        self.elitest_mutation = mutation_config[2]
        self.max_mutation = mutation_config[3]
        self.min_mutation = mutation_config[4]
        self.generations = generations
        self.best_individual = 0
        self.num_threads = num_threads
        self.population_size = population_size
        cache_cls = LocalDataCache if fit_backend == "local" else DataCache
        self.training_cache = cache_cls(maxsize=population_size)
        self.testing_cache = cache_cls(maxsize=population_size)
        self.prediction_cache = PredictionCache(maxsize=population_size*10)
        self.spark = spark
        self.time_taken = []
        self.log_list = []
        self.termination_proportion = 0.25
        self.admitted_rate_threshold = admitted_rate_threshold
        self.stagnation_ceiling = stagnation_ceiling
        self.top_k = top_k
        self.gbt_max_iter = gbt_max_iter
        self.fitness_metric = fitness_metric
        self.final_ranked_population = []

    def initialize_population(self) -> None:
        """
        Initialize the population with all individual features. If self.grammar has a
        non-empty temporal_ops vocabulary and self.seed_fraction > 0, pre-wrap that fraction of
        generation-0 individuals in a random temporal operator (gafactor's seeded-fraction
        bootstrap) instead of leaving temporal-atom introduction entirely to mutation's
        wrap/unwrap coin flip, which measurably adopts too slowly to bootstrap from zero. Reuses
        grammar.wrap() verbatim - it already picks a random op, avoids redundant wraps,
        canonicalizes, and validates - falling back to the unwrapped feature if wrap() finds
        nothing valid within its attempt budget.
        """
        self.population = [feature for feature in self.features]
        if self.grammar.temporal_ops and self.seed_fraction > 0:
            num_to_seed = round(len(self.population) * self.seed_fraction)
            indices = self.rng.sample(range(len(self.population)), num_to_seed)
            for i in indices:
                wrapped = self.grammar.wrap(self.population[i], self.rng)
                if wrapped is not None:
                    self.population[i] = wrapped
    
    @staticmethod
    def evaluate_fitness_static(individual, sector: str, gapreprocessing: GAPreprocessing, training_cache, testing_cache, prediction_cache: PredictionCache, gen: int, grammar: ExpressionGrammar = None, fold_seed: int = None, gbt_max_iter: int = 10, fitness_metric: str = "rmse", fit_backend: str = "spark") -> float:
        """
        Evaluate the fitness of an individual that can be a complex feature expression.

        fold_seed derives this evaluation's GBT seed (see derive_seed in ga/seeding.py) as a
        pure function of (fold_seed, gen, key) rather than a shared-mutable RNG draw, so it's
        safe to call from multiple threads concurrently (evaluate_population runs this inside a
        ThreadPoolExecutor) regardless of which thread finishes first.

        fitness_metric selects what this individual is actually SCORED on, not what the GBT
        model is trained to predict - it always fits next-month returns (labelCol "label")
        regardless. "rmse" (default) is the original behavior: fitness = -RMSE over the fold's
        validation rows, pooled across months (GA minimizes RMSE via maximizing -RMSE).
        "rank_ic" instead computes Spearman rank IC separately within each validation month and
        averages those - the research question's actual estimand - returned directly (GA
        maximizes mean IC, no negation needed since higher IC is already "better"). prediction_cache
        always stores the RAW metric value (RMSE or mean IC) regardless of which one is active -
        the sign flip for "rmse" is applied once, at this function's own return statement, so a
        cache hit and a fresh computation are scored identically either way.

        fit_backend selects the model-fitting implementation - "spark" (default,
        pyspark.ml.regression.GBTRegressor, on the cluster) or "local" (xgboost.XGBRegressor,
        in-process pandas/numpy, no per-individual Spark scheduling overhead and releases the
        GIL during fitting so evaluate_population's ThreadPoolExecutor gets real parallelism -
        see GAConfig.fit_backend's docstring for why this exists and why it's NOT expected to
        numerically reproduce the spark backend). training_cache/
        testing_cache hold Spark DataFrames for "spark" (DataCache) or pandas DataFrames for
        "local" (LocalDataCache) - which one the caller constructed already matches fit_backend,
        this function just uses whichever cache type it's handed.
        """
        grammar = grammar if grammar is not None else ExpressionGrammar(temporal_ops=[])
        is_local = fit_backend == "local"

        def leaf_fn(expression):
            hit_train, training_data = training_cache.get(expression)
            hit_test, testing_data = testing_cache.get(expression)
            if hit_train and hit_test:
                return training_data, testing_data
            if gen != 0:
                raise ValueError(f"Missing cached base feature: {expression}")
            training_data, testing_data = gapreprocessing.generate_training_testing_data(expression, sector)
            if is_local:
                training_data, testing_data = training_data.toPandas(), testing_data.toPandas()
            training_cache.store(expression, training_data)
            testing_cache.store(expression, testing_data)
            return training_data, testing_data

        def temporal_fn(df, op):
            fn = apply_temporal_local if is_local else apply_temporal
            return fn(df, op, date_col=grammar.report_date_col, entity_col=grammar.entity_col)

        combine_fn = combine_dataframes_local if is_local else combine_dataframes

        # Create cache key - canonical form, so structurally-identical individuals with
        # different spellings (e.g. two chained lag operators vs. their collapsed form) share one
        # cache entry instead of being treated as distinct.
        key = str(grammar.canonicalize(individual))
        hit, cached_metric = prediction_cache.get(key)

        # If a cached metric exists (RMSE or mean rank IC, whichever fitness_metric is active for
        # this whole run - see the docstring above), use it. Else create new data and cache it.
        if hit:
            metric = cached_metric

        else:
            # Base feature datasets (raw, pre-winsorization) are cached per-leaf inside leaf_fn
            # itself, keyed by leaf name - not here by the top-level individual's key - so that a
            # leaf reached only via a wrapped/composite gen-0 individual (e.g. a seed_fraction
            # temporal wrap) still gets cached under its own bare name for later generations to
            # find, and a later composite expression recombining this leaf sees its true
            # pre-winsorization value rather than a clipped one.
            training_data, testing_data = grammar.evaluate(individual, leaf_fn, combine_fn, temporal_fn)

            # testing_data was evaluated over train_df UNION eval_df (see
            # GAPreprocessing.generate_training_testing_data's docstring - gives live temporal
            # atoms real preceding history at the eval split's boundary) - trim back to genuine
            # eval rows now that lag/rolling computation is done, before any scoring below sees it.
            if is_local:
                testing_data = testing_data[testing_data["sector_return_date"] >= gapreprocessing.eval_start_date]
            else:
                testing_data = testing_data.filter(col("sector_return_date") >= gapreprocessing.eval_start_date)

            # Deterministic per-(fold, generation, individual) GBT seed - shared by both backends.
            gbt_seed = derive_seed(fold_seed, gen, key)

            if is_local:
                # Winsorize (train-only bounds, no test-set leakage) - pandas equivalent, see
                # arithmetic_local.py.
                training_data, testing_data = winsorize_feature_local(training_data, testing_data)

                feature_cols = ["prev_month_return", "prev_month_sector_return", "feature"]
                train_clean = training_data.dropna(subset=feature_cols + ["target"])
                test_clean = testing_data.dropna(subset=feature_cols + ["target"])

                # max_depth=5/learning_rate=0.1 match MLlib GBTRegressor's own defaults
                # (maxDepth=5, stepSize=0.1); XGBoost's objective="reg:squarederror"/
                # subsample=1.0 defaults already match MLlib's lossType="squared"/
                # subsamplingRate=1.0. XGBRegressor chosen over sklearn's own
                # GradientBoostingRegressor for two reasons: (1) XGBoost releases the GIL during
                # fitting, so the ThreadPoolExecutor evaluate_population already runs this inside
                # gets genuine multi-core parallelism - sklearn's classic GBM barely does; (2)
                # XGBoost's default grow_policy="depthwise" matches MLlib GBTRegressor's own
                # depth-limited tree growth (and its histogram/approximate-quantile split finding
                # is structurally closer to MLlib's than sklearn's exact-sort split finding is) -
                # neither point makes this numerically reproduce the spark backend (different
                # regularization terms in XGBoost's split-gain formula, different RNG internals),
                # see GAConfig.fit_backend's docstring - just a closer structural cousin.
                # n_jobs=1: XGBoost defaults to using every available core internally per fit -
                # with evaluate_population's own ThreadPoolExecutor already running num_threads
                # individuals concurrently, that's thread oversubscription (each of the 8 threads
                # spawning its own multi-core fit, dozens of OS threads fighting over 4 physical
                # cores) rather than 8 clean single-threaded fits sharing the machine. Parallelism
                # belongs at the outer (per-individual) level, not inside each fit.
                model = XGBRegressor(
                    n_estimators=gbt_max_iter, max_depth=5, learning_rate=0.1, random_state=gbt_seed,
                    n_jobs=1,
                )
                model.fit(train_clean[feature_cols].to_numpy(), train_clean["target"].to_numpy())
                predictions = test_clean[["sector_return_date"]].assign(
                    label=test_clean["target"].to_numpy(),
                    prediction=model.predict(test_clean[feature_cols].to_numpy()),
                )

                if fitness_metric == "rank_ic":
                    monthly_ic = _monthly_ic_from_dataframe(predictions)
                    metric = float(monthly_ic.mean()) if len(monthly_ic) else float('-inf')
                else:
                    metric = float(np.sqrt(np.mean((predictions["label"] - predictions["prediction"]) ** 2)))
                prediction_cache.store(key, metric)
                del model, training_data, testing_data, train_clean, test_clean, predictions

            else:
                # Winsorize the evolved "feature" column before the GBT ever sees it - bounds fit
                # on training data only (no test-set leakage), applied to both train/test. A
                # single exploding value from a division or growth computation would otherwise
                # flow straight into model fitting uncapped.
                training_data, testing_data = winsorize_feature(training_data, testing_data)

                # Prepare the data for MLlib - fsym_id/sector_return_date kept through the select
                # (not dropped, as before) so the repartition below can hash-partition on them
                # deterministically. The previous bare .repartition(8) round-robin-assigns rows
                # from a random start position - GBT samples split thresholds per partition, so
                # that made the fitted model nondeterministic on rerun independent of any RNG seed.
                training_data = training_data.select(col("fsym_id"), col("sector_return_date"), col("prev_month_return"), col("prev_month_sector_return"), col("feature"), col("target").alias("label"))
                testing_data = testing_data.select(col("fsym_id"), col("sector_return_date"), col("prev_month_return"), col("prev_month_sector_return"), col("feature"), col("target").alias("label"))

                # Create vec_train and temporarily store
                assembler = VectorAssembler(inputCols=["prev_month_return", "prev_month_sector_return", "feature"], outputCol="features", handleInvalid="skip")
                vec_train = assembler.transform(training_data).select("fsym_id", "sector_return_date", "features", "label").repartition(8, "fsym_id", "sector_return_date").select("features", "label").persist(StorageLevel.MEMORY_AND_DISK)
                vec_train.count()

                # Train GBT model
                gbt = GBTRegressor(featuresCol="features", labelCol="label", maxIter=gbt_max_iter, seed=gbt_seed)
                model = gbt.fit(vec_train)

                # Extract predictions - sector_return_date is kept all the way through (not
                # dropped before persist, the way vec_train's own select drops it) specifically
                # so the rank_ic branch below can group by validation month; harmless to always
                # keep it, so there's one vec_test pipeline shape regardless of fitness_metric.
                vec_test = assembler.transform(testing_data).select("fsym_id", "sector_return_date", "features", "label").repartition(8, "fsym_id", "sector_return_date").select("sector_return_date", "features", "label").persist(StorageLevel.MEMORY_AND_DISK)
                vec_test.count()
                raw_predictions = model.transform(vec_test)
                predictions = raw_predictions.select("sector_return_date", "label", "prediction")

                if fitness_metric == "rank_ic":
                    # Mean monthly Spearman rank IC over the validation period - the research
                    # question's actual estimand (see GAConfig.fitness_metric's docstring). Small
                    # per-individual result set (one fold's own validation rows), so collecting to
                    # the driver and reusing _monthly_ic_from_dataframe (the exact same logic
                    # compute_monthly_ic uses for the final winner's own reported IC) is simpler
                    # and more consistent than hand-rolling a Spark-native per-month Spearman
                    # aggregate.
                    predictions_pd = predictions.toPandas()
                    monthly_ic = _monthly_ic_from_dataframe(predictions_pd)
                    metric = float(monthly_ic.mean()) if len(monthly_ic) else float('-inf')
                else:
                    # Calculate RMSE
                    metric = predictions.select(
                        pow(col("label") - col("prediction"), 2).alias("squared_error")
                    ).agg(
                        sqrt(mean("squared_error")).alias("rmse")
                    ).first()["rmse"]
                prediction_cache.store(key, metric)

                # Tidy variables
                vec_train.unpersist()
                vec_test.unpersist()
                del model, raw_predictions, training_data, testing_data, vec_train, vec_test

        # rmse is minimized -> maximize -metric. rank_ic is maximized directly -> no negation.
        return metric if fitness_metric == "rank_ic" else -metric
    
    def evaluate_population(self, gen: int) -> List[Tuple[str, float]]:
        """
        Evaluate the fitness of the entire population in parallel using concurrent.futures,
        with progress tracking.
        """
        total_individuals = len(self.population)

        # Progress tracking
        completed = 0
        progress_lock = threading.Lock()

        def evaluate_and_track_progress(individual, index):
            nonlocal completed
            fitness = self.evaluate_fitness_static(individual, self.sector, self.gapreprocessing, self.training_cache, self.testing_cache, self.prediction_cache, gen, self.grammar, fold_seed=self.fold_seed, gbt_max_iter=self.gbt_max_iter, fitness_metric=self.fitness_metric, fit_backend=self.fit_backend)
            
            with progress_lock:
                completed += 1
                if completed % max(1, total_individuals // 5) == 0 or completed == total_individuals:
                    print(f"Evaluated {completed}/{total_individuals} individuals: ({individual})")
                    
            del index
            return (individual, fitness)

        # Evaluate individuals with variable number of parallel threads
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            evaluated_population = list(executor.map(
                evaluate_and_track_progress,
                self.population,
                range(total_individuals)
            ))
        # Clean up temporary variables
        del total_individuals, completed, progress_lock #,num_threads

        return evaluated_population
    
    def tournament_selection(self, evaluated_population: List[Tuple[str, float]], num_selections: int) -> List[Tuple[str, None]]:
        """
        Perform tournament selection on the evaluated population to select individuals for crossover.
        """
        selected_individuals = []

        for _ in range(num_selections):
            # Randomly choose individuals for the tournament
            tournament = self.rng.sample(evaluated_population, self.tournament_size)
            # Select the individual with the best fitness
            winner = max(tournament, key=lambda x: x[1])
            selected_individuals.append(winner[0])  # Keep the feature, disregard the operator for now
        
        # Clean up the tournament and evaluated population variables
        del tournament, evaluated_population, winner

        return selected_individuals

    def roulette_selection(self, evaluated_population: List[Tuple[str, float]], num_selections: int) -> List[Tuple[str, None]]:
        """
        Perform roulette selection on the evaluated population to select individuals for crossover.
        """
        # Extract the fitness values from the evaluated population
        fitness_values = [individual[1] for individual in evaluated_population]

        # Calculate the total fitness of the population
        total_fitness = 0
        
        for i in range(len(fitness_values)):
            total_fitness -= fitness_values[i]

        # Handle edge case where total fitness is zero
        if total_fitness == 0:
            probabilities = [1 / len(evaluated_population)] * len(evaluated_population)
        else:
            # Normalize fitness values to create a probability distribution
            probabilities = [-fitness / total_fitness for fitness in fitness_values]

        selected_individuals = []

        for _ in range(num_selections):
            # Select an individual based on the probability distribution
            selected = self.rng.choices(evaluated_population, weights=probabilities, k=1)[0]
            selected_individuals.append(selected[0])  # Keep the feature, disregard the fitness value

        return selected_individuals
    
    def crossover(self, parent1: Tuple, parent2: Tuple) -> Tuple:
        """
        Perform crossover between two parents to produce offspring.
        The crossover will occur within sub-expressions, swapping components.
        """
        offspring = self.crossover_deep(parent1, parent2)
        
        # Clean up parents after crossover
        del parent1, parent2

        return offspring
    
    def crossover_deep(self, expr1: Tuple, expr2: Tuple) -> Tuple:
        """
        Perform a deep crossover within sub-expressions - delegates to self.grammar
        (see grammar.py's ExpressionGrammar.crossover_deep), which reproduces this exact logic
        (including atom-aware fixes for temporal wraps) with a temporal_ops=[] grammar.
        """
        return self.grammar.crossover_deep(expr1, expr2, rng=self.rng)

    def mutate_feature(self, individual):
        """
        Mutate the feature in the individual - delegates to self.grammar.
        """
        return self.grammar.mutate_feature(individual, self.features, rng=self.rng)

    def mutate_operator(self, individual):
        """
        Mutate the operator in the individual - delegates to self.grammar.
        """
        return self.grammar.mutate_operator(individual, rng=self.rng)

    def mutate(self, individual):
        """
        Perform mutation on an individual - delegates to self.grammar, which additionally
        tries a structural temporal wrap/unwrap first when the grammar has a non-empty
        temporal_ops vocabulary (see TEMPORAL_SUBTREE_OPERATORS_PROMPT.md section 5b).
        """
        return self.grammar.mutate(individual, self.features, rng=self.rng)

            
    def get_mutation_rates(self):
        """
        Calculate or pull mutation rates for flat, DIM or DDM mutation.
        """
        # Flat mutation rate
        if self.mutation_method == 'flat':
            crossover_rate = self.crossover_mutation * self.max_mutation
            elitest_rate = self.elitest_mutation * self.max_mutation
        
        # Dynamically increasing mutation rate
        elif self.mutation_method == 'increasing':
            if len(self.results_tracker.diversity) > 0:
                mutation_rate = (self.population_size - self.results_tracker.diversity[-1]) / (self.population_size*(1-self.termination_proportion))
            else:
                mutation_rate = 0

            if mutation_rate > 1: 
                mutation_rate = 1
                
            total_mut = self.min_mutation + (self.max_mutation - self.min_mutation) * mutation_rate
            crossover_rate = self.crossover_mutation * total_mut
            elitest_rate   = self.elitest_mutation   * total_mut
           
        # Dynamically decreasing mutation rate
        elif self.mutation_method == 'decreasing':
            if len(self.results_tracker.diversity) > 0:
                mutation_rate = (self.results_tracker.diversity[-1] - self.population_size*self.termination_proportion) / (self.population_size*(1-self.termination_proportion))
            else:
                mutation_rate = 1
            
            if mutation_rate < 0:
                mutation_rate = 0
            
            total_mut = self.min_mutation + (self.max_mutation - self.min_mutation) * mutation_rate
            crossover_rate = self.crossover_mutation * total_mut
            elitest_rate   = self.elitest_mutation   * total_mut
            
        else:
            raise ValueError(f'Unsupported mutation method: {self.mutation_method}')
            
        true_mutation_rate = crossover_rate + elitest_rate
        self.results_tracker.update_mutation(true_mutation_rate)
        
        return crossover_rate, elitest_rate

    def update_population(self, selected_individuals: List[Tuple], evaluated_population: List[Tuple[str, float]]) -> None:
        """
        Update the population for the next generation after applying crossover and mutation.
        Tracks each accepted individual's canonical form (grammar.canonicalize) in `seen` and
        retries (one forced extra mutation, capped at MAX_DEDUP_ATTEMPTS) any crossover
        offspring/elite copy that would otherwise duplicate an individual already accepted this
        generation - unconditional duplicate acceptance previously let the population collapse
        toward one repeated individual within a handful of generations, paying full
        per-individual fitting cost for no added search breadth once that happens.

        The elitism loop's index 0 (the single best individual this generation,
        sorted_population[0]) is exempted from the dedup retry: with elitest_mutation=False (the
        default config, elitest_rate==0), this is what guarantees best_fitnesses is
        non-decreasing - forcing it through an extra mutation to satisfy uniqueness would silently
        break that guarantee. Every other elite slot (rank 2+) is fair game for the dedup policy,
        same as crossover offspring.
        """
        MAX_DEDUP_ATTEMPTS = 3
        new_population = []
        seen = set()
        crossover_rate, elitest_rate = self.get_mutation_rates()

        def accept(indiv):
            """
            Returns (accepted_individual, was_duplicate) - was_duplicate is True iff the
            individual's canonical form collided with one already accepted this generation and
            had to be force-mutated at least once. Aggregated below into admitted_rate - see
            __init__'s admitted_rate_threshold docstring for why this, not raw diversity, is
            what run()'s early-termination check needs under this policy.
            """
            was_duplicate = False
            for _ in range(MAX_DEDUP_ATTEMPTS):
                key = str(self.grammar.canonicalize(indiv))
                if key not in seen:
                    seen.add(key)
                    return indiv, was_duplicate
                was_duplicate = True
                indiv = self.mutate(indiv)
            seen.add(str(self.grammar.canonicalize(indiv)))
            return indiv, was_duplicate

        duplicate_count = 0

        # Apply crossover and mutation to generate the new population
        for i in range(0, len(selected_individuals), 2):
            parent1 = selected_individuals[i]
            parent2 = selected_individuals[i + 1] if i + 1 < len(selected_individuals) else selected_individuals[0]

            offspring = self.crossover(parent1, parent2)

            # Apply mutation to the crossover offspring based on the mutation rate
            if self.rng.random() < crossover_rate:
                offspring = self.mutate(offspring)

            accepted, was_duplicate = accept(offspring)
            new_population.append(accepted)
            duplicate_count += was_duplicate

            # Clean up the parents and offspring after they are used
            del parent1, parent2, offspring

        sorted_population = sorted(evaluated_population, key=lambda x: x[1], reverse=True)[:]

        # Apply mutation to the elitest offspring based on the mutation rate
        index = 0
        while len(new_population) < len(selected_individuals):
            indiv = sorted_population[index][0]
            if self.rng.random() < elitest_rate:
                indiv = self.mutate(indiv)
            if index == 0:
                seen.add(str(self.grammar.canonicalize(indiv)))
                new_population.append(indiv)
            else:
                accepted, was_duplicate = accept(indiv)
                new_population.append(accepted)
                duplicate_count += was_duplicate
            index += 1

        self.results_tracker.update_admitted_rate(duplicate_count / len(new_population))

        del self.population
        self.population = new_population
        del new_population
        
    @staticmethod
    def evaluate_and_save_predictions(individual, sector: str, gapreprocessing: GAPreprocessing, save_path: str, training_cache, testing_cache, grammar: ExpressionGrammar = None, fold_seed: int = None, gbt_max_iter: int = 10, fit_backend: str = "spark") -> float:
        """
        Evaluate the fitness of an individual that can be a complex feature expression,
        save the predictions and actual values for further analysis.

        Args:
        - individual: The feature expression to evaluate.
        - sector: The sector to use for data generation.
        - gapreprocessing: The GAPreprocessing instance for data generation.
        - save_path: The path where predictions should be saved.
        - fit_backend: "spark" (default) or "local" - see evaluate_fitness_static's docstring.

        Returns:
        - float: The RMSE of the model predictions.
        """
        grammar = grammar if grammar is not None else ExpressionGrammar(temporal_ops=[])
        is_local = fit_backend == "local"

        def leaf_fn(expression):
            hit_train, training_data = training_cache.get(expression)
            hit_test, testing_data = testing_cache.get(expression)
            if not (hit_train and hit_test):
                raise ValueError(f"Missing cached base feature: {expression}")
            return training_data, testing_data

        def temporal_fn(df, op):
            fn = apply_temporal_local if is_local else apply_temporal
            return fn(df, op, date_col=grammar.report_date_col, entity_col=grammar.entity_col)

        combine_fn = combine_dataframes_local if is_local else combine_dataframes
        training_data, testing_data = grammar.evaluate(individual, leaf_fn, combine_fn, temporal_fn)

        # testing_data was evaluated over a lookback-buffered frame (see
        # GAPreprocessing.get_true_test_frame's docstring) - trim back to genuine test rows now
        # that lag/rolling computation is done, before scoring/saving predictions below.
        if is_local:
            testing_data = testing_data[testing_data["sector_return_date"] >= gapreprocessing.true_eval_start_date]
        else:
            testing_data = testing_data.filter(col("sector_return_date") >= gapreprocessing.true_eval_start_date)

        gbt_seed = derive_seed(fold_seed, "evaluate_and_save_predictions", str(grammar.canonicalize(individual)))

        if is_local:
            training_data, testing_data = winsorize_feature_local(training_data, testing_data)
            feature_cols = ["prev_month_return", "prev_month_sector_return", "feature"]
            train_clean = training_data.dropna(subset=feature_cols + ["target"])
            test_clean = testing_data.dropna(subset=feature_cols + ["target"])

            model = XGBRegressor(
                n_estimators=gbt_max_iter, max_depth=5, learning_rate=0.1, random_state=gbt_seed,
                n_jobs=1,
            )
            model.fit(train_clean[feature_cols].to_numpy(), train_clean["target"].to_numpy())
            predictions_pd = test_clean[["fsym_id", "sector_return_date"]].assign(
                label=test_clean["target"].to_numpy(),
                prediction=model.predict(test_clean[feature_cols].to_numpy()),
            )
            rmse = float(np.sqrt(np.mean((predictions_pd["label"] - predictions_pd["prediction"]) ** 2)))
            predictions_pd.to_csv(f"{save_path}", index=False)
            del training_data, testing_data, train_clean, test_clean, model

        else:
            # Winsorize before the GBT sees the feature - see evaluate_fitness_static's docstring.
            training_data, testing_data = winsorize_feature(training_data, testing_data)

            training_data = training_data.select(col("fsym_id"), col("sector_return_date"), col("prev_month_return"), col("prev_month_sector_return"), col("feature"), col("target").alias("label"))
            testing_data = testing_data.select(col("fsym_id"), col("sector_return_date"), col("prev_month_return"), col("prev_month_sector_return"), col("feature"), col("target").alias("label"))

            gbt = GBTRegressor(featuresCol="features", labelCol="label", maxIter=gbt_max_iter, seed=gbt_seed)
            assembler = VectorAssembler(inputCols=["prev_month_return", "prev_month_sector_return", "feature"], outputCol="features", handleInvalid="skip")

            pipeline = Pipeline(stages=[assembler, gbt])

            model = pipeline.fit(training_data)

            predictions = model.transform(testing_data)
            evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
            rmse = evaluator.evaluate(predictions)

            # Convert to Pandas and save as a single CSV file
            predictions_pd = predictions.select("fsym_id", "sector_return_date", "label", "prediction").toPandas()
            predictions_pd.to_csv(f"{save_path}", index=False)

            del training_data, testing_data, predictions, model

        return -rmse
    
    @staticmethod
    def evaluate_and_save_baseline(sector: str, gapreprocessing: GAPreprocessing, save_path: str, training_cache: DataCache, testing_cache: DataCache, fold_seed: int = None, gbt_max_iter: int = 10) -> float:
        """
        Baseline model using ONLY prev_month_return and prev_month_sector_return (no super feature).
        'individual' is hardcoded to 'ff_pbk' and only used to pull aligned cached data.
        Saves predictions and returns negative RMSE (for GA compatibility).
        """

        # Hardcoded base feature name to fetch cached train/test frames - must be a feature
        # actually present in this run's cache (ff_pbk doesn't exist in the Finance-only
        # output; ff_pbk_tang does and is part of the sampled population, see cell 8 fix)
        individual = 'ff_pbk_tang'

        def evaluate_expression(expression):
            """
            Evaluate a simple (base) feature name to obtain aligned train/test DataFrames.
            """
            if isinstance(expression, str):
                hit_train, training_data = training_cache.get(expression)
                hit_test, testing_data = testing_cache.get(expression)
                if not (hit_train and hit_test):
                    raise ValueError(f"Missing cached base feature: {expression}")
                return training_data, testing_data
            else:
                raise ValueError("Invalid expression format: expected a base feature name (string).")

        # Get aligned training/testing frames; we will ignore any 'feature' column
        training_data, testing_data = evaluate_expression(individual)

        # Select ONLY the baseline predictors + label
        training_data = training_data.select(
            col("fsym_id"), col("sector_return_date"),
            col("prev_month_return"), col("prev_month_sector_return"),
            col("target").alias("label")
        )
        testing_data = testing_data.select(
            col("fsym_id"), col("sector_return_date"),
            col("prev_month_return"), col("prev_month_sector_return"),
            col("target").alias("label")
        )

        # Pipeline: assembler on baseline predictors only + GBT
        assembler = VectorAssembler(
            inputCols=["prev_month_return", "prev_month_sector_return"],
            outputCol="features",
            handleInvalid="skip"
        )
        gbt_seed = derive_seed(fold_seed, "evaluate_and_save_baseline")
        gbt = GBTRegressor(featuresCol="features", labelCol="label", maxIter=gbt_max_iter, seed=gbt_seed)
        pipeline = Pipeline(stages=[assembler, gbt])

        model = pipeline.fit(training_data)
        predictions = model.transform(testing_data)

        evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(predictions)

        # Save predictions
        predictions_pd = predictions.select("fsym_id", "sector_return_date", "label", "prediction").toPandas()
        predictions_pd.to_csv(f"{save_path}", index=False)

        # Cleanup
        del training_data, testing_data, predictions, model

        return -rmse

    @staticmethod
    def evaluate_and_save_multiple_predictions(features: list, sector: str, gapreprocessing: GAPreprocessing, save_path: str, training_cache: DataCache, testing_cache: DataCache, grammar: ExpressionGrammar = None, fold_seed: int = None, gbt_max_iter: int = 10) -> float:
        """
        Train a GBT model using MULTIPLE super-features in addition to
        prev_month_return and prev_month_sector_return. Saves predictions
        and returns negative RMSE.

        Assembler columns:
          ["prev_month_return", "prev_month_sector_return", "feature_1", ..., "feature_n"]

        Each feature_i is produced by recursively evaluating the i-th super-feature
        expression with the same combine_dataframes logic as in single-feature training.
        """
        if not features:
            raise ValueError("features must be a non-empty list of super-feature expressions.")

        grammar = grammar if grammar is not None else ExpressionGrammar(temporal_ops=[])

        def leaf_fn(expression):
            hit_train, training_data = training_cache.get(expression)
            hit_test, testing_data = testing_cache.get(expression)
            if not (hit_train and hit_test):
                raise ValueError(f"Missing cached base feature: {expression}")
            return training_data, testing_data

        def temporal_fn(df, op):
            return apply_temporal(df, op, date_col=grammar.report_date_col, entity_col=grammar.entity_col)

        def evaluate_expression(expression):
            return grammar.evaluate(expression, leaf_fn, combine_dataframes, temporal_fn)

        # Evaluate the first super-feature to seed base frames
        first_tr, first_te = evaluate_expression(features[0])

        base_train = first_tr.withColumnRenamed("feature", "feature_1").select(
            col("fsym_id"),
            col("sector_return_date"),
            col("prev_month_return"),
            col("prev_month_sector_return"),
            col("feature_1"),
            col("target")
        )
        base_test = first_te.withColumnRenamed("feature", "feature_1").select(
            col("fsym_id"),
            col("sector_return_date"),
            col("prev_month_return"),
            col("prev_month_sector_return"),
            col("feature_1"),
            col("target")
        )

        feature_cols = ["feature_1"]

        # For each additional super-feature, evaluate and join on keys
        for idx in range(1, len(features)):
            tr_i, te_i = evaluate_expression(features[idx])
            colname = f"feature_{idx+1}"

            tr_i = tr_i.withColumnRenamed("feature", colname).select(
                col("fsym_id"), col("sector_return_date"), colname
            )
            te_i = te_i.withColumnRenamed("feature", colname).select(
                col("fsym_id"), col("sector_return_date"), colname
            )

            base_train = (
                base_train.alias("a")
                .join(tr_i.alias("b"), on=["fsym_id", "sector_return_date"], how="inner")
                .select(
                    F.col("a.fsym_id").alias("fsym_id"),
                    F.col("a.sector_return_date").alias("sector_return_date"),
                    F.col("a.prev_month_return").alias("prev_month_return"),
                    F.col("a.prev_month_sector_return").alias("prev_month_sector_return"),
                    *[F.col(f"a.{c}").alias(c) for c in feature_cols],
                    F.col(f"b.{colname}").alias(colname),
                    F.col("a.target").alias("target"),
                )
            )
            base_test = (
                base_test.alias("a")
                .join(te_i.alias("b"), on=["fsym_id", "sector_return_date"], how="inner")
                .select(
                    F.col("a.fsym_id").alias("fsym_id"),
                    F.col("a.sector_return_date").alias("sector_return_date"),
                    F.col("a.prev_month_return").alias("prev_month_return"),
                    F.col("a.prev_month_sector_return").alias("prev_month_sector_return"),
                    *[F.col(f"a.{c}").alias(c) for c in feature_cols],
                    F.col(f"b.{colname}").alias(colname),
                    F.col("a.target").alias("target"),
                )
            )

            feature_cols.append(colname)

        training_data = base_train.select(
            col("fsym_id"), col("sector_return_date"),
            col("prev_month_return"), col("prev_month_sector_return"),
            *[col(c) for c in feature_cols],
            col("target").alias("label")
        )
        testing_data = base_test.select(
            col("fsym_id"), col("sector_return_date"),
            col("prev_month_return"), col("prev_month_sector_return"),
            *[col(c) for c in feature_cols],
            col("target").alias("label")
        )

        # Winsorize each evolved feature_i column before the GBT sees it - see
        # evaluate_fitness_static's docstring. One feature_i at a time since winsorize_feature
        # fits/clips bounds per named column.
        for feature_col in feature_cols:
            training_data, testing_data = winsorize_feature(training_data, testing_data, column=feature_col)

        # Train & evaluate
        input_cols = ["prev_month_return", "prev_month_sector_return"] + feature_cols
        assembler = VectorAssembler(inputCols=input_cols, outputCol="features", handleInvalid="skip")
        gbt_seed = derive_seed(fold_seed, "evaluate_and_save_multiple_predictions", tuple(str(grammar.canonicalize(f)) for f in features))
        gbt = GBTRegressor(featuresCol="features", labelCol="label", maxIter=gbt_max_iter, seed=gbt_seed)
        pipeline = Pipeline(stages=[assembler, gbt])

        model = pipeline.fit(training_data)
        predictions = model.transform(testing_data)

        evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(predictions)

        # Save predictions
        predictions_pd = predictions.select("fsym_id", "sector_return_date", "label", "prediction").toPandas()
        predictions_pd.to_csv(f"{save_path}", index=False)

        # Cleanup references
        del base_train, base_test, training_data, testing_data, predictions, model

        return -rmse
    
    def checkpoint_state(self, generation: int, fingerprint: str) -> dict:
        """
        Serialize enough state to resume this run from generation+1: the population about to
        start the next generation, fitness/diversity/admitted-rate/mutation histories, best_individual, and
        self.rng's internal state (random.Random.getstate() - a nested tuple of ints/floats,
        JSON round-trips it as nested lists; restore_from_checkpoint below reconstructs the
        tuples). `fingerprint` (see ga/checkpoint.py's compute_fingerprint) is stored alongside
        so a later resume attempt can tell whether the fold's data or GAConfig changed since this
        checkpoint was written.
        """
        return {
            "fingerprint": fingerprint,
            "generation": generation,
            "population": self.population,
            "best_individual": self.best_individual,
            "rng_state": self.rng.getstate(),
            "best_fitnesses": self.results_tracker.best_fitnesses,
            "average_fitnesses": self.results_tracker.average_fitnesses,
            "worst_fitnesses": self.results_tracker.worst_fitnesses,
            "diversity": self.results_tracker.diversity,
            "admitted_rate": self.results_tracker.admitted_rate,
            "mutation": self.results_tracker.mutation,
            "best_individuals": self.results_tracker.best_individuals,
        }

    def restore_from_checkpoint(self, checkpoint: dict) -> int:
        """
        Restore population/histories/rng state from a fingerprint-matched checkpoint (see
        ga/checkpoint.py). Must be called AFTER initialize_population() and a real gen-0
        evaluate_population(gen=0) pass have already run on THIS instance (run_ga_for_fold's
        existing baseline-B capture already does exactly this, unconditionally, for every fold)
        - that pass is what warms training_cache/testing_cache/prediction_cache, and it's
        guaranteed to reproduce the ORIGINAL run's exact generation-0 population bit-for-bit:
        self.features (sampled in __init__) and initialize_population()'s seed-fraction wrap
        selection are the first two things to consume self.rng, in that order, so given the same
        seed (implied by the fingerprint match on fold data + GAConfig) they land on the same
        population every time, regardless of how many generations the original run got through
        before crashing.

        Skipping straight to the checkpointed population without that real gen-0 pass first
        would leave training_cache/testing_cache missing entries for any leaf/atom the
        checkpointed population had already dropped by the time it was saved - a later mutation
        reintroducing it would then raise "Missing cached base feature", a crash that couldn't
        happen in an uninterrupted run of the same fold.

        Returns resume_generation: the first generation index run()'s loop still needs to run.
        """
        self.population = [_lists_to_tuples(ind) for ind in checkpoint["population"]]
        self.best_individual = _lists_to_tuples(checkpoint["best_individual"])
        self.rng.setstate(_lists_to_tuples(checkpoint["rng_state"]))
        self.results_tracker.best_fitnesses = list(checkpoint["best_fitnesses"])
        self.results_tracker.average_fitnesses = list(checkpoint["average_fitnesses"])
        self.results_tracker.worst_fitnesses = list(checkpoint["worst_fitnesses"])
        self.results_tracker.diversity = list(checkpoint["diversity"])
        self.results_tracker.admitted_rate = list(checkpoint.get("admitted_rate", []))
        self.results_tracker.mutation = list(checkpoint["mutation"])
        self.results_tracker.best_individuals = [_lists_to_tuples(b) for b in checkpoint["best_individuals"]]
        return checkpoint["generation"] + 1

    def run(self, start_generation: int = 0, checkpoint_path: str = None, checkpoint_fingerprint: str = None) -> None:
        """
        Run the genetic algorithm for a specified number of generations.

        Args:
            num_generations (int): The number of generations to run.
            start_generation: first generation index to run - 0 for a fresh run, or
                restore_from_checkpoint's return value when resuming.
            checkpoint_path/checkpoint_fingerprint: when both are given, write a checkpoint (see
                checkpoint_state above) after every completed generation - a lightweight progress
                log, not a guarantee that a resumed run skips all re-fitting (see
                restore_from_checkpoint's docstring for what resuming actually costs).
        """
        # Optionally monitor memory usage
        #self.stop_event = threading.Event()
        #monitor_thread = threading.Thread(
        #    target=self.monitor_all_memory,
        #    args=(self.log_list, 20, self.stop_event),  # 5 second interval
        #    daemon=True
        #)
        #monitor_thread.start()

        try:
            for generation in range(start_generation, self.generations):
                print(f"\n--- Generation {generation + 1}/{self.generations} ---")
                begin_time = time.time()

                # Evaluate the fitness of the current population
                evaluated_population = self.evaluate_population(gen=generation)

                # Update the results tracker
                self.results_tracker.update_fitness(evaluated_population)

                # Select individuals for the next generation
                selected_individuals = self.tournament_selection(evaluated_population, len(self.population))

                # Select individuals for the next generation using roulette selection
                #selected_individuals = self.roulette_selection(evaluated_population, len(self.population))

                # Apply crossover and mutation to generate the next generation
                self.update_population(selected_individuals, evaluated_population)

                # Copy the best individual of the generation
                best_individual = max(evaluated_population, key=lambda x: x[1])
                self.results_tracker.update_best_individuals(best_individual[0])
                
                # Sort the population by fitness in descending order and print the top 10 individuals
                top_10 = sorted(evaluated_population, key=lambda x: x[1], reverse=True)[:10]
                print('\n')
                for i, (individual, fitness) in enumerate(top_10, 1):
                    print(f"Rank {i}: Individual: {individual}, Fitness: {fitness}")

                # Update best individual
                self.best_individual = best_individual[0]

                # Update the diversity in the tracker - canonicalized, so structurally-identical
                # individuals with different spellings (e.g. two chained lags vs. their collapsed
                # form) count as one for diversity purposes instead of inflating the count.
                self.results_tracker.update_diversity(
                    [self.grammar.canonicalize(ind) for ind in self.population]
                )

                # Stash a canonicalized snapshot of the top-K individuals from this generation
                # before evaluated_population is discarded below - used by the cross-run
                # stability facility (analysis/stability.py) to compare top-K overlap across
                # seeded reruns of the same fold, not just the single final winner. Placed BEFORE
                # the early-termination check below (not after), so a fold that terminates early
                # still captures its true final generation's top-K rather than an earlier one.
                self.final_ranked_population = [
                    str(self.grammar.canonicalize(ind))
                    for ind, _ in sorted(evaluated_population, key=lambda x: x[1], reverse=True)[:self.top_k]
                ]

                # Early termination - two independent conditions (see __init__'s
                # admitted_rate_threshold/stagnation_ceiling docstrings for the full reasoning):
                #
                # 1. Fitness stagnant AND the search has converged - update_population()'s dedup
                #    policy forces a mutation retry on any canonical-form collision, which keeps
                #    raw population diversity pinned near population_size almost by construction
                #    (a population dominated by one building block still looks fully diverse by
                #    unique-canonical-form count, since the forced retry mutates some other leaf/
                #    operator into something syntactically novel). admitted_rate - the fraction of
                #    each generation's individuals that needed >=1 such retry - is what actually
                #    signals "the search keeps landing on things it's already tried," so it (not
                #    raw diversity) gates this condition.
                # 2. An unconditional fitness-only ceiling, independent of admitted_rate - so a
                #    fold that stays behaviorally diverse (low admitted_rate) but never improves
                #    still has a bound instead of grinding to `generations`.
                fitness_history = self.results_tracker.get_fitnesses()[0]

                if len(fitness_history) >= 10:
                    last_10_fitness = fitness_history[-10:]
                    fitness_stagnant = all(f == last_10_fitness[0] for f in last_10_fitness)

                    recent_admitted_rate = self.results_tracker.admitted_rate[-10:]
                    admitted_rate_avg = sum(recent_admitted_rate) / len(recent_admitted_rate)
                    converged = admitted_rate_avg >= self.admitted_rate_threshold

                    if fitness_stagnant and converged:
                        print(f"Terminating early at Generation {generation+1}: fitness stagnant and admitted-duplicate rate converged ({admitted_rate_avg:.2f} >= {self.admitted_rate_threshold}).")
                        break

                if len(fitness_history) >= self.stagnation_ceiling:
                    last_ceiling_fitness = fitness_history[-self.stagnation_ceiling:]
                    if all(f == last_ceiling_fitness[0] for f in last_ceiling_fitness):
                        print(f"Terminating at Generation {generation+1}: no fitness improvement in {self.stagnation_ceiling} generations (stagnation ceiling).")
                        break

                # Clean up after each generation
                del evaluated_population, selected_individuals, best_individual

                gc.collect()

                finish_time = time.time()
                time_taken = finish_time - begin_time
                print(f'Generation {generation+1} took {time_taken:.2f} seconds')
                self.time_taken.append(time_taken)

                if checkpoint_path and checkpoint_fingerprint:
                    save_checkpoint(checkpoint_path, self.checkpoint_state(generation, checkpoint_fingerprint))
        finally:
            print('Run complete')
        #    self.stop_event.set()
        #    monitor_thread.join()

        # Plot the fitness progression over generations
        self.results_tracker.plot_fitness_progression(save_path="fitness_progression.png")
        
    def monitor_all_memory(self, log_list, interval=20, stop_event=None):
        """
        Background memory logger: driver, system, and executor memory.
        Appends tuples to shared `log_list`.
        """
        process = psutil.Process(os.getpid())
        start_time = time.time()

        while True:
            if stop_event is not None and stop_event.is_set():
                break  # Exit if stop signal is triggered
            
            try:
                driver_rss_mb = process.memory_info().rss / 1024**2
                sys_mem = psutil.virtual_memory()
                system_used_mb = sys_mem.used / 1024**2
                system_avail_mb = sys_mem.available / 1024**2

                exec_used_mb, exec_max_mb = log_executor_memory_rest(driver_host, port)

                log_entry = {
                    'timestamp': time.time() - start_time,
                    'driver_rss_mb': driver_rss_mb,
                    'system_used_mb': system_used_mb,
                    'system_avail_mb': system_avail_mb,
                    'executor_used_mb': exec_used_mb,
                    'executor_max_mb': exec_max_mb
                }

                log_list.append(log_entry)

            except Exception as e:
                print(f"[MONITOR ERROR] {e}")

        if stop_event is not None:
            stop_event.wait(interval)
        else:
            time.sleep(interval)