"""
PipelineConfig/GAConfig — the two axes `run_preprocessing.py`/`run_ga.py` run against: scale
(`fast_mode`/`--fast`, full universe+feature set vs. a reduced one for quick iteration) and, for
the GA only, whether live temporal subtree operators are enabled
(`enable_temporal_operators`/`--no-temporal-operators` - see TEMPORAL_SUBTREE_OPERATORS_PROMPT.md).
These used to be framed as "which frozen notebook variant" (`TEST_CONFIG`/`TEST_V2_CONFIG` etc.,
reproducing `PreProcessing_test.ipynb`/`_v2`/`GA_test.ipynb`/`_v2`), but once
`add_temporal_features` was retired (see below) the `test`/`test_v2` pair had nothing left to
differ on besides an output namespace - replaced by the fast/full axis instead, which is a real,
useful distinction (wall-clock cost) rather than a notebook-parity artifact.

`add_temporal_features` (the old *precomputed* `_lag1`-style leaf columns) defaults to `False`
now - it was retired once the GA's live temporal operators (`genome/grammar.py`/
`operators/temporal.py`) were validated on real fold data to reproduce it (99.5-99.8% exact
match; the remaining ~0.2-0.5% is explained, not a bug - `add_temporal_features` computes over
the full unfiltered report history, while the live evaluator only ever sees reports that
survived `feature_selection_dataset`'s later coverage-window/collision filtering - arguably the
live version's semantics are the more defensible ones). The field (and
`Utils.add_temporal_features` itself) stay available - flip to `True` for a run that wants the
precomputed columns as a fallback/legacy candidate pool alongside the live atoms.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PipelineConfig:
    # Utils(...) constructor args.
    num_partitions: int = 20
    start_date: str = '2001-01-01'
    market_value_threshold: int = 5000
    sector_filter: Optional[str] = 'Finance'

    # Precomputed temporal feature augmentation (Utils.add_temporal_features) - retired, see
    # module docstring above. Utils.add_temporal_features itself stays defined/importable.
    add_temporal_features: bool = False
    temporal_lag_periods: List[int] = field(default_factory=lambda: [1])
    temporal_window_sizes: List[int] = field(default_factory=lambda: [3])

    # Output path namespace - the HDFS walk-forward base checkpoint and every fold's output_dir
    # are rooted here. FULL_CONFIG/FAST_CONFIG below point this at "walk_forward_full"/
    # "walk_forward_fast" specifically so neither can ever collide with the legacy GA_v0-lineage
    # data already sitting at plain "walk_forward" on HDFS.
    walk_forward_namespace: str = "walk_forward_full"

    # Fast mode (--fast on run_preprocessing.py): reduced-scale universe/feature caps, ported
    # from PreProcessing_test.ipynb's own disabled-by-default "FAST TEST PIPELINE" cell-8 block
    # (see CLAUDE.md) rather than invented fresh. Only applied when fast_mode=True; full scale
    # (the default) ignores all four of these.
    fast_mode: bool = False
    fast_n_features: int = 100
    fast_stocks_per_sector: int = 30
    fast_sample_seed: int = 42
    fast_terminal_cap: int = 15  # ConsensusFeatureSelector.TERMINAL_CAP override; full scale uses its class default (None - uncapped)

    # Walk-forward fold boundary computation - configurable since nothing ties these to a
    # specific mode. Both FULL_CONFIG and FAST_CONFIG below override target_dev_folds away from
    # this class default (to 5) - see their own comments for why, and why that's safe to do
    # without also touching target_final_test_folds.
    final_test_fraction_start: float = 0.80
    target_dev_folds: int = 3
    embargo_months: int = 1

    # Width, in years, of every dev-validation and final-test fold's eval window (both - not
    # independently configurable). 2 means each fold covers 2 consecutive calendar years instead
    # of 1. Widening this shrinks the initial-training window (see target_final_test_folds
    # below): with 26 years of history (2001-2026), target_dev_folds=5, target_final_test_folds=5,
    # year_width=2 works out to initial train 2001-2006, 5 dev folds covering 2007-2016, 5
    # final-test folds covering 2017-2026 (each train/inner-val/eval boundary expands by
    # year_width per fold, same walk-forward shape as width 1, just coarser) - both FULL_CONFIG
    # and FAST_CONFIG land here now that they share target_dev_folds=5. See compute_fold_boundaries
    # (evaluation/splits.py): final_test_start_idx is derived from target_final_test_folds alone,
    # computed before dev_validation_years is sliced immediately below it, so growing
    # target_dev_folds only ever eats into the initial-training window, never the already-fixed
    # final-test region.
    year_width: int = 2

    # Overrides final_test_fraction_start: final_test_start_idx is chosen so exactly
    # target_final_test_folds * year_width years fall in the final-test region, counting back
    # from the most recent year - rather than deriving the final-test region's size from a fixed
    # fraction of history and then just chunking whatever falls out of it. This is what forces
    # exactly 5 final-test folds (instead of whatever a fraction-based split happens to produce),
    # at the cost of eating into what would otherwise be dev-validation/initial-training years.
    target_final_test_folds: Optional[int] = 5

    random_seed: int = 42

    @property
    def walk_forward_base_path(self) -> str:
        return f"{self.walk_forward_namespace}/base/experiment_1_df.parquet"

    def fold_output_dir(self, category: str, fold_index: int) -> str:
        return f"{self.walk_forward_namespace}/{category}/fold_{fold_index:02d}"


FULL_CONFIG = PipelineConfig(walk_forward_namespace="walk_forward_full", target_dev_folds=5)
# target_dev_folds=5 (vs the class default of 3): matches FAST_CONFIG's own dev-fold count (see
# year_width's comment above for the exact year-boundary math - initial train shrinks to
# 2001-2006, 5 dev folds cover 2007-2016, final-test region stays UNCHANGED at 2017-2026 since
# target_final_test_folds is derived independently, before dev_validation_years is sliced -
# growing target_dev_folds only ever eats into the initial-training window, never final-test).
# --fast only ever feeds run_ga.py's dev-fold-only hyperparameter sweeps now
# (run_preprocessing.py skips building final-test folds entirely when fast_mode=True - see
# _run_folds), so it gets more dev folds instead of a final-test region it'd never use;
# FULL_CONFIG builds both regions and now matches it on dev-fold count too.
FAST_CONFIG = PipelineConfig(walk_forward_namespace="walk_forward_fast", fast_mode=True, target_dev_folds=5)


@dataclass
class GAConfig:
    walk_forward_namespace: str = "walk_forward_full"
    output_dir: str = "ga"

    target_population_size: int = 100
    # tournament_size/mutation_method/max_mutation/min_mutation/gbt_max_iter (below) and
    # max_features/temporal_wrap_rate/temporal_unwrap_rate (further down) are the 8 hyperparameters
    # the joint Optuna search (optuna_sweep.db, study "ga_temporal_sweep_full_joint", 26/40 completed
    # trials - the 14 failures were cluster/YARN infra issues, not bad configs) tuned jointly across
    # both arms via J(theta) = 0.5*(IC_ON(theta) + IC_OFF(theta)) on development folds, seeds 11/12.
    # Set here to trial 3's values (best J=0.2081; ic_on=0.3029, ic_off=0.1133) as of 2026-08-26 -
    # every unflagged run_ga.py invocation now uses these instead of the untuned originals.
    tournament_size: int = 6                # was 4
    generations: int = 500
    mutation_method: str = 'flat'           # was 'increasing'
    crossover_mutation: bool = True
    elitest_mutation: bool = False
    max_mutation: float = 0.5552            # was 0.4
    min_mutation: float = 0.2053            # was 0.1
    num_threads: int = 8
    num_threads_baseline_c: int = 8

    run_baseline_c: bool = True
    sector: str = "Finance"

    # GBTRegressor's maxIter (= number of boosting trees) used to score every individual's
    # fitness (GeneticAlgorithm1.evaluate_fitness_static and friends, engine.py). Broken out as
    # its own config field so run_ga.py's --gbt-tree-search sweep can vary it per run without
    # touching engine.py's previously-hardcoded maxIter=10. Set to the joint Optuna search's
    # best trial (see tournament_size's comment above) as of 2026-08-26 - was 10.
    gbt_max_iter: int = 55

    # What GeneticAlgorithm1.evaluate_fitness_static scores each individual on - "rank_ic"
    # (default as of 2026-08-26 - matches run_ga.py's own CLI default, which already forced this
    # unconditionally in its config construction regardless of this dataclass field; the two had
    # drifted apart and caused real confusion about which default actually governed a run, so this
    # field was brought into line with run_ga.py's own default rather than leaving them to diverge)
    # or "rmse" (fitness = -RMSE on the fold's validation rows, pooled across months - the
    # original behavior, still available via --fitness-metric rmse for reproducing legacy
    # rmse-scored runs). "rank_ic" fitness = mean monthly Spearman rank IC over the validation
    # period, per the research question's actual estimand - GA maximizes this directly, no
    # negation, unlike RMSE. Does NOT change what the GBTRegressor is trained to predict - it's
    # still fit to next-month returns either way; only the GA's own selection signal changes. See
    # run_ga.py's --fitness-metric flag. NOTE: GeneticAlgorithm1.__init__/evaluate_fitness_static
    # (engine.py) keep their own separate "rmse" defaults deliberately unchanged - a direct,
    # non-run_ga.py caller (e.g. tests/test_ga_engine_local_backend.py's RMSE-shaped sign/magnitude
    # assertions) that constructs GeneticAlgorithm1 without passing fitness_metric still gets rmse.
    fitness_metric: str = "rank_ic"

    # Which implementation GeneticAlgorithm1.evaluate_fitness_static (and its true-test/baseline-A
    # siblings) use to fit/score each individual's GBT model - "spark" (default, unchanged
    # original behavior: pyspark.ml.regression.GBTRegressor per individual, on the cluster) or
    # "local" (xgboost.XGBRegressor per individual, in-process pandas/numpy - see
    # operators/arithmetic_local.py/operators/temporal_local.py for the matching pandas-side
    # combine/winsorize/temporal-window logic). Added after profiling a live run's Spark UI: for
    # data this small (a few hundred rows/fold), per-individual Spark/YARN scheduling overhead
    # (measured ~59% of executor run time is deserialization alone) dwarfs actual GBT compute -
    # "local" exists to eliminate that overhead. XGBoost specifically (over sklearn's own GBT,
    # tried first) because it releases the GIL during fitting - the existing per-individual
    # ThreadPoolExecutor gets real multi-core parallelism this way, unlike sklearn's classic GBM -
    # and defaults to histogram/approximate-quantile split finding (like MLlib's own tree
    # builder), unlike sklearn's exact-sort split finding - both sklearn and XGBoost otherwise
    # share MLlib's depth-wise (not leaf-wise) tree growth.
    # Deliberately NOT numerically reproducible against "spark" even at matched seeds/
    # hyperparameters regardless - XGBoost's split-gain regularization terms and RNG internals
    # differ from MLlib's - this is a separate, parallel methodology arm ("epoch"),
    # not a drop-in replacement; existing seed 7/10-12 Spark-backend results
    # (docs/paper_tables.md) stay reproducible and untouched under fit_backend="spark". Only the
    # per-individual evaluation loop goes local - the one-time per-fold HDFS data load
    # (GAPreprocessing.__init__) still runs on Spark regardless of this setting; see
    # operators/*_local.py's module docstrings for exactly where the Spark->pandas handoff
    # happens. See run_ga.py's --fit-backend flag.
    fit_backend: str = "spark"

    # Base seed for the GA search itself (GeneticAlgorithm1's population init, selection,
    # crossover/mutation, and the per-individual GBT seed derived from it - see
    # ga/seeding.py's derive_seed and ga/algorithms.py's run_ga_for_fold). Distinct from, and
    # unrelated to, PipelineConfig.random_seed above (which seeds ConsensusFeatureSelector
    # during preprocessing) - same "duplicated, not shared, across PipelineConfig/GAConfig"
    # convention as walk_forward_namespace. Each fold derives its own independent seed from this
    # base value (derive_seed(base_seed, fold_name)), so folds processed in one run don't share
    # one RNG stream.
    random_seed: int = 42

    # Temporal subtree operators (TEMPORAL_SUBTREE_OPERATORS_PROMPT.md) - the GA's live
    # lag/delta/growth/mean/std atoms, evaluated during the search itself rather than sourced
    # from precomputed PipelineConfig.add_temporal_features leaf columns. Mirrors
    # PipelineConfig.temporal_lag_periods/temporal_window_sizes' shape but kept as its own
    # fields here (same convention as walk_forward_namespace already being duplicated, not
    # shared, across PipelineConfig/GAConfig) so the GA's operator vocabulary can be tuned
    # independently of whatever the preprocessing step precomputed. run_ga.py's
    # --no-temporal-operators flag flips this off for a run.
    enable_temporal_operators: bool = True
    temporal_lag_periods: List[int] = field(default_factory=lambda: [1, 2, 4])
    temporal_window_sizes: List[int] = field(default_factory=lambda: [2, 3, 4])
    temporal_report_date_col: str = "report_date"

    # Grammar limits (engine.py's crossover_deep used to hardcode both as local constants -
    # promoted here so methodology.py's _random_individual can share the same max_features
    # instead of hand-duplicating it). max_features/temporal_wrap_rate/temporal_unwrap_rate set
    # to the joint Optuna search's best trial as of 2026-08-26 - see tournament_size's comment
    # above; were 5/0.15/0.15.
    max_features: int = 6
    max_nesting: int = 5
    max_temporal_depth: int = 2
    temporal_wrap_rate: float = 0.2703
    temporal_unwrap_rate: float = 0.1711

    # Seeded-fraction bootstrap (gafactor's own measurement: mutation-only introduction of
    # temporal atoms, opposed by an equal-and-opposite unwrap rate above, adopts too slowly to
    # bootstrap from zero - single digits of population adoption by generation 5). Fraction of
    # generation-0 individuals GeneticAlgorithm1.initialize_population() pre-wraps in a random
    # temporal operator instead of leaving every individual as a bare, unwrapped leaf.
    seed_fraction: float = 0.3

    # Early-termination tuning (GeneticAlgorithm1.run()) - see __init__'s
    # admitted_rate_threshold/stagnation_ceiling docstrings in engine.py for the full reasoning.
    # admitted_rate_threshold: fraction (10-generation average) of a generation's individuals
    # that must need a forced dedup-retry mutation, alongside 10 generations of stagnant best
    # fitness, to call the search converged. stagnation_ceiling: unconditional generations-
    # without-improvement backstop, independent of admitted_rate.
    # 0.5 (the original starting point) turned out to be uncalibrated - two live full-scale runs
    # checkpointed mid-flight (2026-08-11) showed admitted_rate oscillating ~0.05-0.2 (10-gen
    # trailing averages ~0.10-0.13) even after best fitness had already gone 17-24 generations
    # without changing, so the converged branch was unreachable in practice and every fold was
    # falling through to the 50-generation stagnation_ceiling backstop instead. 0.15 sits just
    # above the observed noise ceiling during real stagnation so the intended early-convergence
    # path actually fires; re-check against admitted_rate_progression.png once more folds have
    # run under this value; adjust if it fires too eagerly/rarely.
    admitted_rate_threshold: float = 0.15
    stagnation_ceiling: int = 50

    @property
    def mutation_config(self):
        return (self.mutation_method, self.crossover_mutation, self.elitest_mutation,
                self.max_mutation, self.min_mutation)

    def fold_dir_hdfs(self, category: str, fold_name: str) -> str:
        return f"{self.walk_forward_namespace}/{category}/{fold_name}"


# output_dir here is the BASE name - run_ga.py's --no-temporal-operators flag appends
# "_no_temporal" to whichever of these is selected (e.g. "ga_fast" -> "ga_fast_no_temporal"),
# so all 4 combinations of --fast / --no-temporal-operators land in distinct directories:
# ga/, ga_no_temporal/, ga_fast/, ga_fast_no_temporal/.
FULL_GA_CONFIG = GAConfig(walk_forward_namespace="walk_forward_full", output_dir="ga")
FAST_GA_CONFIG = GAConfig(
    walk_forward_namespace="walk_forward_fast", output_dir="ga_fast",
    target_population_size=15, generations=15,
)
