"""
Ported from `research/GA_test.ipynb` cell 6 — see `docs/RESTRUCTURING_TODO.md` and the
port plan in `docs/RESEARCH_STRUCTURE.md` / `docs/GA_structure.md`. Extracted mechanically
from the notebook cell source (not retyped) to avoid transcription drift; the notebook
remains the frozen parity reference. Only the import block was touched: names the cell relied
on getting from the shared notebook kernel namespace (cell 0's imports) are now imported
explicitly here, since a standalone module has no such shared kernel state.
"""

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
import json as _json
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, min as spark_min
from typing import List

class GAPreprocessing:
    """
    Loads one walk-forward fold's data as produced by PreProcessing_test.ipynb's run_fold() -
    already joined (as-of, off fund_report_date), already point-in-time-correct, already
    feature-selected (train-rows-only, per fold). No fundamentals<->returns alignment is
    re-derived here anymore - that used to be this class's job (feature_trimming,
    precompute_sector_avg_returns, the report-expansion/join logic inside
    generate_training_testing_data), and it leaked (zero reporting lag, same-month target
    pairing - see the leakage audit). All of that is now PreProcessing_test's responsibility;
    this class just reads its output.

    fold_dir_hdfs: e.g. "walk_forward_test/development/fold_01" (relative - resolves against
      fs.defaultFS, i.e. HDFS) - where train.csv/{validation,test}.csv live.
    fold_dir_local: e.g. "walk_forward_test/development/fold_01" - where
      selected_features.txt/fold_metadata.json live (local disk - PreProcessing_test's
      write_to_csv writes HDFS, write_fold_metadata/write_consensus_artifacts write
      local disk, for the same fold/output_dir).
    eval_label: 'validation' (development folds) or 'test' (final-test folds), matching
      PreProcessing_test's run_fold() output file naming.

    PreProcessing_test.ipynb's run_fold() now writes a genuine 3-way split for final-test folds:
    train.csv, an inner validation.csv (carved out of what used to be the tail of the training
    window), and test.csv (fold_metadata.json's inner_val_year is non-null exactly when this
    applies). self.eval_df - what cell 19's generate_training_testing_data reads for every
    generation's fitness scoring - is pointed at validation.csv when it exists, so the GA's
    per-generation search never touches test.csv. self.true_eval_df always holds the eval_label
    file's own content (test.csv for final-test folds; for development folds, where there's no
    separate inner split, it's the same object as self.eval_df) - untouched by anything during
    the GA run, and only read by run_ga_for_fold() after a fold's winning individual is already
    locked in, via get_true_test_frame().
    """

    # Structural/id/date/sector columns, never candidate features - mirrors the exclusion list
    # the old get_feature_list() used, plus prev_month_return/prev_month_sector_return: these
    # are hardcoded baseline predictors GA always assembles alongside the one evolved feature
    # (see cell 19's evaluate_fitness_static/VectorAssembler), never something the GA itself
    # evolves/combines - PreProcessing_test force-keeps them in every fold's selected_features.txt
    # regardless of feature selection, so they must be excluded here or the GA would try to
    # mutate/crossover them like ordinary candidates.
    _NON_FEATURE_COLUMNS = {
        'fsym', 'date', 'factset_sector_desc', 'target_date', 'return_year', 'return_month',
        'monthly_return', 'prev_month_return', 'prev_month_sector_return',
    }

    def __init__(self, spark: SparkSession, fold_dir_hdfs: str, fold_dir_local: str, eval_label: str, num_partitions: int = 24):
        self.spark = spark
        self.fold_dir_hdfs = fold_dir_hdfs
        self.fold_dir_local = fold_dir_local
        self.eval_label = eval_label
        self.sector_column = 'factset_sector_desc'
        self.num_partitions = num_partitions

        with open(f"{fold_dir_local}/selected_features.txt") as f:
            self.selected_features = [line.strip() for line in f if line.strip()]

        with open(f"{fold_dir_local}/fold_metadata.json") as f:
            self.fold_metadata = _json.load(f)

        self.train_df = (
            spark.read.option("header", "true").csv(f"{fold_dir_hdfs}/train.csv", inferSchema=True)
            .repartition(num_partitions).cache()
        )

        # has_inner_validation is True exactly for final-test folds under the 3-way split (see
        # PreProcessing_test.ipynb's run_fold(), inner_val_year param) - self.eval_df is pointed at
        # validation.csv so every generation's fitness scoring never touches test.csv; the
        # eval_label file's own content is preserved separately as self.true_eval_df either way,
        # so it's always available for the one honest evaluation after the GA run completes.
        self.has_inner_validation = self.fold_metadata.get('inner_val_year') is not None
        if self.has_inner_validation:
            self.eval_df = (
                spark.read.option("header", "true").csv(f"{fold_dir_hdfs}/validation.csv", inferSchema=True)
                .repartition(num_partitions).cache()
            )
            self.true_eval_df = (
                spark.read.option("header", "true").csv(f"{fold_dir_hdfs}/{eval_label}.csv", inferSchema=True)
                .repartition(num_partitions).cache()
            )
        else:
            self.eval_df = (
                spark.read.option("header", "true").csv(f"{fold_dir_hdfs}/{eval_label}.csv", inferSchema=True)
                .repartition(num_partitions).cache()
            )
            self.true_eval_df = self.eval_df

        # Live temporal atoms (lag1/mean3/etc., operators/temporal.py's apply_temporal) window
        # over whatever frame they're handed - if that frame were exactly self.eval_df/
        # self.true_eval_df alone, the first rows per entity at each split boundary would have no
        # preceding row to look back to and lag/rolling features would wrongly reset there, even
        # though that history was legitimately known (it's sitting right there in train_df/
        # eval_df). generate_training_testing_data/get_true_test_frame below fix this by unioning
        # in the preceding split(s) as lookback context before evaluation, then everything
        # downstream (ga/engine.py's evaluate_fitness_static/evaluate_and_save_predictions) trims
        # back to genuine eval/test rows using these cutoffs, post-evaluation - not a leakage risk
        # (only earlier, already-known rows are added), just recovering information that was being
        # silently discarded at every fold boundary.
        self.eval_start_date = self.eval_df.agg(spark_min('target_date')).first()[0]
        self.true_eval_start_date = self.true_eval_df.agg(spark_min('target_date')).first()[0]

    def get_feature_list(self) -> List[str]:
        return [f for f in self.selected_features if f not in self._NON_FEATURE_COLUMNS]

    @staticmethod
    def sector_names() -> list():
        return ['Finance', 'Commercial Services', 'Communications', 'Consumer Durables',
                'Consumer Non-Durables','Consumer Services','Distribution Services','Electronic Technology',
                'Energy Minerals','Health Services','Health Technology','Industrial Services',
                'Non-Energy Minerals','Process Industries','Producer Manufacturing',
                'Retail Trade', 'Technology Services', 'Transportation', 'Utilities']

    def _feature_frame(self, df: DataFrame, feature: str) -> DataFrame:
        """
        Select+rename one base feature's already-aligned column out of this fold's DataFrame
        into the shape combine_dataframes()/evaluate_expression() (cell 19) already expect -
        fsym_id, sector_return_date, prev_month_return, prev_month_sector_return, feature,
        target - so cell 19 needs zero changes. Also carries the underlying fundamentals
        report's own date through as `report_date`, for live temporal operators
        (`operators.temporal.apply_temporal`) to window over - see
        TEMPORAL_SUBTREE_OPERATORS_PROMPT.md section 2. `date` already survives every fold's
        CSV unconditionally (it's a structural column, never a candidate feature - see
        `_NON_FEATURE_COLUMNS` above), so this is a pure passthrough, not new data.
        dropna() only checks the original 6 columns, deliberately excluding report_date, so a
        null report_date (not expected in practice - `date` is a non-null join key throughout
        the pipeline) can't silently start dropping fold rows that would previously have
        survived.
        """
        base_columns = ['fsym_id', 'sector_return_date', 'prev_month_return', 'prev_month_sector_return', 'feature', 'target']
        return df.select(
            col('fsym').alias('fsym_id'),
            col('target_date').alias('sector_return_date'),
            col('prev_month_return'),
            col('prev_month_sector_return'),
            col(feature).alias('feature'),
            col('monthly_return').alias('target'),
            col('date').alias('report_date'),
        ).dropna(subset=base_columns)

    def generate_training_testing_data(self, feature: str, sector: str = None) -> (DataFrame, DataFrame):
        """
        Same name/signature as the method it replaces (cell 19's evaluate_fitness_static calls
        this exact method for the gen==0 base-feature case), so cell 19 needs zero changes.
        `sector` is accepted but ignored: PreProcessing_test already scopes this whole fold to
        Finance via Utils(sector_filter='Finance'), so there's nothing left to filter by here.

        The eval frame is built over train_df UNION eval_df (not eval_df alone) so a live
        temporal atom's window/lag has real preceding history at the eval split's own boundary -
        see the lookback-buffer note in __init__. The caller (ga/engine.py's
        evaluate_fitness_static) trims the returned test frame back to rows with
        sector_return_date >= self.eval_start_date once temporal evaluation is done, so the
        extra train_df rows never reach fitness scoring - they exist only to feed lag/rolling
        computations correctly.
        """
        train = self._feature_frame(self.train_df, feature)
        eval_source = self.train_df.unionByName(self.eval_df)
        test = self._feature_frame(eval_source, feature)
        return train, test

    def get_true_test_frame(self, feature: str) -> DataFrame:
        """
        Same shape as _feature_frame(self.eval_df, ...), but always against the eval_label
        file's own content (test.csv for final-test folds) regardless of what self.eval_df is
        currently pointed at. Used by run_ga_for_fold() after a fold's GA search has already
        picked its winning individual, to build a fresh testing_cache for the one true,
        previously-untouched evaluation - never called during the generational search itself.

        Same lookback-buffer treatment as generate_training_testing_data: built over train_df
        (+ eval_df too, when eval_df is the separate inner-validation split a final-test fold
        carves out - has_inner_validation - since that split chronologically precedes
        true_eval_df/test.csv) UNION true_eval_df, so lag/rolling features have real history at
        the test split's boundary. Skipped for development folds, where eval_df IS true_eval_df
        (same object) - unioning it in again would just duplicate every row. The caller
        (ga/engine.py's evaluate_and_save_predictions, via _evaluate_on_true_test) trims back to
        sector_return_date >= self.true_eval_start_date before scoring.
        """
        buffer_source = self.train_df
        if self.has_inner_validation:
            buffer_source = buffer_source.unionByName(self.eval_df)
        true_eval_source = buffer_source.unionByName(self.true_eval_df)
        return self._feature_frame(true_eval_source, feature)
