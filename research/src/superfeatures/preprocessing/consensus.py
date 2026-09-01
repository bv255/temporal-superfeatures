"""
Ported from `research/PreProcessing_test.ipynb` cell 6 — see `docs/RESTRUCTURING_TODO.md`
and the port plan in `docs/RESEARCH_STRUCTURE.md`. Extracted mechanically from the notebook
cell source (not retyped) to avoid transcription drift; the notebook remains the frozen parity
reference. Only the import block was touched: names the cell relied on getting from the shared
notebook kernel namespace (star-imports in cell 0, e.g. `from pyspark.sql.types import *`) are
now imported explicitly here, since a standalone module has no such shared kernel state.
"""

"""
ConsensusFeatureSelector: fixed, reproducible, single-pass consensus feature selection.

Scores every candidate feature with three independent methods - Spearman |correlation|,
mutual information, and Random Forest permutation importance - against the FULL training
period in one pass (not multiple historical windows: an earlier draft of this used multiple
chronological windows with a "support frequency" rule, but that was dropped in favor of a
single full-period pass per explicit direction). A feature is retained only if at least 2 of
the 3 methods flag it as top-40% by percentile rank (TOP_PCT_THRESHOLD = 0.60). Retained features are then deduplicated
via hierarchical correlation clustering (average linkage, distance = 1 - |Spearman|, cut at a
fixed distance of 0.20, i.e. |corr| >= 0.80) keeping one representative per cluster. The final
list is uncapped by default (every representative is kept); a terminal cap can still be set via
the terminal_cap constructor param (--fast uses 15, see PipelineConfig.fast_terminal_cap).

Every threshold/parameter below is a fixed constant, not tuned or searched. This class does
not run or reference the genetic algorithm - it only ever produces a feature list for the GA
notebook to consume later, separately.
"""
from sklearn.inspection import permutation_importance
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import pandas as pd
import numpy as np
import random
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from pyspark.sql import DataFrame


class ConsensusResult:
    """Plain container holding every output artifact the consensus selection produces."""
    def __init__(self):
        self.final_features = []
        self.raw_scores_table = None   # feature, method, raw_score, percentile_rank
        self.consensus_table = None    # per retained feature: percentiles, consensus_score, diagnostics
        self.corr_matrix = None        # |Spearman| correlation matrix (retained features only)
        self.cluster_table = None      # feature, cluster_id, is_representative, representative
        self.rejection_log = None      # feature, stage, reason - every feature dropped anywhere
        self.summary_counts = {}


class ConsensusFeatureSelector:
    """Fixed methodology, single training-period pass. See module docstring above."""

    # --- fixed constants - none of these are tuned or searched ---
    TOP_PCT_THRESHOLD = 0.60              # top 40% => percentile rank >= 0.60
    MIN_METHODS_REQUIRED = 2              # retain iff flagged by >= 2 of the 3 methods
    CLUSTER_DISTANCE_THRESHOLD = 0.20     # |corr| >= 0.80  <=>  distance <= 0.20
    TERMINAL_CAP = None                   # no cap by default; --fast overrides via terminal_cap param
    RF_N_ESTIMATORS = 100                 # matches FeatureSelection.random_forest_feature_importance's convention
    PERMUTATION_N_REPEATS = 5
    PERMUTATION_SCORING = 'r2'

    def __init__(self, spark, random_seed: int = 42, terminal_cap: "int | None" = TERMINAL_CAP):
        self.spark = spark
        self.random_seed = random_seed
        self.TERMINAL_CAP = terminal_cap
        random.seed(random_seed)
        np.random.seed(random_seed)

    # ---------------------------------------------------------------- per-method scoring ----

    def score_spearman(self, df, features: list, target: str) -> dict:
        """Absolute Spearman rank correlation with the target, via Spark MLlib across every
        company in the fold - same core mechanism as FeatureSelection.spearman_correlation(),
        minus its company sampling (removed entirely - see class docstring: a fixed random seed
        looked reproducible but wasn't, since Spark doesn't guarantee stable row/partition order
        out of an unordered .collect(), and using the full fold is both simpler and no longer an
        approximation). Returns {feature: {'raw_score': abs_corr|None, 'signed_corr':
        corr|None, 'invalid_reason': str|None}}."""
        print("score_spearman: starting...")
        filled_df = df.fillna(0, subset=features + [target])

        assembler = VectorAssembler(inputCols=features + [target], outputCol="features_vec")
        assembled = assembler.transform(filled_df).select("features_vec")
        corr_array = Correlation.corr(assembled, "features_vec", method="spearman").head()[0].toArray()
        corr_df = pd.DataFrame(corr_array, columns=features + [target], index=features + [target])
        signed = corr_df[target].drop(target)

        scores = {}
        for feature in features:
            val = signed[feature]
            if pd.isna(val):
                scores[feature] = {'raw_score': None, 'signed_corr': None,
                                    'invalid_reason': 'undefined correlation (zero variance across the fold)'}
            else:
                scores[feature] = {'raw_score': abs(val), 'signed_corr': val, 'invalid_reason': None}
        n_invalid = len([v for v in scores.values() if v['raw_score'] is None])
        print(f"score_spearman: done. {n_invalid} features unscoreable.")
        return scores

    def score_mutual_information(self, df, features: list, target: str) -> dict:
        """Mutual information with the target, via sklearn across every company in the fold -
        same core mechanism as FeatureSelection.mutual_information(), minus its company sampling
        (removed - see score_spearman's docstring). random_state is fixed (self.random_seed) so
        the KSG estimator's internal tie-breaking noise is reproducible."""
        print("score_mutual_information: starting...")
        pandas_df = df.select(*features, target).toPandas()

        X = pandas_df[features].fillna(0)
        y = pandas_df[target].fillna(0)

        scores = {}
        valid_features = []
        for f in features:
            if X[f].nunique() <= 1:
                scores[f] = {'raw_score': None, 'invalid_reason': 'constant feature in this fold'}
            else:
                valid_features.append(f)

        if valid_features:
            mi_values = mutual_info_regression(X[valid_features], y, random_state=self.random_seed)
            for f, v in zip(valid_features, mi_values):
                scores[f] = {'raw_score': float(v), 'invalid_reason': None}

        print(f"score_mutual_information: done. {len(features) - len(valid_features)} features unscoreable.")
        return scores

    def score_rf_permutation(self, df, features: list, target: str, sector_col: str) -> dict:
        """Random Forest permutation importance with the target, across every company in every
        sector in the fold (not Finance-only, not sampled) - adapted from
        FeatureSelection.random_forest_feature_importance(), swapping Gini/MDI importance for
        permutation importance and the Finance-only filter for the full cross-sector universe.
        The old version sector-stratified-sampled to approximate proportional sector
        representation without a single-sector filter; using every company achieves that
        exactly rather than approximately, so removing the sampling (see score_spearman's
        docstring) loses nothing here. Fixed, documented RF params: n_estimators=100,
        random_state=<seed> (matches the existing convention, also fixes the permutation step's
        own column-shuffle randomness) - all other RandomForestRegressor params left at
        sklearn's documented defaults - none of this is tuned.

        n_jobs left at sklearn's default (1) on both the fit and the permutation step - DO NOT
        set this to -1 (or anything >1) without reading this whole note first; it's been tried
        twice (2026-08-27) and failed both times.

        n_jobs=-1 seemed safe in theory: this runs entirely on the driver (post-toPandas()), not
        inside any outer per-individual ThreadPoolExecutor the way the GA's local fit backend
        does, so there's no *oversubscription* risk, and sklearn guarantees identical
        trees/importances for a fixed random_state regardless of n_jobs - only wall-clock time
        should change. In practice, sklearn/joblib's process-based (loky) backend for n_jobs>1
        memory-maps large arrays to local temp files to share them with worker processes, and
        that crashed real runs with OSError: No space left on device - not on a YARN worker node,
        on bialobog itself (the driver), whose root disk (also home to /tmp) sits around 83% used
        (26GB free of 152GB), a shared login node that's also a chronically-near-full HDFS
        DataNode. First attempt: crashed after 7 prior successful calls in the same run (all 5 dev
        folds + final_test/fold_01-02), so it isn't a guaranteed-immediate failure. Re-enabled
        anyway on the theory that run_preprocessing.py's --only-folds flag makes a crash cheap to
        recover from (reuses the HDFS checkpoint, skips the ~40-70min global setup) - second
        attempt then failed on the very first fold it touched, zero progress. Two-for-two against
        it in real runs is enough - the speedup isn't worth the retries it costs. If bialobog's
        own disk pressure is ever actually resolved, this is the thing to revisit; until then,
        leave n_jobs unset here."""
        print("score_rf_permutation: starting...")
        pandas_df = df.select(*features, target).toPandas()

        X = pandas_df[features].fillna(0)
        y = pandas_df[target].fillna(0)

        scores = {f: {'raw_score': None, 'invalid_reason': None} for f in features}
        constant_features = [f for f in features if X[f].nunique() <= 1]
        for f in constant_features:
            scores[f]['invalid_reason'] = 'constant feature in this fold'
        valid_features = [f for f in features if f not in constant_features]

        if valid_features:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X[valid_features])
            rf = RandomForestRegressor(n_estimators=self.RF_N_ESTIMATORS, random_state=self.random_seed)
            rf.fit(X_scaled, y)
            perm = permutation_importance(
                rf, X_scaled, y,
                n_repeats=self.PERMUTATION_N_REPEATS,
                random_state=self.random_seed,
                scoring=self.PERMUTATION_SCORING,
            )
            for f, v in zip(valid_features, perm.importances_mean):
                scores[f]['raw_score'] = float(v)

        print(f"score_rf_permutation: done. {len(constant_features)} features unscoreable.")
        return scores

    # ---------------------------------------------------------------- ranking / consensus ----

    @staticmethod
    def to_percentile_ranks(scores: dict) -> dict:
        """Converts {feature: {'raw_score': v, ...}} into {feature: percentile_rank}, where a
        higher percentile means the feature is more important under this method. Features with
        raw_score=None (invalid/unscoreable) are excluded here - they're already recorded via
        each score_*() call's 'invalid_reason'."""
        valid = {f: v['raw_score'] for f, v in scores.items() if v['raw_score'] is not None}
        if not valid:
            return {}
        return pd.Series(valid).rank(pct=True).to_dict()

    def apply_consensus_rule(self, percentiles_by_method: dict, all_features: list) -> tuple:
        """A method 'flags' a feature when its percentile rank >= TOP_PCT_THRESHOLD (top 40%).
        Retained iff flagged by at least MIN_METHODS_REQUIRED (2) of the 3 methods. Returns
        (retained_features: list[str], rejection_log: list[dict])."""
        flagged_by = {}
        for method, percentiles in percentiles_by_method.items():
            for feature, pct in percentiles.items():
                if pct >= self.TOP_PCT_THRESHOLD:
                    flagged_by.setdefault(feature, set()).add(method)

        retained, rejection_log = [], []
        for feature in all_features:
            methods_flagging = flagged_by.get(feature, set())
            if len(methods_flagging) >= self.MIN_METHODS_REQUIRED:
                retained.append(feature)
            else:
                scored_by = [m for m in percentiles_by_method if feature in percentiles_by_method[m]]
                if not scored_by:
                    reason = "not scoreable by any method (invalid/constant in every sampled subset)"
                else:
                    names = ', '.join(sorted(methods_flagging)) if methods_flagging else 'none'
                    reason = f"flagged by {len(methods_flagging)}/3 methods ({names})"
                rejection_log.append({'feature': feature, 'stage': 'consensus_rule', 'reason': reason})
        return retained, rejection_log

    def build_consensus_table(self, retained_features, percentiles_by_method, signed_spearman,
                               missingness, company_coverage, sector_coverage) -> pd.DataFrame:
        """Per retained feature: percentile rank under each method, the equally-weighted
        consensus score C_j (mean of whichever method percentiles are available - a feature
        retained by exactly 2/3 methods may not have a percentile from the 3rd), which
        method(s) flagged it, Spearman sign, and the coverage/missingness diagnostics."""
        rows = []
        for feature in retained_features:
            p_spearman = percentiles_by_method['spearman'].get(feature)
            p_mi = percentiles_by_method['mutual_information'].get(feature)
            p_rf = percentiles_by_method['rf_permutation'].get(feature)
            available = [p for p in (p_spearman, p_mi, p_rf) if p is not None]
            if available:
                _total = 0.0
                for _v in available:
                    _total += _v
                consensus_score = _total / len(available)
            else:
                consensus_score = None
            flagged_by = [m for m, pcts in percentiles_by_method.items()
                          if pcts.get(feature, 0) >= self.TOP_PCT_THRESHOLD]
            signed = signed_spearman.get(feature)
            rows.append({
                'feature': feature,
                'percentile_spearman': p_spearman,
                'percentile_mi': p_mi,
                'percentile_rf_permutation': p_rf,
                'consensus_score': consensus_score,
                'flagged_by_methods': ', '.join(flagged_by),
                'n_methods_flagged': len(flagged_by),
                'spearman_sign': (1 if (signed or 0) > 0 else -1 if (signed or 0) < 0 else 0),
                'missingness': missingness.get(feature),
                'company_coverage': company_coverage.get(feature),
                'sector_coverage': sector_coverage.get(feature),
            })
        return pd.DataFrame(rows).sort_values('consensus_score', ascending=False).reset_index(drop=True)

    # ---------------------------------------------------------------- clustering ----

    def cluster_features(self, df, retained_features: list) -> tuple:
        """Hierarchical clustering (average linkage) on 1 - |Spearman correlation| distance,
        cut at a FIXED distance of CLUSTER_DISTANCE_THRESHOLD (0.20, i.e. |corr| >= 0.80) - not
        tuned or searched. Correlation matrix computed across every company in the fold (no
        sampling - see score_spearman's docstring), replacing
        FeatureSelection.remove_high_corr_features_based_on_mi()'s greedy pairwise removal (and
        its company sampling) with proper hierarchical clustering over the full fold. Returns
        (cluster_assignments: dict[feature, cluster_id], corr_matrix: pd.DataFrame)."""
        print("cluster_features: starting...")
        pandas_df = df.select(*retained_features).fillna(0).toPandas()

        corr_matrix = pandas_df.corr(method='spearman').abs()

        # A feature that's constant across the whole fold has undefined correlation with
        # everything, including itself - pandas.corr() reports that as NaN rather than 1.0 on
        # the diagonal. Left unhandled, that NaN flows into the distance matrix and linkage()
        # either errors unclearly or silently produces a corrupted clustering. Give each such
        # feature its own singleton cluster instead of guessing a distance for it.
        unclusterable = [f for f in retained_features if pd.isna(corr_matrix.loc[f, f])]
        clusterable_features = [f for f in retained_features if f not in unclusterable]

        cluster_assignments = {}
        next_cluster_id = 1

        # A single clusterable feature has nothing to pair a distance against - squareform()
        # on a 1x1 distance matrix returns a length-0 condensed array, which linkage() rejects
        # ("cannot be determined on an empty distance matrix"). Give it its own singleton
        # cluster directly, same treatment as the NaN-diagonal "unclusterable" case below,
        # rather than calling linkage()/squareform() with nothing to cluster.
        if len(clusterable_features) == 1:
            cluster_assignments[clusterable_features[0]] = next_cluster_id
            next_cluster_id += 1
        elif clusterable_features:
            sub_corr = corr_matrix.loc[clusterable_features, clusterable_features]
            distance_matrix = 1 - sub_corr.values
            np.fill_diagonal(distance_matrix, 0)  # numerical noise guard - diagonal is always |corr|=1
            condensed = squareform(distance_matrix, checks=False)

            Z = linkage(condensed, method='average')
            cluster_ids = fcluster(Z, t=self.CLUSTER_DISTANCE_THRESHOLD, criterion='distance')

            for feature, cluster_id in zip(clusterable_features, cluster_ids):
                cluster_assignments[feature] = int(cluster_id)
            next_cluster_id = int(max(cluster_ids)) + 1

        for feature in unclusterable:
            cluster_assignments[feature] = next_cluster_id
            next_cluster_id += 1

        unclusterable_note = (
            f", {len(unclusterable)} features excluded from clustering (constant across the "
            f"fold, each given its own singleton cluster)" if unclusterable else ""
        )
        print(f"cluster_features: done. {len(retained_features)} features -> "
              f"{len(set(cluster_assignments.values()))} clusters{unclusterable_note}.")
        return cluster_assignments, corr_matrix

    def select_representatives(self, cluster_assignments: dict, consensus_table: pd.DataFrame) -> tuple:
        """Within every cluster, keep the feature with the highest consensus_score. Ties broken
        in this fixed order: lower missingness -> greater company_coverage -> greater
        sector_coverage -> higher percentile_spearman (proxy for median |Spearman|) ->
        alphabetical feature name, to guarantee reproducibility."""
        table = consensus_table.set_index('feature')
        clusters = {}
        for feature, cluster_id in cluster_assignments.items():
            clusters.setdefault(cluster_id, []).append(feature)

        def sort_key(f):
            row = table.loc[f]
            return (
                -(row['consensus_score'] if pd.notna(row['consensus_score']) else -1),
                row['missingness'] if pd.notna(row['missingness']) else float('inf'),
                -(row['company_coverage'] if pd.notna(row['company_coverage']) else 0),
                -(row['sector_coverage'] if pd.notna(row['sector_coverage']) else 0),
                -(row['percentile_spearman'] if pd.notna(row['percentile_spearman']) else 0),
                f,
            )

        representatives, cluster_rows = [], []
        for cluster_id, members in clusters.items():
            ordered = sorted(members, key=sort_key)
            representative = ordered[0]
            representatives.append(representative)
            for f in members:
                cluster_rows.append({
                    'feature': f,
                    'cluster_id': int(cluster_id),
                    'is_representative': f == representative,
                    'representative': representative,
                    'consensus_score': table.loc[f, 'consensus_score'],
                })
        cluster_table = pd.DataFrame(cluster_rows).sort_values(['cluster_id', 'feature']).reset_index(drop=True)
        return representatives, cluster_table

    def apply_terminal_cap(self, representatives: list, consensus_table: pd.DataFrame) -> list:
        """If TERMINAL_CAP is None (the default), no cap is applied - every representative is
        kept. Otherwise, if <= TERMINAL_CAP representatives remain, keep all of them; if more,
        keep the TERMINAL_CAP with the highest consensus_score."""
        if self.TERMINAL_CAP is None or len(representatives) <= self.TERMINAL_CAP:
            return representatives
        table = consensus_table.set_index('feature')
        ranked = sorted(representatives, key=lambda f: -(table.loc[f, 'consensus_score'] or -1))
        return ranked[:self.TERMINAL_CAP]

    # ---------------------------------------------------------------- diagnostics ----

    def _compute_diagnostics(self, df, feature_columns: list, sector_col: str) -> tuple:
        """Missingness (fraction null), company coverage (fraction of companies with >=1
        non-null value anywhere in their history), and sector coverage (fraction of sectors
        with >=1 non-null value), per feature - computed across every company in the fold (no
        sampling - see score_spearman's docstring; these are tie-breakers/display diagnostics,
        but there's no reason to approximate them once the scoring methods themselves don't).
        Uses a small, FIXED number of Spark aggregations regardless of feature count (one row
        per company/sector flagging presence, then a single sum) rather than looping per
        feature and re-scanning the table each time - the per-company loop in the original
        Utils.remove_static_features() was exactly this anti-pattern and was fixed earlier for
        the same reason."""
        print("_compute_diagnostics: starting...")
        df = df.cache()

        total_rows = df.count()
        total_companies = df.select('fsym').distinct().count()
        total_sectors = df.select(sector_col).distinct().count()

        # Chunk the feature list rather than building one Spark expression list spanning all
        # ~287 columns at once - Spark's whole-stage codegen compiles a single Java method per
        # stage, capped at 64KB bytecode, and compilation time can blow up badly well before
        # that many columns (this was the actual cause of _compute_diagnostics hanging, not the
        # row count - the sampling above alone didn't fix it). Column count, not row count, was
        # the problem here.
        CHUNK_SIZE = 30

        def _chunks(cols):
            for i in range(0, len(cols), CHUNK_SIZE):
                yield cols[i:i + CHUNK_SIZE]

        missingness = {}
        for chunk in _chunks(feature_columns):
            chunk_counts = df.select(
                [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in chunk]
            ).collect()[0].asDict()
            for c in chunk:
                missingness[c] = chunk_counts[c] / total_rows

        company_coverage = {}
        for chunk in _chunks(feature_columns):
            company_presence = df.groupBy('fsym').agg(
                *[F.max(F.col(c).isNotNull().cast('int')).alias(c) for c in chunk]
            )
            chunk_counts = company_presence.select(
                [F.sum(F.col(c)).alias(c) for c in chunk]
            ).collect()[0].asDict()
            for c in chunk:
                company_coverage[c] = chunk_counts[c] / total_companies

        sector_coverage = {}
        for chunk in _chunks(feature_columns):
            sector_presence = df.groupBy(sector_col).agg(
                *[F.max(F.col(c).isNotNull().cast('int')).alias(c) for c in chunk]
            )
            chunk_counts = sector_presence.select(
                [F.sum(F.col(c)).alias(c) for c in chunk]
            ).collect()[0].asDict()
            for c in chunk:
                sector_coverage[c] = chunk_counts[c] / total_sectors

        print(f"_compute_diagnostics: done. ({total_companies} companies)")
        return missingness, company_coverage, sector_coverage

    # ---------------------------------------------------------------- orchestration ----

    def run(self, experiment_1_df, feature_columns: list, target: str = 'monthly_return',
            sector_col: str = 'factset_sector_desc') -> ConsensusResult:
        print("=== ConsensusFeatureSelector: starting (single-pass, full training period) ===")
        print(f"run params (fixed, not tuned): top_pct_threshold={self.TOP_PCT_THRESHOLD}, "
              f"min_methods_required={self.MIN_METHODS_REQUIRED}, "
              f"cluster_distance_threshold={self.CLUSTER_DISTANCE_THRESHOLD}, "
              f"terminal_cap={self.TERMINAL_CAP}, "
              f"rf_n_estimators={self.RF_N_ESTIMATORS}, permutation_n_repeats={self.PERMUTATION_N_REPEATS}, "
              f"random_seed={self.random_seed}")

        result = ConsensusResult()
        initial_count = len(feature_columns)
        experiment_1_df = experiment_1_df.cache()

        # --- Stage 1: score every feature with all 3 independent methods ---
        raw_scores_by_method = {
            'spearman': self.score_spearman(experiment_1_df, feature_columns, target),
            'mutual_information': self.score_mutual_information(experiment_1_df, feature_columns, target),
            'rf_permutation': self.score_rf_permutation(experiment_1_df, feature_columns, target, sector_col),
        }
        percentiles_by_method = {name: self.to_percentile_ranks(scores) for name, scores in raw_scores_by_method.items()}
        signed_spearman = {f: v['signed_corr'] for f, v in raw_scores_by_method['spearman'].items()}

        raw_rows, invalid_rejections = [], []
        for method, scores in raw_scores_by_method.items():
            for feature, v in scores.items():
                raw_rows.append({
                    'feature': feature, 'method': method, 'raw_score': v['raw_score'],
                    'percentile_rank': percentiles_by_method[method].get(feature),
                })
                if v['raw_score'] is None:
                    invalid_rejections.append({'feature': feature, 'stage': f'score_{method}', 'reason': v['invalid_reason']})
        result.raw_scores_table = pd.DataFrame(raw_rows)

        # --- Stage 2: coverage/missingness diagnostics (computed once, on the full window) ---
        missingness, company_coverage, sector_coverage = self._compute_diagnostics(
            experiment_1_df, feature_columns, sector_col
        )

        # --- Stage 3: consensus rule (retain iff flagged top-40% by >= 2 of 3 methods) ---
        retained_features, consensus_rejections = self.apply_consensus_rule(percentiles_by_method, feature_columns)
        consensus_passing_count = len(retained_features)

        result.consensus_table = self.build_consensus_table(
            retained_features, percentiles_by_method, signed_spearman,
            missingness, company_coverage, sector_coverage,
        )

        # --- Stage 4: correlation clustering on the consensus survivors only ---
        cluster_assignments, result.corr_matrix = self.cluster_features(experiment_1_df, retained_features)
        cluster_count = len(set(cluster_assignments.values()))

        # --- Stage 5: one representative per cluster ---
        representatives, result.cluster_table = self.select_representatives(cluster_assignments, result.consensus_table)
        representative_count = len(representatives)
        cluster_rejections = [
            {'feature': f, 'stage': 'cluster_representative_selection',
             'reason': f"member of cluster {cluster_assignments[f]}, not the highest-consensus representative"}
            for f in retained_features if f not in representatives
        ]

        # --- Stage 6: terminal cap (none by default; see terminal_cap constructor param) ---
        final_features = self.apply_terminal_cap(representatives, result.consensus_table)
        cap_rejections = [
            {'feature': f, 'stage': 'terminal_cap', 'reason': f'ranked below top {self.TERMINAL_CAP} by consensus_score'}
            for f in representatives if f not in final_features
        ]

        result.final_features = final_features
        result.rejection_log = pd.DataFrame(invalid_rejections + consensus_rejections + cluster_rejections + cap_rejections)
        result.summary_counts = {
            'initial_feature_count': initial_count,
            'consensus_passing_count': consensus_passing_count,
            'cluster_count': cluster_count,
            'representative_count': representative_count,
            'final_count_after_cap': len(final_features),
        }

        print("=== ConsensusFeatureSelector: stage-by-stage summary ===")
        for k, v in result.summary_counts.items():
            print(f"  {k}: {v}")
        print(f"  final_features ({len(final_features)}): {final_features}")
        print("=== ConsensusFeatureSelector: done ===")
        return result
