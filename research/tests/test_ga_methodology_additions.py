"""
Unit tests for the pure-Python helpers GA_test.ipynb's "Methodology additions" cell (cell 22)
defines for ga_methodology_additions_prompt.md's items 1/3/4/5/6/7 - rank IC/mean IC/IC-IR,
baseline C's uniform-random expression sampler, the Holm-Bonferroni correction, and the
fold-block bootstrap on IC deltas. See CLAUDE.md's "Methodology additions" section (under
GA_test.ipynb) for the single-variant adaptation this notebook uses (no separate "v2" run to
diff against).

None of this needs Spark - cell 22 is plain Python/pandas/numpy/scipy. `helpers_ns` is
parametrized over BOTH the notebook (loaded via exec() from the notebook's own cell source, same
json-based cell-source-loader convention as legacy/tests/test_preprocessing_v1.py/test_ga_v1.py)
and the package port - now split across `superfeatures.evaluation.metrics` (rank IC, expression
composition, baseline-C sampler) and `superfeatures.analysis.significance` (Holm-Bonferroni,
block bootstrap), formerly one `superfeatures.ga.methodology` module (see
docs/RESTRUCTURING_TODO.md / the port plan) - every test below runs against both sources with no
change to the test bodies themselves, so a failure on only one side means that side has drifted
from the other.

Deliberately not tested here: run_ga_for_fold()'s new wiring (baseline B/C capture,
_evaluate_on_true_test, _run_random_search_baseline_c) or the 4 new driver cells that call these
helpers against real fold data - all of that needs a live cluster-shaped run (real
GAPreprocessing/GeneticAlgorithm1/Spark) to be meaningful, same rationale test_ga_v1.py already
documents for skipping run_ga_for_fold() itself.
"""
import json

import numpy as np
import pandas as pd
import pytest

from superfeatures.evaluation import metrics as _metrics_module
from superfeatures.analysis import significance as _significance_module

# Merged namespace standing in for the former single `superfeatures.ga.methodology` module -
# the split moved names, not behavior, so a plain dict union keeps every lookup below unchanged.
_package_helpers_ns = {**vars(_metrics_module), **vars(_significance_module)}

NOTEBOOK_PATH = "/home/bvail/temporal-superfeatures/legacy/notebooks/GA_test.ipynb"
HELPERS_CELL_INDEX = 22

_HELPER_NAMES = [
    "_count_leaf_features", "classify_leaf", "compute_monthly_ic", "summarize_ic",
    "_random_individual", "holm_bonferroni",
]
# block_bootstrap_ic_delta deliberately excluded from the shared notebook/package parametrization
# below - the package version was redesigned (one-sided, null-centered, within-fold block
# resampling) to match evaluation_framework.md, while GA_test.ipynb's own copy stays frozen at
# its original two-sided/fold-level design. See TestBlockBootstrapICDeltaNotebookOnly/
# TestBlockBootstrapICDeltaPackageOnly below, and analysis/significance.py's docstring.


def _load_cell_source(cell_index: int) -> str:
    with open(NOTEBOOK_PATH) as f:
        nb = json.load(f)
    return "".join(nb["cells"][cell_index]["source"])


@pytest.fixture(scope="session", params=["notebook", "package"])
def helpers_ns(request):
    if request.param == "notebook":
        ns = {}
        exec(_load_cell_source(HELPERS_CELL_INDEX), ns)
        return ns
    return {name: _package_helpers_ns[name] for name in _HELPER_NAMES}


class TestSumBuiltinShadowed:
    """
    Regression test for a real bug hit running GA_test.ipynb: cell 0 does
    `from pyspark.sql.functions import (..., sum, ...)` unqualified, which shadows Python's
    builtin sum() for the rest of the kernel (CLAUDE.md's documented "sum is shadowed globally"
    pitfall) - _count_leaf_features originally called bare sum() and crashed with
    PySparkTypeError: NOT_COLUMN_OR_STR the first time it ran against a nested expression in the
    real notebook. The `helpers_ns` fixture above execs cell 22 in a clean namespace where
    builtin sum() is still intact, so it can't catch this class of bug - this test pre-binds a
    pyspark-sum-like stand-in (raises on non-Column/str input, same as the real error) before
    exec'ing cell 22, reproducing the actual kernel condition.
    """

    @pytest.fixture()
    def shadowed_helpers_ns(self):
        def _fake_pyspark_sum(col):
            if not isinstance(col, str):
                raise TypeError(
                    "[NOT_COLUMN_OR_STR] Argument `col` should be a Column or str, "
                    f"got {type(col).__name__}."
                )
            return col
        ns = {"sum": _fake_pyspark_sum}
        exec(_load_cell_source(HELPERS_CELL_INDEX), ns)
        return ns

    def test_count_leaf_features_nested_expression(self, shadowed_helpers_ns):
        expr = (("a", "+", "b"), "-", "c")
        assert shadowed_helpers_ns["_count_leaf_features"](expr) == 3

    def test_count_leaf_features_flat_expression(self, shadowed_helpers_ns):
        expr = ("a", "+", "b", "-", "c", "*", "d")
        assert shadowed_helpers_ns["_count_leaf_features"](expr) == 4


class TestCountLeafFeatures:
    def test_single_leaf(self, helpers_ns):
        assert helpers_ns["_count_leaf_features"]("ff_roe") == 1

    def test_counts_repeated_leaf_as_two_occurrences(self, helpers_ns):
        # Distinct from _leaf_features (defined in cell 23, not under test here), which would
        # de-dupe this down to a single unique leaf.
        assert helpers_ns["_count_leaf_features"](("ff_roe", "+", "ff_roe")) == 2

    def test_nested_expression(self, helpers_ns):
        expr = (("a", "+", "b"), "-", "c")
        assert helpers_ns["_count_leaf_features"](expr) == 3

    def test_operators_not_counted(self, helpers_ns):
        assert helpers_ns["_count_leaf_features"](("a", "+", "b", "-", "c", "*", "d")) == 4


class TestClassifyLeaf:
    @pytest.mark.parametrize("suffix,expected", [
        ("_lag1", "lag1"),
        ("_delta1", "delta1"),
        ("_growth1", "growth1"),
        ("_mean3", "mean3"),
        ("_std3", "std3"),
    ])
    def test_temporal_suffixes(self, helpers_ns, suffix, expected):
        assert helpers_ns["classify_leaf"](f"ff_roe{suffix}") == expected

    def test_raw_feature(self, helpers_ns):
        assert helpers_ns["classify_leaf"]("ff_roe") == "raw"

    def test_raw_feature_with_unrelated_underscore(self, helpers_ns):
        assert helpers_ns["classify_leaf"]("ff_pbk_tang") == "raw"


class TestMonthlyIC:
    def test_perfect_correlation_per_month(self, helpers_ns, tmp_path):
        # Month 1: prediction/label perfectly rank-aligned (IC=1). Month 2: perfectly inverted
        # (IC=-1). Confirms per-month grouping (never pooled across months).
        rows = []
        for i in range(5):
            rows.append({"fsym_id": f"s{i}", "sector_return_date": "2020-01-01", "label": i, "prediction": i + 1})
        for i in range(5):
            rows.append({"fsym_id": f"s{i}", "sector_return_date": "2020-02-01", "label": i, "prediction": 4 - i})
        path = tmp_path / "predictions.csv"
        pd.DataFrame(rows).to_csv(path, index=False)

        monthly_ic = helpers_ns["compute_monthly_ic"](str(path))
        assert monthly_ic["2020-01-01"] == pytest.approx(1.0)
        assert monthly_ic["2020-02-01"] == pytest.approx(-1.0)

    def test_skips_constant_month(self, helpers_ns, tmp_path):
        rows = [{"fsym_id": f"s{i}", "sector_return_date": "2020-01-01", "label": 1.0, "prediction": i} for i in range(5)]
        path = tmp_path / "predictions.csv"
        pd.DataFrame(rows).to_csv(path, index=False)

        monthly_ic = helpers_ns["compute_monthly_ic"](str(path))
        assert len(monthly_ic) == 0

    def test_summarize_ic_mean_and_ir(self, helpers_ns):
        monthly_ic = pd.Series({"2020-01-01": 1.0, "2020-02-01": -1.0})
        mean_ic, ic_ir = helpers_ns["summarize_ic"](monthly_ic)
        assert mean_ic == pytest.approx(0.0)
        assert ic_ir == pytest.approx(0.0)

    def test_summarize_ic_single_month_ir_is_nan(self, helpers_ns):
        monthly_ic = pd.Series({"2020-01-01": 0.5})
        mean_ic, ic_ir = helpers_ns["summarize_ic"](monthly_ic)
        assert mean_ic == pytest.approx(0.5)
        assert np.isnan(ic_ir)


class TestRandomIndividual:
    def test_structurally_valid_across_many_seeds(self, helpers_ns):
        features = ["a", "b", "c", "d", "e"]
        for _ in range(500):
            individual = helpers_ns["_random_individual"](features, max_features=5)
            if isinstance(individual, str):
                assert individual in features
                continue
            assert isinstance(individual, tuple)
            assert len(individual) % 2 == 1
            leaves = individual[0::2]
            operators = individual[1::2]
            assert all(leaf in features for leaf in leaves)
            assert all(op in {"+", "-", "*", "/"} for op in operators)
            assert 1 <= len(leaves) <= 5

    def test_single_leaf_case_returns_bare_string(self, helpers_ns):
        # With max_features=1, n_leaves is always 1 - confirms the single-leaf branch returns a
        # bare string, not a length-1 tuple (matches the leaf-string base case every other
        # expression-walking helper in this codebase expects).
        for _ in range(20):
            individual = helpers_ns["_random_individual"](["only_feature"], max_features=1)
            assert individual == "only_feature"


class TestHolmBonferroni:
    def test_textbook_example(self, helpers_ns):
        # p=[0.01, 0.04, 0.03, 0.005], n=4, sorted ascending: 0.005, 0.01, 0.03, 0.04
        # adjusted (cumulative max of (n-rank)*p): 0.02, 0.03, 0.06, 0.06
        pvalues = [0.01, 0.04, 0.03, 0.005]
        adjusted = helpers_ns["holm_bonferroni"](pvalues)
        assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])

    def test_monotonic_and_never_below_raw(self, helpers_ns):
        pvalues = [0.001, 0.2, 0.03, 0.15, 0.02]
        adjusted = helpers_ns["holm_bonferroni"](pvalues)
        assert all(a >= p - 1e-12 for a, p in zip(adjusted, pvalues))
        assert all(a <= 1.0 for a in adjusted)

    def test_preserves_input_order(self, helpers_ns):
        pvalues = [0.5, 0.001, 0.2]
        adjusted = helpers_ns["holm_bonferroni"](pvalues)
        assert len(adjusted) == len(pvalues)
        # The smallest raw p-value's adjusted value should stay at index 1 (its original position).
        assert adjusted[1] == min(adjusted)


@pytest.fixture(scope="session")
def notebook_bootstrap_fn():
    ns = {}
    exec(_load_cell_source(HELPERS_CELL_INDEX), ns)
    return ns["block_bootstrap_ic_delta"]


class TestBlockBootstrapICDeltaNotebookOnly:
    """
    GA_test.ipynb's own block_bootstrap_ic_delta is frozen and untouched: two-sided, resamples
    whole fold identities with replacement (not within-fold month blocks), returns ci_low/ci_high.
    See TestBlockBootstrapICDeltaPackageOnly below for the redesigned package version's tests -
    the two no longer share a parametrized fixture/test body (see the _HELPER_NAMES comment
    above).
    """

    def test_known_zero_delta_ci_straddles_zero(self, notebook_bootstrap_fn):
        # Symmetric positive/negative deltas with no true effect - the 95% CI should contain 0.
        fold_month_deltas = {
            "fold_01": [0.05, -0.05, 0.03, -0.03],
            "fold_02": [-0.02, 0.02, 0.04, -0.04],
            "fold_03": [0.01, -0.01, 0.0, 0.0],
        }
        result = notebook_bootstrap_fn(fold_month_deltas, n_resamples=2000, seed=7)
        assert result["ci_low"] <= 0.0 <= result["ci_high"]
        assert result["observed_delta"] == pytest.approx(0.0, abs=1e-9)

    def test_known_positive_delta_significant(self, notebook_bootstrap_fn):
        fold_month_deltas = {
            "fold_01": [0.10, 0.12, 0.11],
            "fold_02": [0.09, 0.13, 0.10],
            "fold_03": [0.11, 0.10, 0.12],
        }
        result = notebook_bootstrap_fn(fold_month_deltas, n_resamples=2000, seed=7)
        assert result["observed_delta"] == pytest.approx(np.mean([v for vs in fold_month_deltas.values() for v in vs]))
        assert result["ci_low"] > 0.0
        assert result["p_value"] < 0.05


class TestBlockBootstrapICDeltaPackageOnly:
    """
    The package's redesigned bootstrap (see analysis/significance.py's docstring and
    evaluation_framework.md): one-sided, null-centered ("basic"/reflected bootstrap)
    lower_bound_95 + p_value, resampling contiguous month-blocks WITHIN each fold rather than
    whole fold identities.
    """

    def test_known_zero_delta_lower_bound_at_or_below_zero(self):
        # Symmetric positive/negative deltas with no true effect - the one-sided lower bound
        # should sit at or below zero and the null shouldn't be rejected.
        fold_month_deltas = {
            "fold_01": [0.05, -0.05, 0.03, -0.03, 0.02, -0.02],
            "fold_02": [-0.02, 0.02, 0.04, -0.04, 0.01, -0.01],
            "fold_03": [0.01, -0.01, 0.0, 0.0, 0.03, -0.03],
            "fold_04": [0.02, -0.02, 0.01, -0.01, 0.0, 0.0],
            "fold_05": [-0.01, 0.01, 0.02, -0.02, 0.03, -0.03],
        }
        result = _significance_module.block_bootstrap_ic_delta(
            fold_month_deltas, block_length=2, n_resamples=2000, seed=7)
        assert result["observed_delta"] == pytest.approx(0.0, abs=1e-9)
        assert result["lower_bound_95"] <= 0.0
        assert result["p_value"] > 0.05

    def test_known_positive_delta_significant(self):
        fold_month_deltas = {
            f"fold_{i:02d}": [0.10, 0.12, 0.11, 0.09, 0.13, 0.10] for i in range(5)
        }
        result = _significance_module.block_bootstrap_ic_delta(
            fold_month_deltas, block_length=2, n_resamples=2000, seed=7)
        assert result["observed_delta"] == pytest.approx(
            np.mean([v for vs in fold_month_deltas.values() for v in vs]))
        assert result["lower_bound_95"] > 0.0
        assert result["p_value"] < 0.05

    def test_below_min_folds_returns_insufficient_folds_flag(self):
        """
        Below the default min_folds=5, the resampling loop is skipped and
        lower_bound_95/p_value come back NaN with insufficient_folds=True, rather than an
        artificially exact-looking result from too little real data.
        """
        result = _significance_module.block_bootstrap_ic_delta(
            {"fold_01": [0.1, 0.2], "fold_02": [0.05, -0.1], "fold_03": [0.2, 0.1]}
        )
        assert result["insufficient_folds"] is True
        assert result["lower_bound_95"] != result["lower_bound_95"]  # NaN
        assert result["p_value"] != result["p_value"]  # NaN

        result_ok = _significance_module.block_bootstrap_ic_delta(
            {f"fold_{i:02d}": [0.1, -0.05, 0.2] for i in range(8)}, n_resamples=200
        )
        assert result_ok["insufficient_folds"] is False

    def test_short_fold_shorter_than_block_length_uses_whole_series(self):
        """
        A fold with fewer months than block_length has nothing shorter to draw blocks from -
        _moving_block_resample falls back to using its whole series as one block every
        replicate, rather than raising or silently truncating.
        """
        fold_month_deltas = {
            f"fold_{i:02d}": [0.1, 0.12] for i in range(5)  # 2 months, block_length default is 3
        }
        result = _significance_module.block_bootstrap_ic_delta(
            fold_month_deltas, n_resamples=200, seed=1)
        assert result["insufficient_folds"] is False
        assert not np.isnan(result["lower_bound_95"])
