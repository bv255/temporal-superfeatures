"""
Unit tests for the pure-Python logic that's new or changed in compare_metrics.ipynb (the real
v2-vs-v3 head-to-head comparison notebook, reading GA_test.ipynb's/GA_test_v2.ipynb's output) -
see the "Close the gaps between compare_metrics.ipynb and GA_v2_v3_handover.md" plan for the
audit this followed. Covers:

- The one-tailed rewrite of block_bootstrap_ic_delta (cell 3) - GA_v2_v3_handover.md Sec 4.1/4.3
  require a one-tailed test (5th-percentile bound, p = P(delta <= 0)), not the two-sided
  version GA_test.ipynb's own copy of this helper still uses.
- matched_month_lift_deltas (cell 3) - the new comparison-#6 (DiD attribution) helper.
- The Delta (effect-size threshold) derivation formula (cell 11).
- The H1 success-criteria verdict logic, evaluate_h1() (cell 17).
- The 6-pair Holm-Bonferroni family (cell 3's holm_bonferroni, unchanged, sanity-checked at the
  new family size).

None of this needs Spark - every cell under test is plain Python/pandas/numpy/scipy, loaded via
exec() from the notebook's own cell source (same json-based cell-source-loader convention as
test_ga_methodology_additions.py/test_preprocessing_v1.py/test_ga_v1.py), so these tests
exercise the real notebook code with no drift risk.

Deliberately not tested here: Item 6 (cell 5) and Item 7's v3-side fields (cell 13) - those were
already exercised transitively by test_ga_methodology_additions.py, since they only read fields
GA_test.ipynb's own (already-tested) helpers computed; and anything that reads real
ga_test_outputs/ga_test_v2_outputs fold_result.json files from disk (cells 1, 5, 9, 13, 15) -
those need a live end-to-end notebook run to be meaningful, same "needs a live cluster-shaped
run" rationale test_ga_v1.py/test_ga_methodology_additions.py already document for excluding
run_ga_for_fold() itself.
"""
import json
import random

import numpy as np
import pandas as pd
import pytest

NOTEBOOK_PATH = "/home/bvail/temporal-superfeatures/legacy/notebooks/compare_metrics.ipynb"
HELPERS_CELL_INDEX = 3
DELTA_DERIVATION_CELL_INDEX = 11
H1_VERDICT_CELL_INDEX = 17


def _load_cell_source(cell_index: int) -> str:
    with open(NOTEBOOK_PATH) as f:
        nb = json.load(f)
    return "".join(nb["cells"][cell_index]["source"])


@pytest.fixture(scope="session")
def helpers_ns():
    # cell 3 relies on numpy/pandas already being imported by cell 1 in the real notebook's
    # shared kernel namespace (it only imports `random` itself) - same "import order matters"
    # pattern test_preprocessing_v1.py/test_ga_v1.py already document for other notebooks.
    ns = {"np": np, "pd": pd}
    exec(_load_cell_source(HELPERS_CELL_INDEX), ns)
    return ns


# ---------------------------------------------------------------------------
# block_bootstrap_ic_delta - one-tailed rewrite
# ---------------------------------------------------------------------------

def _reference_one_tailed_p_value(fold_month_deltas, n_resamples=3000, seed=42):
    """
    Independently reimplements the one-tailed p-value (mean of boot_deltas <= 0) using the exact
    same resampling algorithm/seed as block_bootstrap_ic_delta - a regression pin against
    silently reintroducing the old two-sided `2 * min(p_le, p_ge)` formula, which would no
    longer match this direct computation.
    """
    rng = random.Random(seed)
    fold_names = list(fold_month_deltas.keys())
    boot_deltas = []
    for _ in range(n_resamples):
        resampled_folds = [rng.choice(fold_names) for _ in range(len(fold_names))]
        pooled = [d for f in resampled_folds for d in fold_month_deltas[f]]
        boot_deltas.append(float(np.mean(pooled)) if pooled else float('nan'))
    return float(np.mean(np.array(boot_deltas) <= 0))


class TestBlockBootstrapICDeltaOneTailed:
    def test_returns_one_tailed_fields_not_two_sided(self, helpers_ns):
        """Regression check: the two-sided ci_low/ci_high field names must be gone."""
        fold_month_deltas = {"fold_01": [0.05, 0.06, 0.04], "fold_02": [0.05, 0.07]}
        result = helpers_ns["block_bootstrap_ic_delta"](fold_month_deltas)
        assert set(result.keys()) == {"observed_delta", "ci_5th_percentile", "p_value", "n_resamples"}

    def test_p_value_matches_reference_one_tailed_computation(self, helpers_ns):
        """
        Pins p_value == mean(boot_deltas <= 0) exactly - the old implementation returned
        `2 * min(P(delta<=0), P(delta>=0))`, which would NOT equal this reference for a
        skewed (non-symmetric-around-zero) input like the one below.
        """
        fold_month_deltas = {
            "fold_01": [0.05, 0.06, -0.01, 0.04, 0.03],
            "fold_02": [0.05, 0.07, 0.02],
            "fold_03": [-0.02, 0.05, 0.06, 0.04],
        }
        result = helpers_ns["block_bootstrap_ic_delta"](fold_month_deltas)
        expected_p = _reference_one_tailed_p_value(fold_month_deltas)
        assert result["p_value"] == pytest.approx(expected_p)

    def test_clearly_positive_delta_gives_low_p_and_positive_ci_bound(self, helpers_ns):
        fold_month_deltas = {
            "fold_01": [0.10, 0.12, 0.11, 0.09],
            "fold_02": [0.15, 0.13, 0.14],
            "fold_03": [0.08, 0.09, 0.10],
        }
        result = helpers_ns["block_bootstrap_ic_delta"](fold_month_deltas)
        assert result["ci_5th_percentile"] > 0
        assert result["p_value"] < 0.05

    def test_clearly_negative_delta_gives_high_p_and_negative_ci_bound(self, helpers_ns):
        fold_month_deltas = {
            "fold_01": [-0.10, -0.12, -0.11, -0.09],
            "fold_02": [-0.15, -0.13, -0.14],
            "fold_03": [-0.08, -0.09, -0.10],
        }
        result = helpers_ns["block_bootstrap_ic_delta"](fold_month_deltas)
        assert result["ci_5th_percentile"] < 0
        assert result["p_value"] > 0.9

    def test_observed_delta_is_unresampled_mean(self, helpers_ns):
        fold_month_deltas = {"fold_01": [0.10, 0.20], "fold_02": [0.30]}
        result = helpers_ns["block_bootstrap_ic_delta"](fold_month_deltas)
        assert result["observed_delta"] == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# matched_month_lift_deltas - comparison #6's DiD helper
# ---------------------------------------------------------------------------

class TestMatchedMonthLiftDeltas:
    def test_single_fold_single_month_exact_value(self, helpers_ns):
        v3_results = {"fold_01": {"winner_monthly_ic": {"m1": 0.30}, "baseline_c_monthly_ic": {"m1": 0.10}}}
        v2_results = {"fold_01": {"winner_monthly_ic": {"m1": 0.05}, "baseline_c_monthly_ic": {"m1": 0.02}}}

        result = helpers_ns["matched_month_lift_deltas"](v3_results, v2_results, ["fold_01"])

        # v3_lift[m1] = 0.30 - 0.10 = 0.20, v2_lift[m1] = 0.05 - 0.02 = 0.03, DiD = 0.17
        assert result["fold_01"] == pytest.approx([0.17])

    def test_drops_months_not_common_to_both_variants_lift_series(self, helpers_ns):
        v3_results = {"fold_01": {
            "winner_monthly_ic": {"m1": 0.3, "m2": 0.4, "m3": 0.5},
            "baseline_c_monthly_ic": {"m1": 0.1, "m2": 0.1, "m3": 0.1},
        }}
        v2_results = {"fold_01": {
            "winner_monthly_ic": {"m1": 0.05, "m2": 0.05, "m4": 0.05},
            "baseline_c_monthly_ic": {"m1": 0.02, "m2": 0.02, "m4": 0.02},
        }}

        result = helpers_ns["matched_month_lift_deltas"](v3_results, v2_results, ["fold_01"])

        # v3_lift = {m1: 0.2, m2: 0.3, m3: 0.4}; v2_lift = {m1: 0.03, m2: 0.03, m4: 0.03}.
        # m3 (v3-only) and m4 (v2-only) must be dropped - only m1/m2 are common to both lifts.
        assert len(result["fold_01"]) == 2
        assert sorted(round(v, 6) for v in result["fold_01"]) == sorted([0.17, 0.27])

    def test_fold_missing_from_one_variant_is_skipped(self, helpers_ns):
        v3_results = {"fold_01": {"winner_monthly_ic": {"m1": 0.3}, "baseline_c_monthly_ic": {"m1": 0.1}}}
        v2_results = {}  # fold_01 never ran (or baseline C didn't) on the v2 side

        result = helpers_ns["matched_month_lift_deltas"](v3_results, v2_results, ["fold_01"])

        assert result == {}

    def test_two_folds_both_populated(self, helpers_ns):
        v3_results = {
            "fold_01": {"winner_monthly_ic": {"m1": 0.30}, "baseline_c_monthly_ic": {"m1": 0.10}},
            "fold_02": {"winner_monthly_ic": {"m5": 0.20}, "baseline_c_monthly_ic": {"m5": 0.05}},
        }
        v2_results = {
            "fold_01": {"winner_monthly_ic": {"m1": 0.05}, "baseline_c_monthly_ic": {"m1": 0.02}},
            "fold_02": {"winner_monthly_ic": {"m5": 0.04}, "baseline_c_monthly_ic": {"m5": 0.01}},
        }

        result = helpers_ns["matched_month_lift_deltas"](v3_results, v2_results, ["fold_01", "fold_02"])

        assert set(result.keys()) == {"fold_01", "fold_02"}
        assert result["fold_01"] == pytest.approx([0.17])
        assert result["fold_02"] == pytest.approx([(0.20 - 0.05) - (0.04 - 0.01)])


# ---------------------------------------------------------------------------
# Holm-Bonferroni, sanity-checked at the new 6-comparison family size
# ---------------------------------------------------------------------------

class TestHolmBonferroniSixPairFamily:
    def test_six_pvalues_adjusted_are_monotonic_and_geq_raw(self, helpers_ns):
        raw = [0.001, 0.01, 0.02, 0.03, 0.04, 0.20]
        adjusted = helpers_ns["holm_bonferroni"](raw)

        assert len(adjusted) == 6
        for r, a in zip(raw, adjusted):
            assert a >= r
        # Ascending-sorted adjusted values must be non-decreasing (Holm's running-max property).
        order = sorted(range(6), key=lambda i: raw[i])
        sorted_adjusted = [adjusted[i] for i in order]
        assert sorted_adjusted == sorted(sorted_adjusted)

    def test_six_identical_pvalues(self, helpers_ns):
        raw = [0.01] * 6
        adjusted = helpers_ns["holm_bonferroni"](raw)
        # Holm step-down on 6 identical p-values: multipliers are 6,5,4,3,2,1 with a running max,
        # so every adjusted value should equal 6 * 0.01 = 0.06 (running max never decreases).
        assert adjusted == pytest.approx([0.06] * 6)


# ---------------------------------------------------------------------------
# Delta (effect-size threshold) derivation formula - cell 11
# ---------------------------------------------------------------------------

class TestDeltaDerivation:
    def test_delta_formula_matches_hand_computation(self, tmp_path):
        v2_dev_mean_ics = {"fold_01": 0.10, "fold_02": 0.20, "fold_03": 0.15}
        matched_folds = ["fold_01", "fold_02", "fold_03", "fold_04", "fold_05"]

        ns = {
            "np": np,
            "json": json,
            "COMPARISON_OUTPUT_DIR": str(tmp_path),
            "v2_dev_mean_ics": v2_dev_mean_ics,
            "matched_folds": matched_folds,
        }
        exec(_load_cell_source(DELTA_DERIVATION_CELL_INDEX), ns)

        # s = std([0.10, 0.20, 0.15], ddof=1) = 0.05; n = len(matched_folds) = 5
        expected_s = 0.05
        expected_n = 5
        expected_delta = 2 * (expected_s / np.sqrt(expected_n))

        assert ns["s"] == pytest.approx(expected_s)
        assert ns["n"] == expected_n
        assert ns["DELTA"] == pytest.approx(expected_delta)

        written = json.loads((tmp_path / "delta_derivation.json").read_text())
        assert written["delta"] == pytest.approx(expected_delta)
        assert written["n"] == expected_n

    def test_delta_not_available_with_fewer_than_two_dev_folds(self, tmp_path):
        ns = {
            "np": np,
            "json": json,
            "COMPARISON_OUTPUT_DIR": str(tmp_path),
            "v2_dev_mean_ics": {"fold_01": 0.10},
            "matched_folds": ["fold_01"],
        }
        exec(_load_cell_source(DELTA_DERIVATION_CELL_INDEX), ns)

        assert ns["DELTA"] is None
        assert not (tmp_path / "delta_derivation.json").exists()

    def test_delta_not_available_with_zero_dev_folds(self, tmp_path):
        ns = {
            "np": np,
            "json": json,
            "COMPARISON_OUTPUT_DIR": str(tmp_path),
            "v2_dev_mean_ics": {},
            "matched_folds": [],
        }
        exec(_load_cell_source(DELTA_DERIVATION_CELL_INDEX), ns)

        assert ns["DELTA"] is None
        assert not (tmp_path / "delta_derivation.json").exists()


# ---------------------------------------------------------------------------
# H1 success-criteria verdict - cell 17's evaluate_h1()
# ---------------------------------------------------------------------------

def _composition_df(ic_deltas_by_fold: dict) -> pd.DataFrame:
    rows = [{"fold_name": name, "mean_ic_delta_vs_v2": delta} for name, delta in ic_deltas_by_fold.items()]
    rows.append({"fold_name": "AGGREGATE (mean)", "mean_ic_delta_vs_v2": np.mean(list(ic_deltas_by_fold.values()))})
    return pd.DataFrame(rows)


def _pairwise_df(v3_vs_v2: dict, did: dict) -> pd.DataFrame:
    rows = [
        {"comparison": "v3_vs_v2", **v3_vs_v2},
        {"comparison": "v3_v2_did_attribution", **did},
    ]
    return pd.DataFrame(rows)


class TestH1Verdict:
    def _run(self, pairwise_df, composition_df, delta):
        # matched_folds is only referenced by the cell's own driver code (the h1_record dict,
        # outside evaluate_h1() itself) - not exercised by these tests, just needs to exist.
        ns = {"pd": pd, "json": json, "COMPARISON_OUTPUT_DIR": "/tmp",
              "pairwise_df": pairwise_df, "composition_df": composition_df, "DELTA": delta,
              "matched_folds": ["fold_01", "fold_02"]}
        exec(_load_cell_source(H1_VERDICT_CELL_INDEX), ns)
        return ns["h1_conditions"], ns["h1_verdict"]

    def test_all_conditions_true_gives_supported(self):
        composition_df = _composition_df({"fold_01": 0.05, "fold_02": 0.06, "fold_03": 0.07})
        pairwise_df = _pairwise_df(
            v3_vs_v2={"ci_5th_percentile": 0.02, "ci_5th_percentile_above_zero": True,
                      "significant_at_0.05": True, "holm_adjusted_p_value": 0.01},
            did={"ci_5th_percentile": 0.01, "ci_5th_percentile_above_zero": True,
                 "significant_at_0.05": True, "holm_adjusted_p_value": 0.02},
        )
        conditions, verdict = self._run(pairwise_df, composition_df, delta=0.03)

        assert conditions == {"a_magnitude": True, "b_statistical_reliability": True,
                               "c_cross_fold_consistency": True, "d_attribution": True}
        assert verdict == "H1 SUPPORTED"

    def test_a_and_b_true_d_false_gives_partial_with_exact_sentence(self):
        composition_df = _composition_df({"fold_01": 0.05, "fold_02": 0.06, "fold_03": 0.07})
        pairwise_df = _pairwise_df(
            v3_vs_v2={"ci_5th_percentile": 0.02, "ci_5th_percentile_above_zero": True,
                      "significant_at_0.05": True, "holm_adjusted_p_value": 0.01},
            did={"ci_5th_percentile": -0.01, "ci_5th_percentile_above_zero": False,
                 "significant_at_0.05": False, "holm_adjusted_p_value": 0.30},
        )
        conditions, verdict = self._run(pairwise_df, composition_df, delta=0.03)

        assert conditions["a_magnitude"] is True
        assert conditions["b_statistical_reliability"] is True
        assert conditions["d_attribution"] is False
        assert verdict == (
            "H1 PARTIAL: v3 outperforms v2, but the improvement cannot be confidently "
            "attributed to temporal information specifically rather than search-space size."
        )

    def test_magnitude_below_delta_gives_not_supported_naming_a(self):
        composition_df = _composition_df({"fold_01": 0.05, "fold_02": 0.06, "fold_03": 0.07})
        pairwise_df = _pairwise_df(
            v3_vs_v2={"ci_5th_percentile": 0.02, "ci_5th_percentile_above_zero": True,
                      "significant_at_0.05": True, "holm_adjusted_p_value": 0.01},
            did={"ci_5th_percentile": 0.01, "ci_5th_percentile_above_zero": True,
                 "significant_at_0.05": True, "holm_adjusted_p_value": 0.02},
        )
        # Delta set far above the observed ~0.06 mean IC delta - magnitude condition must fail.
        conditions, verdict = self._run(pairwise_df, composition_df, delta=0.5)

        assert conditions["a_magnitude"] is False
        assert verdict.startswith("H1 NOT SUPPORTED")
        assert "a_magnitude" in verdict

    def test_reliability_false_gives_not_supported_naming_b(self):
        composition_df = _composition_df({"fold_01": 0.05, "fold_02": 0.06, "fold_03": 0.07})
        pairwise_df = _pairwise_df(
            v3_vs_v2={"ci_5th_percentile": -0.01, "ci_5th_percentile_above_zero": False,
                      "significant_at_0.05": False, "holm_adjusted_p_value": 0.40},
            did={"ci_5th_percentile": 0.01, "ci_5th_percentile_above_zero": True,
                 "significant_at_0.05": True, "holm_adjusted_p_value": 0.02},
        )
        conditions, verdict = self._run(pairwise_df, composition_df, delta=0.03)

        assert conditions["a_magnitude"] is True
        assert conditions["b_statistical_reliability"] is False
        assert conditions["d_attribution"] is True
        assert verdict.startswith("H1 NOT SUPPORTED")
        assert "b_statistical_reliability" in verdict
        assert "a_magnitude" not in verdict.split("failed condition(s): ")[1]

    def test_missing_delta_gives_pending_verdict(self):
        composition_df = _composition_df({"fold_01": 0.05, "fold_02": 0.06})
        pairwise_df = _pairwise_df(
            v3_vs_v2={"ci_5th_percentile": 0.02, "ci_5th_percentile_above_zero": True,
                      "significant_at_0.05": True, "holm_adjusted_p_value": 0.01},
            did={"ci_5th_percentile": 0.01, "ci_5th_percentile_above_zero": True,
                 "significant_at_0.05": True, "holm_adjusted_p_value": 0.02},
        )
        conditions, verdict = self._run(pairwise_df, composition_df, delta=None)

        assert conditions["a_magnitude"] is None
        assert verdict == "H1 VERDICT PENDING - one or more conditions could not be evaluated yet"

    def test_missing_pairwise_comparisons_gives_pending_verdict(self):
        composition_df = _composition_df({"fold_01": 0.05, "fold_02": 0.06})
        empty_pairwise_df = pd.DataFrame()  # no comparisons computed yet

        conditions, verdict = self._run(empty_pairwise_df, composition_df, delta=0.01)

        assert conditions["b_statistical_reliability"] is None
        assert conditions["d_attribution"] is None
        assert verdict == "H1 VERDICT PENDING - one or more conditions could not be evaluated yet"
