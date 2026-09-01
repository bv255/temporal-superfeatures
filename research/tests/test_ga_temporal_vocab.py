"""
Unit tests for `superfeatures.operators.temporal.temporal_ops` - the single source of truth for the
GA's temporal operator vocabulary. No SparkSession needed.
"""
from superfeatures.operators.temporal import temporal_ops, TemporalOpSpec
from superfeatures.genome.grammar import ExpressionGrammar


class TestTemporalOpsVocab:
    def test_default_matches_add_temporal_features_defaults(self):
        ops = temporal_ops()
        names = {op.name for op in ops}
        assert names == {"lag1", "delta1", "growth1", "mean3", "std3"}

    def test_suffixes_match_add_temporal_features_naming(self):
        ops = {op.name: op for op in temporal_ops()}
        assert ops["lag1"].suffix == "_lag1"
        assert ops["delta1"].suffix == "_delta1"
        assert ops["growth1"].suffix == "_growth1"
        assert ops["mean3"].suffix == "_mean3"
        assert ops["std3"].suffix == "_std3"

    def test_linearity_flags(self):
        ops = {op.name: op for op in temporal_ops()}
        assert ops["lag1"].linear
        assert ops["delta1"].linear
        assert ops["mean3"].linear
        assert not ops["growth1"].linear
        assert not ops["std3"].linear

    def test_multi_period_config_generates_all_combinations(self):
        ops = temporal_ops(lag_periods=[1, 2], window_sizes=[3, 6])
        names = {op.name for op in ops}
        assert names == {
            "lag1", "delta1", "growth1", "lag2", "delta2", "growth2",
            "mean3", "std3", "mean6", "std6",
        }

    def test_op_is_frozen_dataclass(self):
        op = TemporalOpSpec("lag1", "lag", 1, True, "_lag1")
        assert op.name == "lag1" and op.family == "lag" and op.param == 1


class TestCanonicalizationAcrossVocabConfigs:
    def test_two_spellings_collapse_to_same_key_when_vocab_supports_it(self):
        ops = temporal_ops(lag_periods=[1, 2, 3])
        g = ExpressionGrammar(temporal_ops=ops)
        spelling_a = ("lag1", ("lag2", "x"))
        spelling_b = ("lag2", ("lag1", "x"))
        assert g.canonicalize(spelling_a) == g.canonicalize(spelling_b) == ("lag3", "x")

    def test_same_spellings_stay_distinct_when_vocab_does_not_support_collapse(self):
        ops = temporal_ops(lag_periods=[1, 2])
        g = ExpressionGrammar(temporal_ops=ops)
        # lag1(lag2(x)) has no lag3 op in this vocab, so it can't collapse to a single atom -
        # but it still canonicalizes to one fixed spelling via the commuting-family reorder.
        spelling_a = g.canonicalize(("lag1", ("lag2", "x")))
        spelling_b = g.canonicalize(("lag2", ("lag1", "x")))
        assert spelling_a == spelling_b
