"""
Unit tests for `superfeatures.genome.grammar.ExpressionGrammar` - the atom-aware representation
extracted from `GeneticAlgorithm1` per TEMPORAL_SUBTREE_OPERATORS_PROMPT.md sections 5/7. No
SparkSession needed - pure Python.
"""
import random

import pytest

from superfeatures.genome.grammar import ExpressionGrammar, OPERATORS
from superfeatures.operators.temporal import temporal_ops, TemporalOpSpec


FEATURES = ["a", "b", "c", "d", "e", "f"]


def _grammar(ops=None, **kwargs):
    return ExpressionGrammar(temporal_ops=ops if ops is not None else temporal_ops(), **kwargs)


def _no_temporal_grammar(**kwargs):
    return ExpressionGrammar(temporal_ops=[], **kwargs)


# ---- atom opacity ----

class TestAtomOpacity:
    def test_is_atom_true_for_leaf_atom(self):
        g = _grammar()
        assert g.is_atom(("lag1", "a"))

    def test_is_atom_true_for_subtree_atom(self):
        g = _grammar()
        assert g.is_atom(("lag1", ("a", "/", "b")))

    def test_is_atom_false_for_flat_arithmetic_tuple(self):
        g = _grammar()
        assert not g.is_atom(("a", "+", "b"))

    def test_is_atom_false_when_op_name_not_in_vocabulary(self):
        g = _no_temporal_grammar()
        assert not g.is_atom(("lag1", "a"))

    def test_flatten_expression_treats_atom_as_one_token(self):
        g = _grammar()
        expr = (("lag1", "a"), "+", "b")
        assert g.flatten_expression(expr) == [("lag1", "a"), "+", "b"]

    def test_flatten_expression_matches_original_for_flat_tuple(self):
        g = _grammar()
        assert g.flatten_expression(("a", "+", "b", "-", "c")) == ["a", "+", "b", "-", "c"]

    def test_flatten_to_leaves_unwraps_atoms(self):
        g = _grammar()
        expr = (("lag1", "a"), "+", ("mean3", ("b", "/", "c")))
        assert g.flatten_to_leaves(expr) == ["a", "b", "c"]


# ---- atom-aware counting ----

class TestCounting:
    def test_count_features_atom_counts_as_one(self):
        g = _grammar()
        assert g.count_features(("lag1", ("a", "+", "b", "+", "c"))) == 1

    def test_count_features_through_atoms_recurses(self):
        g = _grammar()
        assert g.count_features_through_atoms(("lag1", ("a", "+", "b", "+", "c"))) == 3

    def test_count_features_unaffected_when_no_atoms(self):
        g = _grammar()
        assert g.count_features(("a", "+", "b")) == 2

    def test_count_temporal_depth_zero_without_atoms(self):
        g = _grammar()
        assert g.count_temporal_depth(("a", "+", "b")) == 0

    def test_count_temporal_depth_counts_stacking(self):
        g = _grammar()
        assert g.count_temporal_depth(("lag1", "a")) == 1
        assert g.count_temporal_depth(("mean3", ("lag1", "a"))) == 2

    def test_count_temporal_depth_ignores_inner_arithmetic_width(self):
        g = _grammar()
        expr = ("lag1", ("a", "+", "b", "+", "c", "+", "d"))
        assert g.count_temporal_depth(expr) == 1

    def test_count_nesting_matches_original_always_one_for_flat_only(self):
        g = _no_temporal_grammar()
        assert g.count_nesting("a") == 0
        assert g.count_nesting(("a", "+", "b")) == 1
        assert g.count_nesting(("a", "+", "b", "-", "c")) == 1


class TestValidExpression:
    def test_valid_flat_expression(self):
        g = _grammar()
        assert g.is_valid_expression(("a", "+", "b"))

    def test_rejects_adjacent_operators(self):
        g = _no_temporal_grammar()
        assert not g._check_adjacency(("a", "+", "-", "b"))

    def test_rejects_over_max_features(self):
        g = _grammar(max_features=2)
        assert not g.is_valid_expression(("a", "+", "b", "+", "c"))

    def test_atom_wrapping_a_valid_subtree_is_valid(self):
        g = _grammar(max_features=5)
        assert g.is_valid_expression(("lag1", ("a", "+", "b")))

    def test_rejects_over_max_temporal_depth(self):
        g = _grammar(max_temporal_depth=1)
        assert not g.is_valid_expression(("mean3", ("lag1", "a")))

    def test_adjacency_checked_inside_atom_inner(self):
        g = _grammar()
        assert not g.is_valid_expression(("lag1", ("a", "+", "-", "b")))


# ---- regression: corruption bugs that would occur without atom-aware fixes ----

class TestCrossoverAtomSafety:
    def test_case2_concatenation_does_not_spread_atom(self):
        # Historically `expr1 + (op,) + expr2` when expr1 is a bare tuple (an atom, e.g.
        # ('lag1', 'a')) would spread the atom's own 2 elements into the result instead of
        # nesting it - producing a corrupt even-length tuple. is_leaf_token must catch atoms
        # so they get wrapped in a singleton tuple first, same as a plain leaf string.
        g = _grammar()
        rng = random.Random(1)
        atom = ("lag1", "a")
        for _ in range(50):
            result = g.crossover_deep(atom, "b", rng)
            assert g.is_valid_expression(result) or result in (atom, "b")
            flat = g.flatten_expression(result) if isinstance(result, tuple) and not g.is_atom(result) else [result]
            assert len(flat) % 2 == 1

    def test_insert_chunk_treats_atom_as_single_token(self):
        g = _grammar()
        rng = random.Random(2)
        atom = ("lag1", "a")
        chunk = ("b", "+", "c")
        result = g._insert_chunk(atom, chunk, rng)
        assert isinstance(result, tuple)
        assert len(result) % 2 == 1
        assert atom in result

    def test_extract_linear_feature_chunk_can_select_atoms(self):
        g = _grammar()
        rng = random.Random(3)
        expr = (("lag1", "a"), "+", "b", "+", "c")
        chunk = g._extract_linear_feature_chunk(expr, 2, rng)
        assert chunk is not None
        assert len(chunk) == 3

    def test_crossover_many_seeds_always_valid(self):
        g = _grammar(max_features=5)
        rng = random.Random(7)
        exprs = ["a", ("lag1", "b"), ("c", "+", "d"), ("mean3", ("e", "-", "f"))]
        for _ in range(200):
            e1 = rng.choice(exprs)
            e2 = rng.choice(exprs)
            result = g.crossover_deep(e1, e2, rng)
            assert g.is_valid_expression(result)


# ---- mutation ----

class TestMutate:
    def test_mutate_feature_leaf(self):
        g = _no_temporal_grammar()
        rng = random.Random(1)
        result = g.mutate_feature("a", FEATURES, rng)
        assert result in FEATURES

    def test_mutate_feature_preserves_atom_wrapper(self):
        g = _grammar()
        rng = random.Random(1)
        result = g.mutate_feature(("lag1", "a"), FEATURES, rng)
        assert result[0] == "lag1"
        assert result[1] in FEATURES

    def test_mutate_operator_on_atom_changes_op_name_only(self):
        g = _grammar()
        rng = random.Random(1)
        result = g.mutate_operator(("lag1", "a"), rng)
        assert result[1] == "a"
        assert result[0] in g.ops_by_name

    def test_mutate_never_raises_over_many_seeds(self):
        g = _grammar(max_features=5)
        exprs = ["a", ("lag1", "b"), ("c", "+", "d"), ("mean3", ("e", "-", "f"))]
        for seed in range(200):
            rng = random.Random(seed)
            expr = rng.choice(exprs)
            result = g.mutate(expr, FEATURES, rng)
            assert g.is_valid_expression(result)

    def test_mutate_with_empty_temporal_ops_never_produces_atom(self):
        g = _no_temporal_grammar()
        exprs = ["a", ("c", "+", "d")]
        for seed in range(200):
            rng = random.Random(seed)
            expr = rng.choice(exprs)
            result = g.mutate(expr, FEATURES, rng)
            assert not g.is_atom(result)


# ---- wrap / unwrap ----

class TestWrapUnwrap:
    def test_wrap_leaf_produces_atom(self):
        g = _grammar()
        rng = random.Random(5)
        found = False
        for seed in range(100):
            result = g.wrap("a", random.Random(seed))
            if result is not None:
                assert g.is_atom(result)
                found = True
        assert found

    def test_wrap_returns_none_without_vocabulary(self):
        g = _no_temporal_grammar()
        assert g.wrap("a", random.Random(1)) is None

    def test_wrap_subtree_run_of_two_or_more(self):
        g = _grammar()
        found_subtree = False
        for seed in range(300):
            result = g.wrap(("a", "+", "b", "*", "c"), random.Random(seed))
            if result is not None and g.is_atom(result) and g.count_features_through_atoms(result) >= 2:
                found_subtree = True
                break
        assert found_subtree

    def test_wrap_rejects_redundant_additive_run(self):
        g = _grammar(ops=[TemporalOpSpec("lag1", "lag", 1, True, "_lag1")])
        for seed in range(200):
            result = g.wrap(("a", "+", "b"), random.Random(seed))
            if result is not None:
                # a linear op over a purely additive run of >=2 features is redundant and
                # must never be produced structurally (leaf-level wrap already reaches it)
                assert not (g.is_atom(result) and g.count_features_through_atoms(result) >= 2)

    def test_unwrap_leaf_atom_recovers_leaf(self):
        g = _grammar()
        result = g.unwrap(("lag1", "a"), random.Random(1))
        assert result == "a"

    def test_unwrap_returns_none_for_plain_expression(self):
        g = _grammar()
        assert g.unwrap(("a", "+", "b"), random.Random(1)) is None

    def test_unwrap_returns_none_without_vocabulary(self):
        g = _no_temporal_grammar()
        assert g.unwrap(("lag1", "a"), random.Random(1)) is None

    def test_wrap_then_unwrap_round_trip_same_leaf_multiset(self):
        g = _grammar()
        base = ("a", "+", "b", "*", "c")
        for seed in range(50):
            wrapped = g.wrap(base, random.Random(seed))
            if wrapped is None or not g.is_atom(wrapped):
                continue
            unwrapped = g.unwrap(wrapped, random.Random(seed))
            if unwrapped is None:
                continue
            assert g.is_valid_expression(unwrapped)
            assert sorted(g.flatten_to_leaves(unwrapped)) == sorted(g.flatten_to_leaves(wrapped))


class TestRedundancyMatrix:
    LAG = TemporalOpSpec("lag1", "lag", 1, True, "_lag1")
    GROWTH = TemporalOpSpec("growth1", "growth", 1, False, "_growth1")

    def test_linear_op_purely_additive_run_is_redundant(self):
        g = _grammar()
        assert g.is_redundant_wrap(self.LAG, ["a", "+", "b"])
        assert g.is_redundant_wrap(self.LAG, ["a", "-", "b", "+", "c"])

    def test_linear_op_with_multiplicative_token_not_redundant(self):
        g = _grammar()
        assert not g.is_redundant_wrap(self.LAG, ["a", "*", "b"])
        assert not g.is_redundant_wrap(self.LAG, ["a", "/", "b"])

    def test_nonlinear_op_never_redundant(self):
        g = _grammar()
        assert not g.is_redundant_wrap(self.GROWTH, ["a", "+", "b"])
        assert not g.is_redundant_wrap(self.GROWTH, ["a", "*", "b"])

    def test_run_length_one_never_redundant(self):
        g = _grammar()
        assert not g.is_redundant_wrap(self.LAG, ["a"])


class TestCanonicalize:
    def test_collapses_stacked_lags_when_combined_exists(self):
        ops = temporal_ops(lag_periods=[1, 2, 3])
        g = _grammar(ops=ops)
        result = g.canonicalize(("lag1", ("lag2", "a")))
        assert result == ("lag3", "a")

    def test_leaves_stacked_lags_nested_when_combined_missing(self):
        ops = temporal_ops(lag_periods=[1])
        g = _grammar(ops=ops)
        result = g.canonicalize(("lag1", ("lag1", "a")))
        assert result == ("lag1", ("lag1", "a"))

    def test_delta_stacking_never_collapses(self):
        ops = temporal_ops(lag_periods=[1, 2])
        g = _grammar(ops=ops)
        result = g.canonicalize(("delta1", ("delta2", "a")))
        assert result[0] == "delta1"
        assert g.is_atom(result[1])

    def test_growth_never_reordered_or_collapsed(self):
        ops = temporal_ops(lag_periods=[1, 2])
        g = _grammar(ops=ops)
        expr = ("growth1", ("lag2", "a"))
        assert g.canonicalize(expr) == expr

    def test_commuting_families_reorder_to_one_spelling(self):
        ops = temporal_ops(lag_periods=[1], window_sizes=[3])
        g = _grammar(ops=ops)
        a = g.canonicalize(("mean3", ("lag1", "x")))
        b = g.canonicalize(("lag1", ("mean3", "x")))
        assert a == b

    def test_std_never_reordered(self):
        ops = temporal_ops(lag_periods=[1], window_sizes=[3])
        g = _grammar(ops=ops)
        expr = ("std3", ("lag1", "x"))
        assert g.canonicalize(expr) == expr


class TestClassifyLeafOrAtom:
    def test_classifies_atom_by_family(self):
        g = _grammar()
        assert g.classify_leaf_or_atom(("lag1", "a")) == "lag"
        assert g.classify_leaf_or_atom(("mean3", ("a", "+", "b"))) == "mean"

    def test_classifies_legacy_suffixed_string_as_before(self):
        g = _grammar()
        assert g.classify_leaf_or_atom("ff_roic_lag1") == "lag1"

    def test_classifies_plain_string_as_raw(self):
        g = _grammar()
        assert g.classify_leaf_or_atom("ff_roic") == "raw"
