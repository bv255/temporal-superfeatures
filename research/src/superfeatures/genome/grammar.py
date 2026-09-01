"""
Atom-aware representation/mutation/crossover logic for GA individuals, extracted out of
`GeneticAlgorithm1` (`engine.py`) per TEMPORAL_SUBTREE_OPERATORS_PROMPT.md section 7.

An individual is a bare leaf string, a flat odd-length tuple alternating
leaf/operator/leaf/... (unchanged from the original grammar), or - new - a temporal atom:
a 2-tuple `(op_name, inner)` where `inner` is itself a leaf string or a flat tuple, e.g.
`('lag1', 'ff_roic')` or `('lag1', ('ff_mkt_val', '/', 'ff_op_margin'))`. Every
tuple-walking method here checks `is_atom` before the generic tuple case, since an atom is
itself a 2-tuple.

With `temporal_ops=[]` (the default `GeneticAlgorithm1` still constructs when no grammar is
passed), `is_atom` is always False and every method below reproduces the original
`engine.py` methods' behavior and random-draw sequence exactly - this is what keeps a
plain arithmetic-only individual evaluating/mutating/crossing-over identically before and
after this refactor.

`OPERATORS` is an ordered tuple, not a set, so `rng.choice(OPERATORS)` reproduces the exact
same sequence `random.choice(['+', '-', '*', '/'])` did (a set's iteration order is not
guaranteed to match the original literal's order across runs).
"""
import random
from typing import List, Optional

from ..operators.temporal import TemporalOpSpec
from ..evaluation.metrics import classify_leaf

OPERATORS = ('+', '-', '*', '/')
FAMILY_RANK = {'lag': 0, 'delta': 1, 'mean': 2}


class ExpressionGrammar:
    def __init__(
        self,
        temporal_ops: Optional[List[TemporalOpSpec]] = None,
        max_features: int = 5,
        max_nesting: int = 5,
        max_temporal_depth: int = 2,
        temporal_wrap_rate: float = 0.15,
        temporal_unwrap_rate: float = 0.15,
        report_date_col: str = "report_date",
        entity_col: str = "fsym_id",
    ):
        self.temporal_ops = list(temporal_ops) if temporal_ops else []
        self.ops_by_name = {op.name: op for op in self.temporal_ops}
        self.max_features = max_features
        self.max_nesting = max_nesting
        self.max_temporal_depth = max_temporal_depth
        self.temporal_wrap_rate = temporal_wrap_rate
        self.temporal_unwrap_rate = temporal_unwrap_rate
        self.report_date_col = report_date_col
        self.entity_col = entity_col

    # ---- representation ----

    def is_atom(self, token) -> bool:
        return (
            isinstance(token, tuple) and len(token) == 2
            and isinstance(token[0], str) and token[0] in self.ops_by_name
        )

    def is_leaf_token(self, token) -> bool:
        return (isinstance(token, str) and token not in OPERATORS) or self.is_atom(token)

    def flatten_expression(self, expr):
        if isinstance(expr, tuple) and not self.is_atom(expr):
            result = []
            for item in expr:
                result.extend(self.flatten_expression(item))
            return result
        return [expr]

    def flatten_to_leaves(self, expr):
        if self.is_atom(expr):
            return self.flatten_to_leaves(expr[1])
        if isinstance(expr, str):
            return [] if expr in OPERATORS else [expr]
        if isinstance(expr, tuple):
            result = []
            for item in expr:
                result.extend(self.flatten_to_leaves(item))
            return result
        return []

    def count_features(self, expr) -> int:
        if self.is_atom(expr):
            return 1
        if isinstance(expr, str):
            return 0 if expr in OPERATORS else 1
        if isinstance(expr, tuple):
            return sum(self.count_features(e) for e in expr)
        return 0

    def count_features_through_atoms(self, expr) -> int:
        if self.is_atom(expr):
            return self.count_features_through_atoms(expr[1])
        if isinstance(expr, str):
            return 0 if expr in OPERATORS else 1
        if isinstance(expr, tuple):
            return sum(self.count_features_through_atoms(e) for e in expr)
        return 0

    def count_nesting(self, expr, level: int = 0) -> int:
        if self.is_atom(expr):
            return self.count_nesting(expr[1], level + 1)
        if isinstance(expr, tuple):
            return max((self.count_nesting(e, level + 1) for e in expr), default=level)
        return level

    def count_temporal_depth(self, expr) -> int:
        if self.is_atom(expr):
            return 1 + self.count_temporal_depth(expr[1])
        if isinstance(expr, tuple):
            return max((self.count_temporal_depth(e) for e in expr), default=0)
        return 0

    def _check_adjacency(self, expr) -> bool:
        if self.is_atom(expr):
            return self._check_adjacency(expr[1])
        if isinstance(expr, tuple):
            last_was_operator = False
            for item in expr:
                if isinstance(item, str) and item in OPERATORS:
                    if last_was_operator:
                        return False
                    last_was_operator = True
                else:
                    last_was_operator = False
                    if not self._check_adjacency(item):
                        return False
            return True
        return True

    def is_valid_expression(self, expr) -> bool:
        # count_features_through_atoms (not count_features, which treats a whole temporal atom
        # as a single token) - otherwise repeated wrapping/crossover can grow an individual's
        # real base-feature count past max_features while this check keeps passing.
        # count_features itself must stay used elsewhere (crossover_deep and its helpers rely on
        # its token-position arithmetic over the flat tuple representation, a different concept).
        return (
            self._check_adjacency(expr)
            and self.count_features_through_atoms(expr) <= self.max_features
            and self.count_nesting(expr) <= self.max_nesting
            and self.count_temporal_depth(expr) <= self.max_temporal_depth
        )

    # ---- crossover ----

    def crossover_deep(self, expr1, expr2, rng=random):
        features1 = self.count_features(expr1)
        features2 = self.count_features(expr2)

        if features1 + features2 == 2:
            operator = rng.choice(OPERATORS)
            new_expr1 = (expr1, operator, expr2)
            new_expr2 = (expr2, operator, expr1)
        elif features1 + features2 <= self.max_features:
            e1 = (expr1,) if self.is_leaf_token(expr1) else expr1
            e2 = (expr2,) if self.is_leaf_token(expr2) else expr2
            operator = rng.choice(OPERATORS)
            new_expr1 = e1 + (operator,) + e2
            new_expr2 = e2 + (operator,) + e1
        elif features1 + features2 > self.max_features and features1 < self.max_features and features2 < self.max_features:
            features_allowed1 = self.max_features - features1
            chunk1 = self._extract_linear_feature_chunk(expr2, features_allowed1, rng)
            if chunk1:
                trial_expr1 = self._insert_chunk(expr1, chunk1, rng)
                new_expr1 = trial_expr1 if self.is_valid_expression(trial_expr1) else expr1
            else:
                new_expr1 = expr1

            features_allowed2 = self.max_features - features2
            chunk2 = self._extract_linear_feature_chunk(expr1, features_allowed2, rng)
            if chunk2:
                trial_expr2 = self._insert_chunk(expr2, chunk2, rng)
                new_expr2 = trial_expr2 if self.is_valid_expression(trial_expr2) else expr2
            else:
                new_expr2 = expr2
        elif (features1 == self.max_features and features2 < self.max_features) or (
            features2 == self.max_features and features1 < self.max_features
        ):
            if features1 < self.max_features:
                new_expr1 = self._replace_chunk(expr2, expr1, rng)
                features_allowed = self.max_features - features1
                chunk = self._extract_linear_feature_chunk(expr2, features_allowed, rng)
                if chunk:
                    trial_expr2 = self._insert_chunk(expr1, chunk, rng)
                    new_expr2 = trial_expr2 if self.is_valid_expression(trial_expr2) else expr2
                else:
                    new_expr2 = expr2
            else:
                new_expr2 = self._replace_chunk(expr1, expr2, rng)
                features_allowed = self.max_features - features2
                chunk = self._extract_linear_feature_chunk(expr1, features_allowed, rng)
                if chunk:
                    trial_expr1 = self._insert_chunk(expr2, chunk, rng)
                    new_expr1 = trial_expr1 if self.is_valid_expression(trial_expr1) else expr1
                else:
                    new_expr1 = expr1
        elif features1 == self.max_features and features2 == self.max_features:
            tokens1 = self.flatten_expression(expr1)
            tokens2 = self.flatten_expression(expr2)
            assert len(tokens1) == len(tokens2) == 2 * self.max_features - 1
            valid_cut_points = list(range(1, len(tokens1) - 1, 2))
            cut = rng.choice(valid_cut_points)
            new_expr1 = tuple(tokens1[:cut] + tokens2[cut:])
            new_expr2 = tuple(tokens2[:cut] + tokens1[cut:])

        if not self.is_valid_expression(new_expr1):
            new_expr1 = expr1
        if not self.is_valid_expression(new_expr2):
            new_expr2 = expr2

        return new_expr1 if rng.random() > 0.5 else new_expr2

    def _extract_linear_feature_chunk(self, expr, required_features, rng=random):
        tokens = self.flatten_expression(expr)
        chunk_length = 2 * required_features - 1
        valid_chunks = []
        for i in range(len(tokens) - chunk_length + 1):
            chunk = tokens[i:i + chunk_length]
            if (
                all(self.is_leaf_token(chunk[j]) for j in range(0, len(chunk), 2))
                and all(chunk[j] in OPERATORS for j in range(1, len(chunk), 2))
            ):
                valid_chunks.append(chunk)
        return tuple(rng.choice(valid_chunks)) if valid_chunks else None

    def _insert_chunk(self, base_expr, chunk, rng=random):
        if self.is_leaf_token(base_expr):
            if rng.random() < 0.5:
                return tuple(chunk) + (rng.choice(OPERATORS), base_expr)
            else:
                return (base_expr,) + (rng.choice(OPERATORS),) + tuple(chunk)
        elif isinstance(base_expr, tuple):
            elements = list(base_expr)
            insert_pos = rng.randrange(0, len(elements) + 1)
            operator = rng.choice(OPERATORS)
            if insert_pos % 2 == 0:
                elements[insert_pos:insert_pos] = list(chunk) + [operator]
            else:
                elements[insert_pos:insert_pos] = [operator] + list(chunk)
            return tuple(elements)
        else:
            return base_expr

    def _replace_chunk(self, base_expr, chunk_to_insert, rng=random):
        base_tokens = self.flatten_expression(base_expr)
        insert_features = self.count_features(chunk_to_insert)
        chunk_len = 2 * insert_features - 1
        valid_ranges = []
        for i in range(len(base_tokens) - chunk_len + 1):
            window = base_tokens[i:i + chunk_len]
            if (
                all(self.is_leaf_token(window[j]) for j in range(0, len(window), 2))
                and all(window[j] in OPERATORS for j in range(1, len(window), 2))
            ):
                valid_ranges.append(i)
        if not valid_ranges:
            return base_expr
        start = rng.choice(valid_ranges)
        new_tokens = base_tokens[:start] + list(self.flatten_expression(chunk_to_insert)) + base_tokens[start + chunk_len:]
        return tuple(new_tokens)

    # ---- mutation ----

    def mutate_feature(self, individual, feature_pool, rng=random):
        if self.is_atom(individual):
            op_name, inner = individual
            return (op_name, self.mutate_feature(inner, feature_pool, rng))
        if isinstance(individual, str):
            return rng.choice(feature_pool)
        if isinstance(individual, tuple):
            part_to_mutate = rng.choice(range(0, len(individual), 2))
            new_parts = list(individual)
            new_parts[part_to_mutate] = self.mutate_feature(individual[part_to_mutate], feature_pool, rng)
            return tuple(new_parts)
        raise ValueError(f"Invalid individual format for mutation: {individual}")

    def mutate_operator(self, individual, rng=random):
        if self.is_atom(individual):
            op_name, inner = individual
            choices = [name for name in self.ops_by_name if name != op_name] or [op_name]
            return (rng.choice(choices), inner)
        if isinstance(individual, tuple):
            operator_positions = range(1, len(individual), 2)
            operator_to_mutate = rng.choice(operator_positions)
            new_operator = rng.choice(OPERATORS)
            new_parts = list(individual)
            new_parts[operator_to_mutate] = new_operator
            return tuple(new_parts)
        raise ValueError(f"Invalid individual format for mutation: {individual}")

    def mutate(self, individual, feature_pool, rng=random):
        if self.temporal_ops:
            r = rng.random()
            if r < self.temporal_wrap_rate:
                wrapped = self.wrap(individual, rng)
                if wrapped is not None:
                    return wrapped
            elif r < self.temporal_wrap_rate + self.temporal_unwrap_rate:
                unwrapped = self.unwrap(individual, rng)
                if unwrapped is not None:
                    return unwrapped

        if self.is_atom(individual):
            if rng.random() < 0.5:
                return (individual[0], self.mutate_feature(individual[1], feature_pool, rng))
            return self.mutate_operator(individual, rng)
        if isinstance(individual, str):
            return self.mutate_feature(individual, feature_pool, rng)
        if isinstance(individual, tuple) and len(individual) % 2 == 1:
            if rng.random() < 0.5:
                return self.mutate_operator(individual, rng)
            return self.mutate_feature(individual, feature_pool, rng)
        raise ValueError(f"Unsupported individual format for mutation: {individual}")

    # ---- structural temporal wrap/unwrap ----

    def _flatten_one_level(self, expr):
        if isinstance(expr, tuple) and not self.is_atom(expr):
            return list(expr)
        return [expr]

    def is_redundant_wrap(self, op: TemporalOpSpec, run_tokens: list) -> bool:
        if len(run_tokens) < 3:
            return False
        if not op.linear:
            return False
        operator_tokens = run_tokens[1::2]
        return all(tok in ('+', '-') for tok in operator_tokens)

    def wrap(self, expression, rng=random, max_attempts: int = 20):
        if not self.temporal_ops:
            return None
        tokens = self._flatten_one_level(expression)
        feature_positions = [i for i in range(0, len(tokens), 2) if self.is_leaf_token(tokens[i])]
        if not feature_positions:
            return None

        for _ in range(max_attempts):
            op = rng.choice(self.temporal_ops)
            run_len = rng.randint(1, len(feature_positions))
            start_i = rng.randrange(0, len(feature_positions) - run_len + 1)
            start = feature_positions[start_i]
            end = feature_positions[start_i + run_len - 1]
            run_tokens = tokens[start:end + 1]

            if self.is_redundant_wrap(op, run_tokens):
                continue

            inner = run_tokens[0] if run_len == 1 else tuple(run_tokens)
            new_tokens = tokens[:start] + [(op.name, inner)] + tokens[end + 1:]
            new_expr = new_tokens[0] if len(new_tokens) == 1 else tuple(new_tokens)
            new_expr = self.canonicalize(new_expr)
            if self.is_valid_expression(new_expr):
                return new_expr
        return None

    def _unwrap_candidates(self, expr):
        if self.is_atom(expr):
            op_name, inner = expr
            yield inner
            for cand in self._unwrap_candidates(inner):
                yield (op_name, cand)
        elif isinstance(expr, tuple):
            tokens = list(expr)
            for i in range(0, len(tokens), 2):
                slot = tokens[i]
                if isinstance(slot, str):
                    continue
                for cand in self._unwrap_candidates(slot):
                    if isinstance(cand, tuple) and not self.is_atom(cand):
                        spliced = tokens[:i] + list(cand) + tokens[i + 1:]
                    else:
                        spliced = tokens[:i] + [cand] + tokens[i + 1:]
                    yield tuple(spliced)

    def unwrap(self, expression, rng=random):
        if not self.temporal_ops:
            return None
        candidates = []
        seen = set()
        for cand in self._unwrap_candidates(expression):
            cand = self.canonicalize(cand)
            if self.is_valid_expression(cand):
                key = repr(cand)
                if key not in seen:
                    seen.add(key)
                    candidates.append(cand)
        return rng.choice(candidates) if candidates else None

    # ---- canonical form ----

    def canonicalize(self, expr):
        if self.is_atom(expr):
            return self._canonicalize_atom(expr)
        if isinstance(expr, tuple):
            return tuple(self.canonicalize(e) for e in expr)
        return expr

    def _canonicalize_atom(self, expr):
        op_name, inner = expr
        inner = self.canonicalize(inner)
        op = self.ops_by_name[op_name]

        if self.is_atom(inner):
            inner_op = self.ops_by_name.get(inner[0])
            if inner_op is not None and op.family == inner_op.family and op.linear and inner_op.linear:
                combined_name = f"{op.family}{op.param + inner_op.param}"
                if combined_name in self.ops_by_name:
                    return self.canonicalize((combined_name, inner[1]))

            if (
                inner_op is not None
                and op.family in FAMILY_RANK
                and inner_op.family in FAMILY_RANK
            ):
                op_key = (FAMILY_RANK[op.family], op.param)
                inner_key = (FAMILY_RANK[inner_op.family], inner_op.param)
                if op_key > inner_key:
                    return self.canonicalize((inner_op.name, (op.name, inner[1])))

        return (op_name, inner)

    # ---- evaluation (spark-agnostic recursive walker) ----

    def evaluate(self, expression, leaf_fn, combine_fn, temporal_fn=None):
        if self.is_atom(expression):
            op_name, inner = expression
            train, test = self.evaluate(inner, leaf_fn, combine_fn, temporal_fn)
            if temporal_fn is None:
                raise ValueError(f"temporal_fn required to evaluate atom {expression}")
            op = self.ops_by_name[op_name]
            return temporal_fn(train, op), temporal_fn(test, op)
        if isinstance(expression, str):
            return leaf_fn(expression)
        if isinstance(expression, tuple) and len(expression) % 2 == 1:
            data, test = self.evaluate(expression[0], leaf_fn, combine_fn, temporal_fn)
            for i in range(1, len(expression), 2):
                operator = expression[i]
                next_data, next_test = self.evaluate(expression[i + 1], leaf_fn, combine_fn, temporal_fn)
                data = combine_fn(data, next_data, operator)
                test = combine_fn(test, next_test, operator)
            return data, test
        raise ValueError("Invalid expression format")

    # ---- leaf classification ----

    def classify_leaf_or_atom(self, token) -> str:
        if self.is_atom(token):
            return self.ops_by_name[token[0]].family
        return classify_leaf(token)
