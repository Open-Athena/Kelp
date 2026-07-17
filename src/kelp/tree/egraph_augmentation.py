# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""E-graph-based augmentation for the subtree bank.

Uses egglog (equality saturation) to generate semantically related variants
of Python expressions. Given a BinOp like `a + b`, rewrite rules produce
equivalents like `b + a` and near-misses like `a - b`, dramatically expanding
the subtree bank with principled, composable transformations.

This replaces the ad-hoc operator perturbation and synthetic template
strategies in augmentation.py with a single, extensible rule system.

Requires: pip install egglog
"""

import ast
import logging
import random

from egglog import EGraph, Expr, StringLike, birewrite, rewrite, vars_

from kelp.tree.ast_positions import node_source_span
from kelp.tree.mutation import Mutation
from kelp.tree.subtree_bank import (
    EXPRESSION_TYPES,
    SubtreeBank,
    SubtreeEntry,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# E-graph schema: a term language for Python expressions
# ---------------------------------------------------------------------------


class PyExpr(Expr):
    """E-graph sort representing a Python expression.

    Constructors mirror Python's AST expression nodes. Each leaf is a Var
    (variable name or literal) and each internal node is an operator.
    """

    @classmethod
    def var(cls, name: StringLike) -> "PyExpr":
        """A leaf: variable reference or literal."""
        ...

    def __add__(self, other: "PyExpr") -> "PyExpr": ...
    def __sub__(self, other: "PyExpr") -> "PyExpr": ...
    def __mul__(self, other: "PyExpr") -> "PyExpr": ...

    @classmethod
    def floordiv(cls, left: "PyExpr", right: "PyExpr") -> "PyExpr": ...

    @classmethod
    def mod(cls, left: "PyExpr", right: "PyExpr") -> "PyExpr": ...

    @classmethod
    def pow_(cls, base: "PyExpr", exp: "PyExpr") -> "PyExpr": ...

    def __neg__(self) -> "PyExpr": ...

    @classmethod
    def uadd(cls, operand: "PyExpr") -> "PyExpr": ...

    # Comparisons (return PyExpr, not bool, to stay within the sort)
    @classmethod
    def lt(cls, left: "PyExpr", right: "PyExpr") -> "PyExpr": ...

    @classmethod
    def lte(cls, left: "PyExpr", right: "PyExpr") -> "PyExpr": ...

    @classmethod
    def gt(cls, left: "PyExpr", right: "PyExpr") -> "PyExpr": ...

    @classmethod
    def gte(cls, left: "PyExpr", right: "PyExpr") -> "PyExpr": ...

    @classmethod
    def eq_(cls, left: "PyExpr", right: "PyExpr") -> "PyExpr": ...

    @classmethod
    def neq(cls, left: "PyExpr", right: "PyExpr") -> "PyExpr": ...

    # Boolean ops
    @classmethod
    def and_(cls, left: "PyExpr", right: "PyExpr") -> "PyExpr": ...

    @classmethod
    def or_(cls, left: "PyExpr", right: "PyExpr") -> "PyExpr": ...

    @classmethod
    def not_(cls, operand: "PyExpr") -> "PyExpr": ...


# Convenience: the zero and one constants used in identity rules.
ZERO = PyExpr.var("__ZERO__")
ONE = PyExpr.var("__ONE__")


# ---------------------------------------------------------------------------
# Rewrite rules
# ---------------------------------------------------------------------------


def _register_rules(egraph: EGraph) -> None:
    """Register all rewrite rules on the given e-graph.

    Rules are grouped into:
    1. Semantic equivalences (bidirectional) — produce equivalent expressions
    2. Near-miss noise rules (unidirectional) — produce similar-but-wrong variants
       useful for training the model to distinguish subtle differences
    """
    a, b, c = vars_("a b c", PyExpr)

    # --- Semantic equivalences (bidirectional) ---
    egraph.register(
        # Commutativity of addition and multiplication
        birewrite(a + b).to(b + a),
        birewrite(a * b).to(b * a),
        # Associativity
        birewrite((a + b) + c).to(a + (b + c)),
        birewrite((a * b) * c).to(a * (b * c)),
        # Commutativity of boolean ops
        birewrite(PyExpr.and_(a, b)).to(PyExpr.and_(b, a)),
        birewrite(PyExpr.or_(a, b)).to(PyExpr.or_(b, a)),
        # Comparison flip
        birewrite(PyExpr.lt(a, b)).to(PyExpr.gt(b, a)),
        birewrite(PyExpr.lte(a, b)).to(PyExpr.gte(b, a)),
        # Double negation
        rewrite(-(-a)).to(a),  # noqa: B002 -- egglog DSL: builds a nested unary-negation PyExpr, not Python decrement
        rewrite(PyExpr.not_(PyExpr.not_(a))).to(a),
        # Unary add is identity
        rewrite(PyExpr.uadd(a)).to(a),
    )

    # --- Near-miss noise rules (unidirectional) ---
    # These generate "wrong" variants that are structurally similar to the
    # correct expression. The model must learn to distinguish them.
    egraph.register(
        # Operator swaps within arithmetic
        rewrite(a + b).to(a - b),
        rewrite(a - b).to(a + b),
        rewrite(a * b).to(PyExpr.floordiv(a, b)),
        # Comparison boundary shifts
        rewrite(PyExpr.lt(a, b)).to(PyExpr.lte(a, b)),
        rewrite(PyExpr.gt(a, b)).to(PyExpr.gte(a, b)),
        rewrite(PyExpr.lte(a, b)).to(PyExpr.lt(a, b)),
        rewrite(PyExpr.gte(a, b)).to(PyExpr.gt(a, b)),
        # Equality/inequality flip
        rewrite(PyExpr.eq_(a, b)).to(PyExpr.neq(a, b)),
        rewrite(PyExpr.neq(a, b)).to(PyExpr.eq_(a, b)),
        # Boolean op swap
        rewrite(PyExpr.and_(a, b)).to(PyExpr.or_(a, b)),
        rewrite(PyExpr.or_(a, b)).to(PyExpr.and_(a, b)),
        # Sign flips on structured expressions (not bare variables, which
        # would cause an "ungrounded variable" error in egglog).
        rewrite(a + b).to(-(a + b)),
        rewrite(a - b).to(-(a - b)),
        rewrite(a * b).to(-(a * b)),
    )


# ---------------------------------------------------------------------------
# AST ↔ e-graph conversion
# ---------------------------------------------------------------------------

# Maps Python AST BinOp types to PyExpr constructor functions.
_BINOP_TO_EGRAPH: dict[type, str] = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.FloorDiv: "floordiv",
    ast.Mod: "mod",
    ast.Pow: "pow_",
}

_UNARYOP_TO_EGRAPH: dict[type, str] = {
    ast.USub: "neg",
    ast.UAdd: "uadd",
    ast.Not: "not_",
}

_CMPOP_TO_EGRAPH: dict[type, str] = {
    ast.Lt: "lt",
    ast.LtE: "lte",
    ast.Gt: "gt",
    ast.GtE: "gte",
    ast.Eq: "eq_",
    ast.NotEq: "neq",
}

_BOOLOP_TO_EGRAPH: dict[type, str] = {
    ast.And: "and_",
    ast.Or: "or_",
}


def _ast_to_pyexpr(node: ast.expr) -> PyExpr | None:
    """Convert a Python AST expression node to a PyExpr e-graph term.

    Returns None if the node type is not supported.
    """
    if isinstance(node, ast.BinOp):
        op_name = _BINOP_TO_EGRAPH.get(type(node.op))
        if op_name is None:
            return None
        left = _ast_to_pyexpr(node.left)
        right = _ast_to_pyexpr(node.right)
        if left is None or right is None:
            return None
        if op_name == "add":
            return left + right
        elif op_name == "sub":
            return left - right
        elif op_name == "mul":
            return left * right
        elif op_name == "floordiv":
            return PyExpr.floordiv(left, right)
        elif op_name == "mod":
            return PyExpr.mod(left, right)
        elif op_name == "pow_":
            return PyExpr.pow_(left, right)

    elif isinstance(node, ast.UnaryOp):
        op_name = _UNARYOP_TO_EGRAPH.get(type(node.op))
        if op_name is None:
            return None
        operand = _ast_to_pyexpr(node.operand)
        if operand is None:
            return None
        if op_name == "neg":
            return -operand
        elif op_name == "uadd":
            return PyExpr.uadd(operand)
        elif op_name == "not_":
            return PyExpr.not_(operand)

    elif isinstance(node, ast.BoolOp):
        op_name = _BOOLOP_TO_EGRAPH.get(type(node.op))
        if op_name is None:
            return None
        # BoolOp has a list of values; fold pairwise left-to-right.
        terms = [_ast_to_pyexpr(v) for v in node.values]
        if any(t is None for t in terms):
            return None
        result = terms[0]
        method = getattr(PyExpr, op_name)
        for t in terms[1:]:
            result = method(result, t)
        return result

    elif isinstance(node, ast.Compare) and len(node.ops) == 1:
        # Only handle simple comparisons (one operator).
        op_name = _CMPOP_TO_EGRAPH.get(type(node.ops[0]))
        if op_name is None:
            return None
        left = _ast_to_pyexpr(node.left)
        right = _ast_to_pyexpr(node.comparators[0])
        if left is None or right is None:
            return None
        method = getattr(PyExpr, op_name)
        return method(left, right)

    elif isinstance(node, ast.Name):
        return PyExpr.var(node.id)

    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return PyExpr.var(repr(node.value))
        elif isinstance(node.value, bool):
            return PyExpr.var(str(node.value))
        elif isinstance(node.value, str):
            return PyExpr.var(repr(node.value))

    return None


# Reverse mapping: PyExpr string representation → Python AST source.
_EGRAPH_OP_TO_PYTHON: dict[str, str] = {
    "__add__": "+",
    "__sub__": "-",
    "__mul__": "*",
    "floordiv": "//",
    "mod": "%",
    "pow_": "**",
    "lt": "<",
    "lte": "<=",
    "gt": ">",
    "gte": ">=",
    "eq_": "==",
    "neq": "!=",
    "and_": "and",
    "or_": "or",
}


def _pyexpr_to_source(expr: PyExpr) -> str | None:
    """Convert a PyExpr e-graph term back to Python source code.

    Uses string representation parsing since egglog's Python API returns
    structured Expr objects that we can inspect via repr().
    """
    s = repr(expr)
    return _parse_pyexpr_repr(s)


def _parse_pyexpr_repr(s: str) -> str | None:
    """Parse the repr of a PyExpr back into Python source.

    egglog repr looks like:
      PyExpr.var("x")
      PyExpr.var("x") + PyExpr.var("y")
      PyExpr.lt(PyExpr.var("a"), PyExpr.var("b"))
      -(PyExpr.var("x"))
      PyExpr.not_(PyExpr.var("x"))
    """
    s = s.strip()

    # Leaf: PyExpr.var("...") — must match exactly one var call, not a
    # longer expression that happens to start/end with var quotes.
    _VAR_PREFIX = 'PyExpr.var("'
    if s.startswith(_VAR_PREFIX):
        close = s.find('")', len(_VAR_PREFIX))
        if close != -1 and close + 2 == len(s):
            name = s[len(_VAR_PREFIX) : close]
            if name == "__ZERO__":
                return "0"
            elif name == "__ONE__":
                return "1"
            return name

    # Negation: -(expr)
    if s.startswith("-(") and s.endswith(")"):
        inner = _parse_pyexpr_repr(s[2:-1])
        if inner is None:
            return None
        return f"-({inner})" if " " in inner else f"-{inner}"

    # Unary: PyExpr.uadd(expr) or PyExpr.not_(expr)
    if s.startswith("PyExpr.uadd(") and s.endswith(")"):
        inner = _parse_pyexpr_repr(s[len("PyExpr.uadd(") : -1])
        if inner is None:
            return None
        return f"+({inner})" if " " in inner else f"+{inner}"

    if s.startswith("PyExpr.not_(") and s.endswith(")"):
        inner = _parse_pyexpr_repr(s[len("PyExpr.not_(") : -1])
        if inner is None:
            return None
        return f"not ({inner})" if " " in inner else f"not {inner}"

    # Binary class methods: PyExpr.op(left, right)
    for op_name, py_op in _EGRAPH_OP_TO_PYTHON.items():
        prefix = f"PyExpr.{op_name}("
        if s.startswith(prefix) and s.endswith(")"):
            inner = s[len(prefix) : -1]
            left_str, right_str = _split_args(inner)
            if left_str is None or right_str is None:
                return None
            left = _parse_pyexpr_repr(left_str)
            right = _parse_pyexpr_repr(right_str)
            if left is None or right is None:
                return None
            return f"({left}) {py_op} ({right})" if py_op in ("and", "or") else f"{left} {py_op} {right}"

    # Infix operators: (expr) op (expr) — e.g., PyExpr.var("a") + PyExpr.var("b")
    for _op_name, _py_op in [("__add__", "+"), ("__sub__", "-"), ("__mul__", "*")]:
        # egglog uses Python operators, so repr shows: expr + expr, expr - expr, etc.
        # Find the operator between balanced parenthesized expressions.
        for delim in [" + ", " - ", " * "]:
            idx = _find_toplevel_operator(s, delim)
            if idx is not None:
                left = _parse_pyexpr_repr(s[:idx])
                right = _parse_pyexpr_repr(s[idx + len(delim) :])
                if left is not None and right is not None:
                    return f"{left} {delim.strip()} {right}"

    # Parenthesized: (expr)
    if s.startswith("(") and s.endswith(")"):
        return _parse_pyexpr_repr(s[1:-1])

    return None


def _find_toplevel_operator(s: str, op: str) -> int | None:
    """Find the position of a top-level infix operator in a repr string.

    Skips operators inside parentheses or quotes.
    """
    depth = 0
    in_string = False
    string_char = None
    i = 0
    while i < len(s):
        ch = s[i]
        if in_string:
            if ch == string_char and (i == 0 or s[i - 1] != "\\"):
                in_string = False
        elif ch in ('"', "'"):
            in_string = True
            string_char = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and s[i : i + len(op)] == op:
            return i
        i += 1
    return None


def _split_args(s: str) -> tuple[str | None, str | None]:
    """Split a string like 'expr1, expr2' at the top-level comma."""
    depth = 0
    in_string = False
    string_char = None
    for i, ch in enumerate(s):
        if in_string:
            if ch == string_char and (i == 0 or s[i - 1] != "\\"):
                in_string = False
        elif ch in ('"', "'"):
            in_string = True
            string_char = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            return s[:i].strip(), s[i + 1 :].strip()
    return None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _operator_multiset(expr_source: str) -> tuple[str, ...] | None:
    """Sorted multiset of operator node types in an expression, or None if it
    doesn't parse. Used to tell a semantic near-miss (operator changed) from a
    pure reordering (commutativity), which leaves the operator multiset intact.
    """
    try:
        tree = ast.parse(expr_source, mode="eval")
    except SyntaxError:
        return None
    ops: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.BinOp, ast.UnaryOp)):
            ops.append(type(node.op).__name__)
        elif isinstance(node, ast.BoolOp):
            ops.append(type(node.op).__name__)
        elif isinstance(node, ast.Compare):
            ops.extend(type(o).__name__ for o in node.ops)
    return tuple(sorted(ops))


def near_miss_variants(expr_source: str, max_variants: int = 8) -> list[str]:
    """E-graph variants of an expression that CHANGE an operator (semantic bugs).

    Filters ``generate_expression_variants`` down to the near-misses -- variants
    whose operator multiset differs from the original (e.g. ``==`` -> ``!=``,
    ``+`` -> ``-``, ``and`` -> ``or``) -- dropping pure reorderings that stay
    semantically equivalent. These are realistic, in-context single-operator
    bugs (same variables, same structure), unlike swapping in an unrelated
    subtree from the bank.
    """
    base_ops = _operator_multiset(expr_source)
    if base_ops is None:
        return []
    out = []
    for v in generate_expression_variants(expr_source, max_variants=max_variants):
        v_ops = _operator_multiset(v)
        if v_ops is not None and v_ops != base_ops:
            out.append(v)
    return out


def near_miss_corrupt_program(
    source: str,
    num_steps: int,
    rng: random.Random,
    max_variants: int = 8,
) -> tuple[str, list[Mutation]]:
    """Corrupt a program with realistic in-context near-miss bugs.

    Drop-in alternative to ``mutation.corrupt_program``: instead of replacing a
    subtree with an unrelated one from the bank, it applies an e-graph near-miss
    rewrite (operator flip) to one of the program's own arithmetic / comparison /
    boolean expressions. The result is a plausible single-operator bug over the
    program's own variables. Re-parses after each step so offsets stay valid;
    applies fewer than ``num_steps`` if it runs out of eligible expressions.
    """
    current = source
    mutations: list[Mutation] = []
    _EXPR = (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare)
    for _ in range(num_steps):
        try:
            tree = ast.parse(current)
        except SyntaxError:
            break
        # Candidate expression nodes with a resolvable source span.
        cands = []
        for node in ast.walk(tree):
            if isinstance(node, _EXPR):
                span = node_source_span(current, node)
                if span is not None:
                    cands.append((node, span))
        rng.shuffle(cands)
        applied = False
        for _node, (start, end) in cands:
            original = current[start:end]
            variants = near_miss_variants(original, max_variants=max_variants)
            if not variants:
                continue
            replacement = rng.choice(variants)
            candidate = current[:start] + replacement + current[end:]
            try:
                ast.parse(candidate)
            except SyntaxError:
                continue
            mutations.append(
                Mutation(
                    start=start,
                    end=end,
                    replacement=replacement,
                    node_type=type(_node).__name__,
                    original=original,
                )
            )
            current = candidate
            applied = True
            break
        if not applied:
            break
    return current, mutations


def generate_expression_variants(
    source: str,
    max_variants: int = 10,
    max_iterations: int = 5,
) -> list[str]:
    """Generate expression variants of a Python expression using e-graph rewriting.

    Args:
        source: A Python expression string (e.g., "a + b", "x > 0 and y < 10").
        max_variants: Maximum number of variants to extract.
        max_iterations: Bound on equality saturation iterations.

    Returns:
        List of Python source strings that are variants of the input.
        Includes both semantic equivalents and near-miss corruptions.
        Does not include the original expression.
    """
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError:
        return []

    pyexpr = _ast_to_pyexpr(tree.body)
    if pyexpr is None:
        return []

    egraph = EGraph()
    _register_rules(egraph)
    expr_handle = egraph.let("expr", pyexpr)
    egraph.run(max_iterations)

    raw_variants = egraph.extract_multiple(expr_handle, max_variants + 1)
    results = []
    for variant in raw_variants:
        py_source = _pyexpr_to_source(variant)
        if py_source is None:
            continue
        # Normalize through ast to get canonical formatting.
        try:
            normalized = ast.unparse(ast.parse(py_source, mode="eval").body)
        except SyntaxError:
            continue
        if normalized != source and normalized not in results:
            results.append(normalized)
        if len(results) >= max_variants:
            break

    return results


def augment_bank_with_egraph(
    bank: SubtreeBank,
    max_variants_per_entry: int = 5,
    max_iterations: int = 5,
) -> tuple[SubtreeBank, int]:
    """Augment a subtree bank by generating e-graph expression variants.

    For each expression-type entry in the bank, runs equality saturation
    with rewrite rules and adds the resulting variants as new entries.

    Args:
        bank: The subtree bank to augment.
        max_variants_per_entry: Max variants to generate per expression entry.
        max_iterations: Equality saturation iteration bound.

    Returns:
        Tuple of (augmented bank, number of new entries added).
    """
    augmented = bank.copy()
    seen: set[tuple[str, str]] = augmented.source_keys()

    added = 0
    expression_entries = 0
    skipped_conversion = 0

    for node_type, entries in bank.entries.items():
        if node_type not in EXPRESSION_TYPES:
            continue

        for entry in entries:
            expression_entries += 1
            variants = generate_expression_variants(
                entry.source,
                max_variants=max_variants_per_entry,
                max_iterations=max_iterations,
            )

            if not variants:
                skipped_conversion += 1
                continue

            for variant_source in variants:
                # Determine the actual AST node type of the variant.
                try:
                    variant_tree = ast.parse(variant_source, mode="eval")
                    variant_type = type(variant_tree.body).__name__
                except SyntaxError:
                    continue

                if variant_type not in EXPRESSION_TYPES:
                    continue

                if (variant_type, variant_source) in seen:
                    continue

                seen.add((variant_type, variant_source))
                augmented.add(
                    SubtreeEntry(
                        source=variant_source,
                        node_type=variant_type,
                        stmt_count=0,
                    )
                )
                added += 1

    logger.info(
        f"E-graph augmentation: {expression_entries} expression entries processed, "
        f"{skipped_conversion} skipped (unsupported), {added} new variants added"
    )

    return augmented, added
