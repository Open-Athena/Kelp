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

"""Tests for AST-based mutation (tree diffusion forward process)."""

import ast
import random

import pytest

from kelp.tree.mutation import (
    Mutation,
    _find_candidates,
    corrupt_program,
    flip_one_operator,
    operator_flip_corrupt_program,
    random_mutation,
    swap_one_variable,
    variable_swap_corrupt_program,
)
from kelp.tree.subtree_bank import SubtreeBank

CORPUS = [
    """\
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for i in range(2, n + 1):
        a, b = b, a + b
    return b
""",
    """\
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
""",
    """\
def greet(name):
    msg = f"Hello, {name}!"
    print(msg)
    return msg
""",
    """\
def add(a, b):
    return a + b
""",
    """\
def maximum(lst):
    result = lst[0]
    for x in lst[1:]:
        if x > result:
            result = x
    return result
""",
    """\
def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
""",
]


@pytest.fixture
def bank():
    return SubtreeBank.from_corpus(CORPUS)


def test_mutation_apply():
    m = Mutation(start=4, end=9, replacement="world", node_type="Name", original="hello")
    result = m.apply("say hello there")
    assert result == "say world there"


def test_random_mutation_produces_valid_python(bank):
    source = CORPUS[0]  # fibonacci
    rng = random.Random(42)

    mutation = random_mutation(source, bank, rng=rng)
    assert mutation is not None

    mutated = mutation.apply(source)
    # The result must be valid Python.
    try:
        ast.parse(mutated)
    except SyntaxError:
        pytest.fail(f"Mutation produced invalid Python:\n{mutated}")


def test_random_mutation_changes_source(bank):
    source = CORPUS[0]  # fibonacci
    rng = random.Random(42)

    mutation = random_mutation(source, bank, rng=rng)
    assert mutation is not None
    assert mutation.replacement != mutation.original


def test_random_mutation_returns_none_for_invalid_source(bank):
    mutation = random_mutation("this is not python{{{", bank)
    assert mutation is None


def test_random_mutation_returns_none_for_empty_bank():
    bank = SubtreeBank()
    mutation = random_mutation("x = 1\n", bank)
    assert mutation is None


def test_random_mutation_multiple_seeds_produce_different_results(bank):
    source = CORPUS[4]  # maximum -- has multiple candidate nodes
    results = set()
    for seed in range(20):
        mutation = random_mutation(source, bank, rng=random.Random(seed))
        if mutation is not None:
            results.add((mutation.start, mutation.end, mutation.replacement))

    # With 20 different seeds, we should get at least 2 distinct mutations.
    assert len(results) >= 2, "Expected diverse mutations across seeds"


def test_corrupt_program_produces_valid_python(bank):
    source = CORPUS[0]  # fibonacci
    rng = random.Random(42)

    corrupted, mutations = corrupt_program(source, num_steps=3, bank=bank, rng=rng)

    assert len(mutations) > 0
    assert corrupted != source

    try:
        ast.parse(corrupted)
    except SyntaxError:
        pytest.fail(f"corrupt_program produced invalid Python:\n{corrupted}")


def test_corrupt_program_returns_mutations_in_order(bank):
    source = CORPUS[0]
    rng = random.Random(42)

    corrupted, mutations = corrupt_program(source, num_steps=3, bank=bank, rng=rng)

    # Replay the mutations to verify they produce the same result.
    current = source
    for m in mutations:
        current = m.apply(current)
    assert current == corrupted


def test_corrupt_program_single_step(bank):
    source = CORPUS[3]  # add -- very short
    rng = random.Random(42)

    corrupted, mutations = corrupt_program(source, num_steps=1, bank=bank, rng=rng)

    assert len(mutations) <= 1
    if mutations:
        assert corrupted != source


def test_corrupt_program_zero_steps(bank):
    source = CORPUS[0]
    corrupted, mutations = corrupt_program(source, num_steps=0, bank=bank)
    assert corrupted == source
    assert mutations == []


def test_corrupt_program_graceful_when_no_mutations_possible():
    """If the bank has no matching types, corruption returns the original."""
    bank = SubtreeBank()  # empty bank
    source = "x = 1\n"
    corrupted, mutations = corrupt_program(source, num_steps=5, bank=bank)
    assert corrupted == source
    assert mutations == []


def test_corrupt_many_programs_all_valid(bank):
    """Smoke test: corrupt every corpus program and check validity."""
    rng = random.Random(123)
    for source in CORPUS:
        corrupted, _mutations = corrupt_program(source, num_steps=2, bank=bank, rng=rng)
        try:
            ast.parse(corrupted)
        except SyntaxError:
            pytest.fail(
                f"corrupt_program produced invalid Python for:\n"
                f"--- original ---\n{source}\n"
                f"--- corrupted ---\n{corrupted}"
            )


def test_find_candidates_skips_root_functiondef(bank):
    """Root-level FunctionDef nodes should never be mutation candidates.

    Replacing the root FunctionDef swaps the entire program for another,
    turning repair into synthesis. See DIAGNOSTIC_REPORT.md Finding 1.
    """
    source = "def add(a, b):\n    return a + b\n"
    tree = ast.parse(source)

    candidates = _find_candidates(source, tree, max_edit_stmts=3, bank=bank)
    candidate_types = [c.node_type for c in candidates]

    # The root FunctionDef must not appear, but inner nodes should.
    assert "FunctionDef" not in candidate_types
    # Return and BinOp are inside the function body — they should be eligible.
    assert any(t in candidate_types for t in ("Return", "BinOp"))


def test_find_candidates_excludes_docstring(bank):
    """A bare string-literal statement (docstring) must never be a bank-swap
    candidate: swapping it makes the repair target a verbatim string the model
    must reproduce rather than a localized edit.
    """
    source = 'def f(x):\n    """Compute something important about x."""\n    return x + 1\n'
    tree = ast.parse(source)

    candidates = _find_candidates(source, tree, max_edit_stmts=3, bank=bank)
    docstring = '"""Compute something important about x."""'
    for c in candidates:
        assert source[c.start : c.end] != docstring, "docstring statement leaked into candidates"
    # A real, non-docstring node is still eligible (the filter is not over-broad).
    assert any(source[c.start : c.end] == "x + 1" for c in candidates)


def test_corrupt_program_never_targets_docstring(bank):
    """End-to-end: across many seeds, corruption never rewrites the docstring."""
    source = 'def f(x):\n    """Compute something important about x."""\n    return x + 1\n'
    for seed in range(30):
        _corrupted, mutations = corrupt_program(source, num_steps=1, bank=bank, rng=random.Random(seed))
        for m in mutations:
            assert m.original.strip() != '"""Compute something important about x."""'


def test_corruption_preserves_function_signature(bank):
    """After corruption, the top-level function name and args should survive."""
    source = CORPUS[0]  # fibonacci

    for seed in range(20):
        corrupted, mutations = corrupt_program(source, num_steps=3, bank=bank, rng=random.Random(seed))
        if not mutations:
            continue
        # The corrupted program should still define 'fibonacci'.
        tree = ast.parse(corrupted)
        top_funcs = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
        assert "fibonacci" in top_funcs, f"Corruption replaced the root FunctionDef (seed={seed}):\n{corrupted}"


# ---------------------------------------------------------------------------
# Realistic in-context corruption: operator flip
# ---------------------------------------------------------------------------


def test_operator_flip_fires_inside_calls_and_subscripts():
    """The defining contract: op-flip corrupts an operator whose operands are a
    call and a subscript -- exactly the code the e-graph cannot model -- while
    leaving both operands byte-for-byte intact.
    """
    source = "def f(a, b):\n    return self.compute(a) + b[0]\n"
    mutation = flip_one_operator(source, random.Random(0))

    assert mutation is not None
    assert mutation.original == "+" and mutation.replacement != "+"
    corrupted = mutation.apply(source)
    ast.parse(corrupted)  # still valid Python
    # Operands survive verbatim; only the operator token changed.
    assert "self.compute(a)" in corrupted and "b[0]" in corrupted
    assert corrupted != source


def test_operator_flip_returns_none_without_operator():
    """No flippable operator -> None, so the caller can fall through to another
    corruption mechanism instead of silently no-op'ing."""
    source = "def f(x):\n    return g(x)\n"
    assert flip_one_operator(source, random.Random(0)) is None
    corrupted, mutations = operator_flip_corrupt_program(source, num_steps=3, rng=random.Random(0))
    assert corrupted == source and mutations == []


# ---------------------------------------------------------------------------
# Realistic in-context corruption: variable swap
# ---------------------------------------------------------------------------


def test_variable_swap_uses_only_in_scope_names():
    """A swap replaces one name read with a *different* name already present in
    the program (in-context), never an alien token."""
    # Names that reach the swap pool are ast.Name reads (width, height, max) --
    # parameters are ast.arg and don't count, so the body must use them.
    source = "def f(width, height):\n    return max(width, height)\n"
    mutation = swap_one_variable(source, random.Random(0))

    assert mutation is not None
    assert mutation.node_type == "Name"
    assert mutation.replacement in {"max", "width", "height"}
    assert mutation.replacement != mutation.original
    corrupted = mutation.apply(source)
    ast.parse(corrupted)
    assert corrupted != source


def test_variable_swap_returns_none_with_one_name():
    """Fewer than two distinct names -> nothing plausible to swap in -> None."""
    source = "def f():\n    return x\n"  # only one name: x
    assert swap_one_variable(source, random.Random(0)) is None
    corrupted, mutations = variable_swap_corrupt_program(source, num_steps=2, rng=random.Random(0))
    assert corrupted == source and mutations == []
