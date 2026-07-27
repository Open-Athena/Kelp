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

"""Shared program corpus for Kelp tree diffusion training and evaluation."""

import ast

from etils import epath

CORPUS_SEPARATOR = "# ---"
"""Sentinel line separating programs in corpus files.

Programs may contain internal blank lines, so we use this sentinel
instead of blank-line separation. See prepare_corpus.py for the writer
and load_corpus() below for the reader.
"""


def _finalize_program(lines: list[str]) -> str | None:
    """Strip leading/trailing blank lines and join into a program.

    Returns the program text (newline-terminated) or None if the block is
    empty once blank lines are removed. Applied uniformly to every program so
    whitespace is symmetric across the corpus, not just for the final block.
    """
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    if start == end:
        return None
    return "\n".join(lines[start:end]) + "\n"


def load_corpus(path: str) -> list[str]:
    """Load a corpus from a local or ``gs://`` file.

    Programs are separated by lines containing only '# ---'.
    This allows programs to contain internal blank lines. Leading/trailing
    blank lines are stripped from every program identically. The path is opened
    via ``etils.epath`` so ``gs://`` corpora work without a local copy, and
    lines are streamed (not slurped) so large corpora don't sit in memory. We
    iterate the file object rather than ``str.splitlines()`` because the latter
    also splits on form-feed / NEL / Unicode line separators that are valid
    bytes inside Python source and would corrupt those programs.
    """
    programs: list[str] = []
    current_lines: list[str] = []

    with epath.Path(path).open("r") as f:
        for line in f:
            if line.rstrip() == CORPUS_SEPARATOR:
                program = _finalize_program(current_lines)
                if program is not None:
                    programs.append(program)
                current_lines = []
            else:
                current_lines.append(line.rstrip())

    program = _finalize_program(current_lines)
    if program is not None:
        programs.append(program)

    return programs


def extract_docstring(source: str) -> str | None:
    """Extract the docstring from a Python function source.

    Parses the source and returns the docstring of the first function
    definition found, or None if there is no docstring or parsing fails.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node)
            return docstring
    return None


def is_valid_python(source: str) -> bool:
    """Check whether a string is syntactically valid Python."""
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False


def extract_spec_asserts(source: str, max_asserts: int = 4) -> str | None:
    """Derive a specification block (assert lines) from a function's doctests.

    The training-side analog of the MBPP assert lists used at eval: each doctest
    example whose expression is a single call and whose expected output is a
    literal becomes ``assert <call> == <literal>``, normalizing both spec
    sources to the same format (kelp_v2.md M2, issue #147). Examples with
    non-literal output (reprs of objects, multi-line wants, statements) are
    skipped rather than guessed at. Returns None when nothing qualifies.
    """
    import doctest

    docstring = extract_docstring(source)
    if not docstring:
        return None

    try:
        examples = doctest.DocTestParser().get_examples(docstring)
    except ValueError:
        return None

    asserts: list[str] = []
    for ex in examples:
        expr = ex.source.strip()
        want = ex.want.strip()
        if not want or "\n" in want or "\n" in expr:
            continue
        try:
            mod = ast.parse(expr)
            if len(mod.body) != 1 or not isinstance(mod.body[0], ast.Expr):
                continue
            ast.literal_eval(want)
        except (SyntaxError, ValueError):
            continue
        asserts.append(f"assert {expr} == {want}")
        if len(asserts) >= max_asserts:
            break

    return "\n".join(asserts) if asserts else None


# Toy corpus of 15 small Python functions used for training and evaluation.
# Each program is a standalone function covering basic arithmetic, comparisons,
# and control flow patterns.
TOY_CORPUS = [
    "def add(a, b):\n    return a + b\n",
    "def sub(a, b):\n    return a - b\n",
    "def mul(a, b):\n    return a * b\n",
    "def div(a, b):\n    return a / b\n",
    "def neg(x):\n    return -x\n",
    "def square(x):\n    return x * x\n",
    "def double(x):\n    return x + x\n",
    "def is_positive(x):\n    return x > 0\n",
    "def is_zero(x):\n    return x == 0\n",
    "def identity(x):\n    return x\n",
    "def abs_val(x):\n    if x < 0:\n        return -x\n    return x\n",
    "def max_val(a, b):\n    if a > b:\n        return a\n    return b\n",
    "def min_val(a, b):\n    if a < b:\n        return a\n    return b\n",
    "def clamp(x, lo, hi):\n    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n    return x\n",
    "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n - 1) + fib(n - 2)\n",
]
