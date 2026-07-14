# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hand-written evaluation tasks and the decontamination signatures derived
from them.

This is the single source of truth for the small hand-checked repair tasks.
The evaluation CLI consumes ``EVAL_TASKS``; corpus preparation consumes
``EVAL_SIGNATURES`` to exclude any training program that embeds an eval task
(train/test decontamination). Deriving the signatures from the tasks keeps the
two in sync automatically — editing ``EVAL_TASKS`` can no longer silently stale
the decontamination filter.
"""

# Each task is {name, clean program, list of (call_expression, expected_output)}.
EVAL_TASKS: list[dict] = [
    {
        "name": "add",
        "clean": "def add(a, b):\n    return a + b\n",
        "tests": [
            ("add(1, 2)", "3"),
            ("add(0, 0)", "0"),
            ("add(-1, 1)", "0"),
            ("add(10, 20)", "30"),
        ],
    },
    {
        "name": "sub",
        "clean": "def sub(a, b):\n    return a - b\n",
        "tests": [
            ("sub(5, 3)", "2"),
            ("sub(0, 0)", "0"),
            ("sub(1, 5)", "-4"),
        ],
    },
    {
        "name": "mul",
        "clean": "def mul(a, b):\n    return a * b\n",
        "tests": [
            ("mul(3, 4)", "12"),
            ("mul(0, 5)", "0"),
            ("mul(-2, 3)", "-6"),
        ],
    },
    {
        "name": "neg",
        "clean": "def neg(x):\n    return -x\n",
        "tests": [
            ("neg(5)", "-5"),
            ("neg(-3)", "3"),
            ("neg(0)", "0"),
        ],
    },
    {
        "name": "abs_val",
        "clean": "def abs_val(x):\n    if x < 0:\n        return -x\n    return x\n",
        "tests": [
            ("abs_val(5)", "5"),
            ("abs_val(-3)", "3"),
            ("abs_val(0)", "0"),
        ],
    },
    {
        "name": "max_val",
        "clean": "def max_val(a, b):\n    if a > b:\n        return a\n    return b\n",
        "tests": [
            ("max_val(3, 5)", "5"),
            ("max_val(5, 3)", "5"),
            ("max_val(4, 4)", "4"),
        ],
    },
    {
        "name": "min_val",
        "clean": "def min_val(a, b):\n    if a < b:\n        return a\n    return b\n",
        "tests": [
            ("min_val(3, 5)", "3"),
            ("min_val(5, 3)", "3"),
            ("min_val(4, 4)", "4"),
        ],
    },
    {
        "name": "clamp",
        "clean": (
            "def clamp(x, lo, hi):\n    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n    return x\n"  # noqa: E501 -- literal program sample
        ),
        "tests": [
            ("clamp(5, 1, 10)", "5"),
            ("clamp(-1, 0, 10)", "0"),
            ("clamp(15, 0, 10)", "10"),
        ],
    },
    {
        "name": "double",
        "clean": "def double(x):\n    return x + x\n",
        "tests": [
            ("double(3)", "6"),
            ("double(0)", "0"),
            ("double(-2)", "-4"),
        ],
    },
    {
        "name": "square",
        "clean": "def square(x):\n    return x * x\n",
        "tests": [
            ("square(3)", "9"),
            ("square(0)", "0"),
            ("square(-2)", "4"),
        ],
    },
]

# Signature (first line) of each eval task's clean program. Any training program
# that contains one of these as a substring is excluded to prevent train/test
# leakage — this catches both the function itself and indirect leakage (test
# fixtures/helpers that embed eval task code as string literals). Derived from
# EVAL_TASKS so it can never drift out of sync.
EVAL_SIGNATURES: list[str] = [task["clean"].splitlines()[0] for task in EVAL_TASKS]
