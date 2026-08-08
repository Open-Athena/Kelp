# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for executable spec synthesis (M2 / exp12)."""

from kelp.cli._test_runner import default_runner
from kelp.spec_synthesis import synthesize_spec


def test_eval_expr_returns_repr_and_rejects_nondeterminism():
    """The worker returns the repr of a pure call, and refuses a value that
    changes between its two evaluations -- the contract that keeps flaky
    asserts out of training data."""
    runner = default_runner()
    assert runner.eval_expr("def f(x):\n    return x * 2", "f(3)") == "6"
    impure = "state = []\ndef g():\n    state.append(1)\n    return len(state)"
    assert runner.eval_expr(impure, "g()") is None
    assert runner.eval_expr("def h():\n    raise ValueError", "h()") is None


def test_synthesize_layers_validate_and_are_deterministic():
    """Green doctests are kept, stale doctests are dropped, fuzz asserts are
    true-by-construction, and the whole result is reproducible."""
    source = (
        "def double(x):\n"
        '    """Double x.\n'
        "\n"
        "    >>> double(2)\n"
        "    4\n"
        "    >>> double(3)\n"
        "    7\n"
        '    """\n'
        "    return x * 2\n"
    )
    a = synthesize_spec(source)
    b = synthesize_spec(source)
    assert a.asserts == b.asserts  # deterministic across runs
    assert "assert double(2) == 4" in a.asserts  # green doctest kept
    assert "assert double(3) == 7" not in a.asserts  # stale doctest dropped
    assert a.n_fuzz > 0  # fuzzing filled remaining slots
    runner = default_runner()
    for line in a.asserts:
        assert runner.run(source, line) is True  # every emitted assert is executable-true


def test_synthesize_skips_unfuzzable_and_impure():
    """No function, wide/vararg signatures, and impure functions yield no fuzz
    asserts instead of flaky ones."""
    assert synthesize_spec("x = 1\n").spec is None
    assert synthesize_spec("def f(*args):\n    return args\n").n_fuzz == 0
    impure = "import random\ndef r(a):\n    return random.random() + a\n"
    assert synthesize_spec(impure).spec is None


def test_impure_function_doctest_is_rejected():
    """A nondeterministic function's doctest must not become a training spec,
    even if a lucky run would pass it -- impurity gates the doctest layer too."""
    src = (
        'import random\ndef roll():\n    """Roll.\n\n    >>> roll()\n    3\n    """\n    return random.randint(1, 6)\n'
    )
    result = synthesize_spec(src)
    assert result.spec is None and result.skip_reason == "impure"


def test_sidecar_key_survives_corpus_round_trip(tmp_path):
    """Sidecar keys are built on normalized text; a program with trailing
    whitespace on an internal line must produce the SAME key after being
    written and reloaded through load_corpus (which rstrips every line) --
    the mismatch here silently orphaned specs in exp12 training."""
    from kelp.cli.prepare_corpus import write_corpus
    from kelp.corpus import load_corpus, normalize_program
    from kelp.spec_synthesis import corpus_spec_key

    raw = "def f(x):  \n    y = 1  \n    return x + y\n"
    normalized = normalize_program(raw)
    assert normalized != raw  # the trailing spaces are real
    out = tmp_path / "c.txt"
    write_corpus([normalized], out)
    (reloaded,) = load_corpus(str(out))
    assert corpus_spec_key(reloaded) == corpus_spec_key(normalized)
