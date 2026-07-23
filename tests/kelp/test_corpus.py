# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for corpus file loading."""

from kelp.corpus import extract_spec_asserts, load_corpus


def test_load_corpus_strips_whitespace_symmetrically(tmp_path):
    """Every program gets identical leading/trailing blank-line stripping --
    not just the final block (regression for the last-program-only asymmetry).
    """
    # First program has surrounding blank lines; the separated blocks must be
    # stripped the same way the final block is.
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text(
        "\n\ndef a():\n    return 1\n\n\n# ---\ndef b():\n    return 2\n",
    )

    programs = load_corpus(str(corpus_file))

    assert programs == ["def a():\n    return 1\n", "def b():\n    return 2\n"]


def test_load_corpus_preserves_internal_blank_lines_and_drops_empty_blocks(tmp_path):
    """Blank lines *inside* a program survive; an all-blank block is not a
    program and is skipped."""
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text(
        "def a():\n    x = 1\n\n    return x\n# ---\n\n\n# ---\ndef b():\n    return 2\n",
    )

    programs = load_corpus(str(corpus_file))

    assert programs == ["def a():\n    x = 1\n\n    return x\n", "def b():\n    return 2\n"]


def test_write_then_load_round_trips_via_epath(tmp_path):
    """write_corpus -> load_corpus preserves programs; both go through epath so
    the same code path serves local and gs:// paths."""
    from kelp.cli.prepare_corpus import write_corpus

    progs = ["def a():\n    return 1\n", "def b():\n    '''doc'''\n    return 2\n"]
    out = tmp_path / "corpus.txt"
    write_corpus(progs, out)
    assert load_corpus(str(out)) == progs


# --- extract_spec_asserts tests (issue #147) ---


def test_extract_spec_asserts_normalizes_doctests_to_mbpp_style():
    """Doctest examples with literal outputs become assert lines (the same
    format as MBPP eval specs); non-literal and multi-line wants are skipped
    rather than guessed at."""
    source = (
        "def add(a, b):\n"
        '    """Add two numbers.\n'
        "\n"
        "    >>> add(1, 2)\n"
        "    3\n"
        "    >>> add('a', 'b')\n"
        "    'ab'\n"
        "    >>> add(object(), object())\n"
        "    <unrepresentable>\n"
        '    """\n'
        "    return a + b\n"
    )
    spec = extract_spec_asserts(source)
    assert spec == "assert add(1, 2) == 3\nassert add('a', 'b') == 'ab'"


def test_extract_spec_asserts_none_without_usable_examples():
    """No docstring, no doctests, or statement-only examples yield None -- the
    caller then simply trains that example unconditioned (spec dropout path)."""
    assert extract_spec_asserts("def f(x):\n    return x\n") is None
    assert extract_spec_asserts('def f(x):\n    """Docs, no examples."""\n    return x\n') is None
    stmt_only = 'def f(x):\n    """\n    >>> y = f(1)\n    """\n    return x\n'
    assert extract_spec_asserts(stmt_only) is None
