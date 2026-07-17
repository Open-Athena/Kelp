# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for corpus file loading."""

from kelp.corpus import load_corpus


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
