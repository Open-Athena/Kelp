# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared AST source-position helpers."""

import ast

from kelp.tree.ast_positions import find_span_end, linecol_to_offset, node_source_span


def test_linecol_to_offset_first_line():
    source = "hello world\nsecond line\n"
    assert linecol_to_offset(source, 1, 0) == 0
    assert linecol_to_offset(source, 1, 6) == 6


def test_linecol_to_offset_second_line():
    source = "hello world\nsecond line\n"
    assert linecol_to_offset(source, 2, 0) == 12
    assert linecol_to_offset(source, 2, 7) == 19


def test_node_source_span():
    source = "x = 1 + 2\n"
    tree = ast.parse(source)
    # The BinOp node should span "1 + 2".
    assign = tree.body[0]
    binop = assign.value
    span = node_source_span(source, binop)
    assert span is not None
    start, end = span
    assert source[start:end] == "1 + 2"


def test_find_span_end_expression():
    source = "x = 1 + 2\n"
    # The BinOp "1 + 2" starts at offset 4.
    end = find_span_end(source, 4)
    assert end is not None
    assert source[4:end] == "1 + 2"


def test_find_span_end_call():
    source = "x = foo(1)\n"
    # The Call "foo(1)" starts at offset 4.
    end = find_span_end(source, 4)
    assert end is not None
    assert source[4:end] == "foo(1)"


def test_find_span_end_no_match():
    source = "x = 1\n"
    # Offset 99 doesn't correspond to any node.
    end = find_span_end(source, 99)
    assert end is None


def test_find_span_end_invalid_python():
    end = find_span_end("def (broken", 0)
    assert end is None


def test_valid_edit_start_offsets_matches_find_span_end():
    """valid_edit_start_offsets returns exactly the offsets where find_span_end
    succeeds -- the set used to build the decode-time position mask."""
    from kelp.tree.ast_positions import valid_edit_start_offsets

    src = "def f(x):\n    return x + 1\n"
    starts = valid_edit_start_offsets(src)
    assert starts  # non-empty for real code
    for s in starts:
        assert find_span_end(src, s) is not None
    # an offset mid-token is not a node start
    assert 3 not in starts
    assert find_span_end(src, 3) is None
