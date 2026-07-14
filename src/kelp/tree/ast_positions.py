# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared AST source-position helpers.

Utilities for mapping between Python ``ast`` node positions and character
offsets into the source string, and for enumerating the AST nodes that the
tree-diffusion process may edit. Centralized here so mutation, tokenization,
edit-path computation, and inference all agree on position geometry rather than
each reimplementing it.
"""

import ast
from collections.abc import Iterator
from dataclasses import dataclass

from kelp.tree.subtree_bank import EXTRACTABLE_TYPES, count_statements


def linecol_to_offset(source: str, line: int, col: int) -> int:
    """Convert a 1-based line and 0-based column to a character offset.

    Args:
        source: The source string.
        line: 1-based line number (as returned by ast nodes).
        col: 0-based column offset.

    Returns:
        0-based character offset into source.
    """
    current_line = 1
    for i, ch in enumerate(source):
        if current_line == line:
            return i + col
        if ch == "\n":
            current_line += 1
    return len(source) + col


def node_source_span(source: str, node: ast.AST) -> tuple[int, int] | None:
    """Return the (start, end) character offsets for an AST node.

    Returns None if the node lacks source position info. This is the single
    place that reads the position attributes ``ast.AST`` does not declare on
    its base class, so the ``type: ignore`` lives here alone.
    """
    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        return None
    if node.end_lineno is None or node.end_col_offset is None:  # type: ignore[attr-defined]  # guarded by hasattr above
        return None

    start = linecol_to_offset(source, node.lineno, node.col_offset)  # type: ignore[attr-defined]
    end = linecol_to_offset(source, node.end_lineno, node.end_col_offset)  # type: ignore[attr-defined]
    return (start, end)


@dataclass(frozen=True)
class PositionedNode:
    """An extractable AST node with its resolved source span and statement count."""

    node: ast.AST
    node_type: str
    start: int
    end: int
    stmt_count: int


def iter_editable_nodes(source: str, tree: ast.AST, max_edit_stmts: int) -> Iterator[PositionedNode]:
    """Yield extractable, positioned AST nodes eligible for editing.

    A node is yielded when its type is in ``EXTRACTABLE_TYPES``, it has valid
    source position info, and its statement count is ``<= max_edit_stmts``.
    Callers add their own further filters (e.g. non-root, bank membership,
    minimum span) on top of this common set.
    """
    for node in ast.walk(tree):
        type_name = type(node).__name__
        if type_name not in EXTRACTABLE_TYPES:
            continue
        span = node_source_span(source, node)
        if span is None:
            continue
        stmt_count = count_statements(node)
        if stmt_count > max_edit_stmts:
            continue
        start, end = span
        yield PositionedNode(node=node, node_type=type_name, start=start, end=end, stmt_count=stmt_count)


def find_span_end(source: str, start_offset: int) -> int | None:
    """Find the end offset of the smallest extractable node starting at ``start_offset``.

    Walks the AST for the innermost (smallest) extractable node whose start
    position equals ``start_offset`` and returns its end offset, or None if
    there is no such node (or the source does not parse).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    best_end = None
    best_size = float("inf")
    for node in ast.walk(tree):
        if type(node).__name__ not in EXTRACTABLE_TYPES:
            continue
        span = node_source_span(source, node)
        if span is None:
            continue
        node_start, node_end = span
        if node_start == start_offset:
            size = node_end - node_start
            if size < best_size:
                best_end = node_end
                best_size = size
    return best_end
