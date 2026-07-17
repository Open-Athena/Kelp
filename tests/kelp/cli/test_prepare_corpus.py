# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for corpus preparation helpers."""

from kelp.cli.prepare_corpus import extract_functions_from_file

SOURCE = '''
def documented(x):
    """Double x."""
    return x + x


def undocumented(y):
    return y * y
'''


def test_require_docstring_keeps_only_documented_functions():
    """require_docstring filters to functions carrying a docstring (the prompt
    signal); without it, both are kept."""
    all_funcs = extract_functions_from_file(SOURCE, max_length=512)
    documented = extract_functions_from_file(SOURCE, max_length=512, require_docstring=True)

    assert len(all_funcs) == 2
    assert len(documented) == 1
    assert '"""Double x."""' in documented[0]
