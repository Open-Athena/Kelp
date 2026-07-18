# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for corpus preparation helpers."""

from kelp.cli.prepare_corpus import extract_functions_from_file, extract_local_functions

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


def test_extract_local_functions_scans_dir_under_excluded_ancestor(tmp_path):
    """EXCLUDE_DIRS is matched relative to source_dir, so an explicitly-requested
    library living under a '.venv' ancestor (site-packages) is still scanned --
    while a nested 'tests' dir within the tree is skipped."""
    pkg = tmp_path / ".venv" / "site-packages" / "mylib"
    pkg.mkdir(parents=True)
    (pkg / "mod.py").write_text(SOURCE)
    (pkg / "tests").mkdir()
    (pkg / "tests" / "test_mod.py").write_text('def helper(z):\n    """Doc."""\n    return z\n')

    funcs = extract_local_functions(pkg, max_length=512, require_docstring=True)

    # The library's documented function is found despite the '.venv' ancestor;
    # the nested tests/ dir is excluded.
    assert any('"""Double x."""' in f for f in funcs)
    assert not any('"""Doc."""' in f for f in funcs)


def test_extract_local_functions_skips_nested_site_packages(tmp_path):
    """A nested site-packages / _vendor tree (e.g. a stdlib install bundling
    pip's vendored libs) is excluded, so scanning one library never pulls in
    third-party code of unknown provenance."""
    root = tmp_path / "python3.12"
    (root).mkdir(parents=True)
    (root / "statistics.py").write_text(SOURCE)  # "real" stdlib module
    vendored = root / "site-packages" / "pip" / "_vendor"
    vendored.mkdir(parents=True)
    (vendored / "rich.py").write_text('def rich_fn(a):\n    """Vendored."""\n    return a + 1\n')

    funcs = extract_local_functions(root, max_length=512, require_docstring=True)

    assert any('"""Double x."""' in f for f in funcs)  # real module kept
    assert not any('"""Vendored."""' in f for f in funcs)  # vendored code excluded
