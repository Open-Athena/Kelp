# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Kelp tree layer: AST representation and corruption.

Holds the forward (corruption) process over Python ASTs — the SubtreeBank of
replacement fragments, the mutation operators that swap type-compatible
subtrees, the minimal edit-path (TreeDiff) back to a clean program, source
tokenization, and bank augmentation. Every operation preserves syntactic
validity: each intermediate program is valid Python.
"""

from kelp.tree.augmentation import augment_bank
from kelp.tree.mutation import Mutation, corrupt_program, random_mutation
from kelp.tree.subtree_bank import EXTRACTABLE_TYPES, SubtreeBank, SubtreeEntry
from kelp.tree.tokenizer import EditTokenizer
from kelp.tree.tree_diff import Edit, find_path, one_step_edit, tree_diff

__all__ = [
    "SubtreeBank",
    "SubtreeEntry",
    "EXTRACTABLE_TYPES",
    "Mutation",
    "corrupt_program",
    "random_mutation",
    "Edit",
    "tree_diff",
    "find_path",
    "one_step_edit",
    "EditTokenizer",
    "augment_bank",
]
