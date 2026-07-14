# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Kelp inference: iterative program repair.

The reverse (denoising) process — best-of-N and beam search over
model-predicted edits, bracket-constrained decoding to keep partial edits
valid, and execution-guided reranking of candidate repairs against tests.
"""

from kelp.inference.beam_search import BeamCandidate, beam_search, best_of_n, generate_edit
from kelp.inference.constrained_decoding import apply_bracket_constraints, validate_edit
from kelp.inference.reranking import (
    RankedCandidate,
    execute_python_with_tests,
    rerank_candidates,
    score_candidate,
)

__all__ = [
    "BeamCandidate",
    "generate_edit",
    "beam_search",
    "best_of_n",
    "apply_bracket_constraints",
    "validate_edit",
    "RankedCandidate",
    "score_candidate",
    "rerank_candidates",
    "execute_python_with_tests",
]
