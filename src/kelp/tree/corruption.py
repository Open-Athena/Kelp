# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The shared corruption policy — the single definition of a "realistic" bug.

Both the training data pipeline (`kelp.training.generation`) and the MBPP
evaluator (`kelp.cli.evaluate_mbpp`) corrupt programs through this one function,
so a run is always *tested* on the same distribution it was *trained* on. Keeping
the policy in one leaf module (it imports from `mutation` and
`egraph_augmentation` but nothing imports back) is what makes that guarantee
cheap to hold.

See docs/vet-cond-v1-failure-analysis.md for why train/eval corruption match is
load-bearing for interpreting the experiment.
"""

import random

from kelp.tree.egraph_augmentation import near_miss_corrupt_program
from kelp.tree.mutation import (
    corrupt_program,
    operator_flip_corrupt_program,
    variable_swap_corrupt_program,
)
from kelp.tree.subtree_bank import SubtreeBank

# Corruption mode labels, useful for gut-check rendering and per-type eval
# breakdowns. "bank-swap" is the out-of-context fallback; the rest are the
# realistic in-context modes.
REALISTIC_MODES = ("op-flip", "var-swap", "near-miss")
BANK_SWAP = "bank-swap"


def corrupt_realistic(
    source: str,
    *,
    num_steps: int,
    bank: SubtreeBank,
    rng: random.Random,
    p_near_miss: float,
    max_edit_stmts: int = 3,
) -> tuple[str, str]:
    """Corrupt ``source``, preferring realistic in-context bugs.

    With probability ``p_near_miss`` the realistic cascade is tried, most- to
    least-applicable, using the first mode that actually changes the program:

    1. **operator-flip** — a plausible single-operator bug (``+``→``*``,
       ``==``→``!=``), fires even inside calls/subscripts the e-graph can't model;
    2. **variable-swap** — the wrong-variable bug, for the operator-free code
       (calls, assignments, returns) that dominates real corpora;
    3. **e-graph near-miss** — algebraic rewrites, as a last resort.

    When the realistic branch is skipped (probability ``1 - p_near_miss``) or no
    realistic mode applies (trivial functions: no flippable operator, < 2 in-scope
    names), it falls back to an out-of-context bank subtree swap so a corruption
    is always produced.

    ``p_near_miss <= 0`` reproduces the original bank-swap-only behavior *without
    consuming an rng draw* (short-circuit), preserving determinism of existing
    seeds.

    Returns ``(corrupted_source, mode)`` where ``mode`` is one of
    :data:`REALISTIC_MODES` or :data:`BANK_SWAP`.
    """
    if p_near_miss > 0 and rng.random() < p_near_miss:
        corrupted, _ = operator_flip_corrupt_program(source, num_steps=num_steps, rng=rng)
        if corrupted != source:
            return corrupted, "op-flip"
        corrupted, _ = variable_swap_corrupt_program(source, num_steps=num_steps, rng=rng)
        if corrupted != source:
            return corrupted, "var-swap"
        corrupted, _ = near_miss_corrupt_program(source, num_steps=num_steps, rng=rng)
        if corrupted != source:
            return corrupted, "near-miss"
    corrupted, _ = corrupt_program(
        source, num_steps=num_steps, bank=bank, max_edit_stmts=max_edit_stmts, rng=rng
    )
    return corrupted, BANK_SWAP
