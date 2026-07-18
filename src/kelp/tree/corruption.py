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
    flip_one_operator,
    operator_flip_corrupt_program,
    swap_one_variable,
    variable_swap_corrupt_program,
)
from kelp.tree.subtree_bank import SubtreeBank

# Corruption mode labels, useful for gut-check rendering and per-type eval
# breakdowns. "bank-swap" is the out-of-context fallback; "skipped" means no
# corruption was produced (caller should drop the example); the rest are the
# realistic in-context modes.
REALISTIC_MODES = ("op-flip", "var-swap", "near-miss")
BANK_SWAP = "bank-swap"
SKIPPED = "skipped"


def corrupt_realistic(
    source: str,
    *,
    num_steps: int,
    bank: SubtreeBank,
    rng: random.Random,
    p_near_miss: float,
    max_edit_stmts: int = 3,
    allow_bank_swap: bool = True,
) -> tuple[str, str]:
    """Corrupt ``source``, preferring realistic in-context bugs.

    With probability ``p_near_miss`` the realistic cascade is tried, most- to
    least-applicable, using the first mode that actually changes the program:

    1. **operator-flip** — a plausible single-operator bug (``+``→``*``,
       ``==``→``!=``), fires even inside calls/subscripts the e-graph can't model;
    2. **variable-swap** — the wrong-variable bug, for the operator-free code
       (calls, assignments, returns) that dominates real corpora;
    3. **e-graph near-miss** — algebraic rewrites, as a last resort.

    When no realistic mode applies (trivial functions: no flippable operator,
    < 2 in-scope names) the fallback depends on ``allow_bank_swap``:

    - ``True`` (default): fall back to an out-of-context bank subtree swap so a
      corruption is always produced (returns mode :data:`BANK_SWAP`).
    - ``False``: produce **no** corruption — return ``(source, SKIPPED)`` so the
      caller drops the example. This eliminates alien out-of-context grafts
      entirely; combined with the cascade it yields realistic-or-drop training.
      In this mode the realistic cascade is *always* attempted regardless of
      ``p_near_miss`` (skipping it would drop corruptible programs for no gain).

    ``allow_bank_swap=True`` with ``p_near_miss <= 0`` reproduces the original
    bank-swap-only behavior *without consuming an rng draw* (short-circuit),
    preserving determinism of existing seeds.

    Returns ``(corrupted_source, mode)`` where ``mode`` is one of
    :data:`REALISTIC_MODES`, :data:`BANK_SWAP`, or :data:`SKIPPED`.
    """
    attempt_realistic = (not allow_bank_swap) or (p_near_miss > 0 and rng.random() < p_near_miss)
    if attempt_realistic:
        corrupted, _ = operator_flip_corrupt_program(source, num_steps=num_steps, rng=rng)
        if corrupted != source:
            return corrupted, "op-flip"
        corrupted, _ = variable_swap_corrupt_program(source, num_steps=num_steps, rng=rng)
        if corrupted != source:
            return corrupted, "var-swap"
        corrupted, _ = near_miss_corrupt_program(source, num_steps=num_steps, rng=rng)
        if corrupted != source:
            return corrupted, "near-miss"
    if not allow_bank_swap:
        return source, SKIPPED
    corrupted, _ = corrupt_program(source, num_steps=num_steps, bank=bank, max_edit_stmts=max_edit_stmts, rng=rng)
    return corrupted, BANK_SWAP


def has_corruptible_content(source: str) -> bool:
    """True if a realistic in-context corruption can be produced for ``source``.

    A program is corruptible-realistic when it has a flippable operator or at
    least two distinct in-scope names to swap between. Trivial functions
    (abstract stubs, one-line wrappers, single-name / docstring-only bodies) have
    neither, so the only corruption available for them is an alien bank swap.

    Used at corpus-prep time to drop such programs, so they never enter the
    corpus and waste sampling that would only be dropped again at generation.
    (Existence check only; the returned corruption is discarded.)
    """
    probe = random.Random(0)
    return flip_one_operator(source, probe) is not None or swap_one_variable(source, probe) is not None
