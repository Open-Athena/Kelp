# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared corruption policy (train/eval single source of truth)."""

import random

from kelp.tree.corruption import (
    BANK_SWAP,
    REALISTIC_MODES,
    SKIPPED,
    corrupt_realistic,
    has_corruptible_content,
)
from kelp.tree.subtree_bank import SubtreeBank

CORPUS = [
    "def is_even(x):\n    return x % 2 == 0\n",
    "def add(a, b):\n    return a + b\n",
    "def pick(lo, hi):\n    return max(lo, hi)\n",
]


def _bank():
    return SubtreeBank.from_corpus(CORPUS)


def test_p_near_miss_one_prefers_realistic_mode():
    """With p_near_miss=1.0, a program with a flippable operator is corrupted
    in-context (a realistic mode), never via an alien bank swap."""
    bank = _bank()
    src = "def f(x):\n    return x % 2 == 0\n"
    corrupted, mode = corrupt_realistic(src, num_steps=1, bank=bank, rng=random.Random(0), p_near_miss=1.0)
    assert corrupted != src
    assert mode in REALISTIC_MODES


def test_falls_back_to_bank_swap_when_no_realistic_mode_applies():
    """A trivial function (no operator, single in-scope name) has no realistic
    corruption available, so even at p_near_miss=1.0 it falls back to bank-swap."""
    bank = _bank()
    src = "def f(self):\n    return self.compute()\n"
    _corrupted, mode = corrupt_realistic(src, num_steps=1, bank=bank, rng=random.Random(0), p_near_miss=1.0)
    assert mode == BANK_SWAP


def test_no_bank_swap_fallback_skips_trivial_program_unchanged():
    """With allow_bank_swap=False, a trivial program is returned unchanged and
    marked SKIPPED (the caller drops it) -- never an alien graft."""
    bank = _bank()
    src = "def f(self):\n    return self.compute()\n"
    corrupted, mode = corrupt_realistic(
        src, num_steps=1, bank=bank, rng=random.Random(0), p_near_miss=1.0, allow_bank_swap=False
    )
    assert mode == SKIPPED
    assert corrupted == src


def test_no_bank_swap_fallback_attempts_cascade_regardless_of_p_near_miss():
    """allow_bank_swap=False means realistic-only: the cascade is attempted even
    at p_near_miss=0.0, so a corruptible program is still corrupted in-context."""
    bank = _bank()
    src = "def f(x):\n    return x % 2 == 0\n"
    corrupted, mode = corrupt_realistic(
        src, num_steps=1, bank=bank, rng=random.Random(0), p_near_miss=0.0, allow_bank_swap=False
    )
    assert mode in REALISTIC_MODES
    assert corrupted != src


def test_has_corruptible_content():
    """True when a flippable operator or >=2 in-scope names exist; False for
    trivial single-name / operator-free bodies."""
    assert has_corruptible_content("def f(x):\n    return x % 2 == 0\n")  # operator
    assert has_corruptible_content("def f(a, b):\n    return g(a, b)\n")  # two names
    assert not has_corruptible_content("def f(self):\n    return self.compute()\n")  # trivial
    assert not has_corruptible_content("def f(x):\n    return x\n")  # single name, no op


def test_p_near_miss_zero_is_deterministic_bank_swap_without_extra_draw():
    """p_near_miss=0.0 must reproduce plain bank-swap AND not consume an rng
    draw for the (skipped) near-miss coin -- so pre-existing seeds are stable."""
    bank = _bank()
    src = CORPUS[0]
    # Same seed through corrupt_realistic(p=0) and a bare corrupt_program must agree.
    from kelp.tree.mutation import corrupt_program

    c1, mode = corrupt_realistic(src, num_steps=2, bank=bank, rng=random.Random(7), p_near_miss=0.0)
    c2, _ = corrupt_program(src, num_steps=2, bank=bank, rng=random.Random(7))
    assert c1 == c2
    assert mode == BANK_SWAP
