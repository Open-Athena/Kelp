# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the streaming (seed-driven) training dataloader.

Orthogonal axes:
- The reuse-shuffle buffer feeds each example exactly reuse_factor times.
- The curriculum gate only yields examples within the current difficulty.
- The inline iterator matches the create_edit_data_iter batch contract and is
  deterministic per seed.

The multiprocess generation source is smoke-verified out-of-band (it spawns
processes that re-import JAX, which is slow/heavy for CI); the deterministic
buffering/gating/batching logic above is what these tests pin down.
"""

import random
from collections import Counter

import jax.numpy as jnp
import pytest

from kelp.model.config import EditModelConfig
from kelp.training.engine import EditTrainingConfig
from kelp.training.generation import TrainingExample
from kelp.training.streaming import (
    ReuseShuffleBuffer,
    _all_eligible,
    _within_difficulty,
    create_streaming_data_iter,
)
from kelp.tree.subtree_bank import SubtreeBank
from kelp.tree.tokenizer import EditTokenizer

CORPUS = [
    "def add(a, b):\n    return a + b\n",
    "def sub(a, b):\n    return a - b\n",
    "def mul(a, b):\n    return a * b\n",
    "def square(x):\n    return x * x\n",
]
MAX_SEQ_LEN = 128


def _ex(corruption_steps: int, *, is_random: bool = False, tag: int = 0) -> TrainingExample:
    return TrainingExample(
        token_ids=[3, tag, tag],
        loss_mask=[0, 1, 1],
        corruption_steps=corruption_steps,
        is_random=is_random,
        prompt_used=False,
    )


def _drain(buf, eligible):
    """Mimic the consumer loop: take an entry, spend one use, put back if any
    remain -- returning examples until a finite source is exhausted."""
    out = []
    while True:
        try:
            entry = buf.take(eligible)
        except StopIteration:
            return out
        out.append(entry[0])
        entry[1] -= 1
        if entry[1] > 0:
            buf.put(entry)


def test_reuse_buffer_feeds_each_example_reuse_factor_times():
    """With reuse_factor R and a finite source of N distinct examples, each is
    fed exactly R times."""
    examples = [_ex(1, tag=i) for i in range(5)]
    buf = ReuseShuffleBuffer(iter(examples), capacity=5, reuse_factor=3, rng=random.Random(0))

    drawn = _drain(buf, _all_eligible)

    assert len(drawn) == 5 * 3
    counts = Counter(id(ex) for ex in drawn)
    assert set(counts.values()) == {3}  # every example exactly 3 times


def test_curriculum_gate_only_yields_within_difficulty():
    """The gate admits only examples at or below the effective difficulty
    (random-branch examples always pass)."""
    examples = [_ex(steps, tag=steps) for steps in (1, 2, 3, 4, 5)] + [_ex(0, is_random=True, tag=99)]
    buf = ReuseShuffleBuffer(iter(examples), capacity=6, reuse_factor=1, rng=random.Random(0))

    drawn = _drain(buf, _within_difficulty(2))

    # Only corruption_steps 1,2 and the random example are eligible; 3,4,5 stay out.
    assert all(ex.is_random or ex.corruption_steps <= 2 for ex in drawn)
    assert {ex.corruption_steps for ex in drawn if not ex.is_random} == {1, 2}
    assert any(ex.is_random for ex in drawn)


def test_buffer_bounded_under_active_gate():
    """On an INFINITE source with a strict gate rejecting most examples, memory
    stays O(capacity) -- the reservoir must evict saturating ineligibles, not
    grow (regression for the unbounded-growth / OOM bug)."""

    def infinite_source():
        rng = random.Random(0)
        i = 0
        while True:
            yield _ex(rng.randint(1, 20), tag=i)  # ceiling=20, uniform difficulty
            i += 1

    cap = 128
    buf = ReuseShuffleBuffer(infinite_source(), capacity=cap, reuse_factor=1, rng=random.Random(1))
    gate = _within_difficulty(1)  # early curriculum: only depth-1 eligible (~1/20)
    for _ in range(3000):
        entry = buf.take(gate)
        entry[1] -= 1  # reuse_factor=1 -> spent, not returned
    assert len(buf._items) <= 4 * cap, f"reservoir grew to {len(buf._items)} (expected O({cap}))"


def test_batch_takes_distinct_entries_with_reuse():
    """Even with reuse_factor>1, one batch of take()s draws DISTINCT entries
    (feeding the same example twice in one gradient step would over-weight it);
    put() then returns them so reuse spreads across later batches."""
    examples = [_ex(1, tag=i) for i in range(20)]
    buf = ReuseShuffleBuffer(iter(examples), capacity=20, reuse_factor=4, rng=random.Random(0))

    entries = [buf.take(_all_eligible) for _ in range(8)]  # one batch
    assert len({id(e) for e in entries}) == 8, "duplicate entry within a batch"

    for e in entries:  # spend one use and return (reuse across batches)
        e[1] -= 1
        buf.put(e)
    # A subsequent batch is drawable (entries came back with uses remaining).
    assert len([buf.take(_all_eligible) for _ in range(8)]) == 8


@pytest.fixture
def train_cfg():
    tok = EditTokenizer(max_seq_len=MAX_SEQ_LEN)
    model = EditModelConfig(
        vocab_size=tok.vocab_size,
        hidden_dim=32,
        intermediate_dim=64,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=MAX_SEQ_LEN,
    )
    return EditTrainingConfig(model=model, max_seq_len=MAX_SEQ_LEN, batch_size=4, wandb_project=None)


def _iter(train_cfg, **kw):
    bank = SubtreeBank.from_corpus(CORPUS)
    tok = EditTokenizer(max_seq_len=MAX_SEQ_LEN)
    return create_streaming_data_iter(CORPUS, bank, tok, train_cfg, buffer_size=32, **kw)


def test_inline_iter_matches_batch_contract(train_cfg):
    """Streaming batches have the same shapes/dtypes as create_edit_data_iter."""
    batch = next(_iter(train_cfg, seed=1))
    assert batch["token_ids"].shape == (train_cfg.batch_size, MAX_SEQ_LEN)
    assert batch["loss_mask"].shape == (train_cfg.batch_size, MAX_SEQ_LEN)
    assert batch["token_ids"].dtype == jnp.int32
    assert batch["loss_mask"].dtype == jnp.float32


def test_inline_iter_deterministic_per_seed(train_cfg):
    """Same seed -> identical batches (reproducible from corpus+seed alone)."""
    b1 = next(_iter(train_cfg, seed=7))
    b2 = next(_iter(train_cfg, seed=7))
    assert jnp.array_equal(b1["token_ids"], b2["token_ids"])
    assert jnp.array_equal(b1["loss_mask"], b2["loss_mask"])
