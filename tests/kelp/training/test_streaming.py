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
from kelp.training.streaming import ReuseShuffleBuffer, create_streaming_data_iter
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


def test_reuse_buffer_feeds_each_example_reuse_factor_times():
    """With reuse_factor R and a finite source of N distinct examples, each is
    drawn exactly R times."""
    examples = [_ex(1, tag=i) for i in range(5)]
    buf = ReuseShuffleBuffer(iter(examples), capacity=5, reuse_factor=3, rng=random.Random(0))

    drawn = []
    try:
        while True:
            drawn.append(buf.draw(lambda ex: True))
    except StopIteration:
        pass

    assert len(drawn) == 5 * 3
    counts = Counter(id(ex) for ex in drawn)
    assert set(counts.values()) == {3}  # every example exactly 3 times


def test_curriculum_gate_only_yields_within_difficulty():
    """The gate admits only examples at or below the effective difficulty
    (random-branch examples always pass)."""
    examples = [_ex(steps, tag=steps) for steps in (1, 2, 3, 4, 5)] + [_ex(0, is_random=True, tag=99)]
    buf = ReuseShuffleBuffer(iter(examples), capacity=6, reuse_factor=1, rng=random.Random(0))

    effective_max = 2
    eligible = lambda ex: ex.is_random or ex.corruption_steps <= effective_max  # noqa: E731

    drawn = []
    try:
        while True:
            drawn.append(buf.draw(eligible))
    except StopIteration:
        pass

    # Only corruption_steps 1,2 and the random example are eligible; 3,4,5 stay out.
    assert all(ex.is_random or ex.corruption_steps <= 2 for ex in drawn)
    assert {ex.corruption_steps for ex in drawn if not ex.is_random} == {1, 2}
    assert any(ex.is_random for ex in drawn)


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
