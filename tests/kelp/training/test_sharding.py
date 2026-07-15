# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for data-parallel sharding of the training step.

Correctness axes covered:
- The mesh/placement helpers produce the intended shardings.
- Wiring the sharded path introduces no numerical change vs. the plain path
  (verified on whatever devices are present; exact on 1 device).
- Multi-device data parallelism runs and the batch-divisibility contract holds
  (exercised only when >1 device is available).

The N-chip scaling itself is verified out-of-band by running with
XLA_FLAGS=--xla_force_host_platform_device_count=8 (see the module docstring
of kelp.training.sharding); that needs the flag set before JAX initializes, so
it is a standalone run rather than an in-process test.
"""

import jax
import jax.numpy as jnp
import pytest

from kelp.model.config import EditModelConfig
from kelp.model.edit_model import init_edit_params
from kelp.training.engine import (
    EditTrainingConfig,
    EditTrainingState,
    create_edit_data_iter,
    create_edit_optimizer,
    make_edit_train_step,
)
from kelp.training.sharding import (
    DATA_AXIS,
    data_parallel_size,
    make_data_parallel_mesh,
    replicate,
    shard_batch,
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


@pytest.fixture
def tokenizer():
    return EditTokenizer(max_seq_len=MAX_SEQ_LEN)


@pytest.fixture
def model_cfg(tokenizer):
    return EditModelConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=64,
        intermediate_dim=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        max_seq_len=MAX_SEQ_LEN,
    )


@pytest.fixture
def train_cfg(model_cfg):
    # batch_size 8 is divisible by common device counts (1, 2, 4, 8).
    return EditTrainingConfig(model=model_cfg, max_seq_len=MAX_SEQ_LEN, total_steps=2, batch_size=8, wandb_project=None)


def _one_batch(tokenizer, train_cfg):
    bank = SubtreeBank.from_corpus(CORPUS)
    return next(create_edit_data_iter(corpus=CORPUS, bank=bank, tokenizer=tokenizer, config=train_cfg))


def test_mesh_placement_helpers(model_cfg, tokenizer, train_cfg):
    """replicate spans all devices; shard_batch splits the batch axis."""
    mesh = make_data_parallel_mesh()
    n = data_parallel_size(mesh)
    assert n == len(jax.devices())

    params = init_edit_params(model_cfg, key=jax.random.PRNGKey(0))
    rep = replicate(params, mesh)
    # A replicated array is addressable on every device.
    assert len(rep.token_embed.sharding.device_set) == n

    batch = shard_batch(_one_batch(tokenizer, train_cfg), mesh)
    # The batch axis is partitioned over the data axis; values are preserved.
    assert batch["token_ids"].shape == (train_cfg.batch_size, MAX_SEQ_LEN)
    assert batch["token_ids"].sharding.spec[0] == DATA_AXIS


def test_sharded_step_matches_plain(model_cfg, tokenizer, train_cfg):
    """Sharding the step changes placement, not numerics."""
    params = init_edit_params(model_cfg, key=jax.random.PRNGKey(0))
    optimizer = create_edit_optimizer(train_cfg)
    opt_state = optimizer.init(params)
    key = jax.random.PRNGKey(0)
    state = EditTrainingState(step=0, params=params, opt_state=opt_state, key=key)
    batch = _one_batch(tokenizer, train_cfg)

    train_step = make_edit_train_step(model_cfg, optimizer)

    _, plain_metrics = train_step(state, batch)

    mesh = make_data_parallel_mesh()
    _, sharded_metrics = train_step(replicate(state, mesh), shard_batch(batch, mesh))

    for k in ("loss", "grad_norm", "accuracy"):
        assert jnp.allclose(plain_metrics[k], sharded_metrics[k], rtol=1e-5, atol=1e-5), k


def test_batch_not_divisible_raises(model_cfg):
    """A batch size not divisible by the device count is a clear error."""
    from kelp.training.engine import train_edit_model

    mesh = make_data_parallel_mesh()
    n = data_parallel_size(mesh)
    if n == 1:
        pytest.skip("divisibility only constrains multi-device meshes")
    cfg = EditTrainingConfig(
        model=model_cfg, max_seq_len=MAX_SEQ_LEN, total_steps=1, batch_size=n + 1, wandb_project=None
    )
    with pytest.raises(ValueError, match="divisible"):
        train_edit_model(config=cfg, data_iter=iter([]), mesh=mesh)
