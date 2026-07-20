# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the AR edit-prediction model."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from kelp.model.config import EditModelConfig
from kelp.model.edit_model import (
    _to_compute_dtype,
    ar_loss,
    forward,
    init_edit_params,
)


@pytest.fixture
def tiny_cfg():
    return EditModelConfig(
        vocab_size=128,
        hidden_dim=64,
        intermediate_dim=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        max_seq_len=64,
    )


@pytest.fixture
def params(tiny_cfg):
    key = jax.random.PRNGKey(0)
    return init_edit_params(tiny_cfg, key=key)


def test_init_params_shapes(params, tiny_cfg):
    assert params.token_embed.shape == (tiny_cfg.vocab_size, tiny_cfg.hidden_dim)
    assert params.output_proj.shape == (tiny_cfg.hidden_dim, tiny_cfg.vocab_size)
    assert params.final_norm.shape == (tiny_cfg.hidden_dim,)
    assert len(params.blocks) == tiny_cfg.num_layers


def test_init_params_no_timestep_embed(params):
    """EditModelParams should NOT have a timestep_embed field."""
    assert not hasattr(params, "timestep_embed")


def test_init_params_block_shapes(params, tiny_cfg):
    block = params.blocks[0]
    D = tiny_cfg.hidden_dim
    N = tiny_cfg.num_heads
    H = tiny_cfg.head_dim
    I = tiny_cfg.intermediate_dim  # noqa: E741 -- matches D/N/M/H dim naming

    assert block.attn.w_q.shape == (D, N * H)
    assert block.attn.w_k.shape == (D, N * H)  # num_kv_heads == num_heads
    assert block.attn.w_v.shape == (D, N * H)
    assert block.attn.w_o.shape == (N * H, D)
    assert block.mlp_gate.shape == (D, I)
    assert block.mlp_up.shape == (D, I)
    assert block.mlp_down.shape == (I, D)
    assert block.rms_attn.shape == (D,)
    assert block.rms_mlp.shape == (D,)


def test_forward_output_shape(params, tiny_cfg):
    batch_size, seq_len = 2, 16
    token_ids = jax.random.randint(jax.random.PRNGKey(1), (batch_size, seq_len), 1, 100)

    logits = forward(params, token_ids, tiny_cfg)
    assert logits.shape == (batch_size, seq_len, tiny_cfg.vocab_size)


def test_forward_output_dtype_is_float32(params, tiny_cfg):
    """Logits should always be float32 regardless of compute_dtype."""
    token_ids = jax.random.randint(jax.random.PRNGKey(1), (1, 8), 1, 100)
    logits = forward(params, token_ids, tiny_cfg)
    assert logits.dtype == jnp.float32


def test_forward_is_causal(params, tiny_cfg):
    """Verify that the model is causal: changing a future token should not
    affect the logits at an earlier position."""
    seq_len = 12
    token_ids = jax.random.randint(jax.random.PRNGKey(1), (1, seq_len), 1, 100)

    logits_a = forward(params, token_ids, tiny_cfg)

    # Change the last token.
    token_ids_b = token_ids.at[0, -1].set(50)
    logits_b = forward(params, token_ids_b, tiny_cfg)

    # All positions except the last should be identical.
    assert jnp.allclose(logits_a[0, :-1], logits_b[0, :-1], atol=1e-5)
    # The last position should differ.
    assert not jnp.allclose(logits_a[0, -1], logits_b[0, -1], atol=1e-5)


def test_fused_attention_matches_dense_at_real_positions(params, tiny_cfg):
    """The fused (AttentionMask/splash) path must be numerically identical to the
    dense reference path at every real (non-pad) position -- that equivalence is
    the whole safety case for swapping in the fused kernel. seq_len is a multiple
    of 128 so the same mask form is valid on TPU; on CPU both route to reference
    attention, which is exactly what makes this comparison meaningful."""
    seq_len = 128
    token_ids = jax.random.randint(jax.random.PRNGKey(3), (2, seq_len), 1, 100)
    # Mark the tail of row 0 as padding so the padding-segment logic is exercised.
    token_ids = token_ids.at[0, 100:].set(tiny_cfg.pad_token_id)
    not_pad = token_ids != tiny_cfg.pad_token_id

    dense = forward(params, token_ids, tiny_cfg, fused_attention=False)
    fused = forward(params, token_ids, tiny_cfg, fused_attention=True)

    real = not_pad[:, :, None]
    assert jnp.allclose(jnp.where(real, dense, 0.0), jnp.where(real, fused, 0.0), atol=1e-5)


def test_forward_padding_masked(params, tiny_cfg):
    """Padding tokens should not affect non-padding logits."""
    tokens = jnp.array([[10, 20, 30, 0, 0]])  # Last two are padding.
    logits_a = forward(params, tokens, tiny_cfg)

    tokens_b = jnp.array([[10, 20, 30, 50, 60]])  # No padding.
    logits_b = forward(params, tokens_b, tiny_cfg)

    # First 3 positions should be the same (padding doesn't attend).
    assert jnp.allclose(logits_a[0, :3], logits_b[0, :3], atol=1e-5)


def test_ar_loss_returns_scalar_and_metrics(params, tiny_cfg):
    batch_size, seq_len = 2, 16
    token_ids = jax.random.randint(jax.random.PRNGKey(1), (batch_size, seq_len), 1, 100)
    # Loss on the last 4 tokens only.
    loss_mask = jnp.zeros((batch_size, seq_len))
    loss_mask = loss_mask.at[:, -4:].set(1.0)

    loss, metrics = ar_loss(params, token_ids, loss_mask, tiny_cfg)

    assert loss.shape == ()
    assert loss.dtype == jnp.float32
    assert "accuracy" in metrics
    assert "perplexity" in metrics
    assert "num_loss_tokens" in metrics


def test_ar_loss_is_positive(params, tiny_cfg):
    batch_size, seq_len = 2, 16
    token_ids = jax.random.randint(jax.random.PRNGKey(1), (batch_size, seq_len), 1, 100)
    loss_mask = jnp.ones((batch_size, seq_len))

    loss, _ = ar_loss(params, token_ids, loss_mask, tiny_cfg)
    assert float(loss) > 0


def test_ar_loss_zero_mask_gives_zero_loss(params, tiny_cfg):
    batch_size, seq_len = 2, 16
    token_ids = jax.random.randint(jax.random.PRNGKey(1), (batch_size, seq_len), 1, 100)
    loss_mask = jnp.zeros((batch_size, seq_len))

    loss, _ = ar_loss(params, token_ids, loss_mask, tiny_cfg)
    assert float(loss) == 0.0


def test_ar_loss_grad_flows(params, tiny_cfg):
    """Verify gradients flow through the loss."""
    batch_size, seq_len = 1, 8
    token_ids = jax.random.randint(jax.random.PRNGKey(1), (batch_size, seq_len), 1, 100)
    loss_mask = jnp.ones((batch_size, seq_len))

    def loss_fn(p):
        loss, _ = ar_loss(p, token_ids, loss_mask, tiny_cfg)
        return loss

    grads = jax.grad(loss_fn)(params)
    # Check that gradients are non-zero for at least some params.
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
    assert float(grad_norm) > 0


def test_float32_compute_dtype_is_noop_cast(params, tiny_cfg):
    """The default float32 path must be byte-for-byte unchanged: casting to the
    compute dtype returns the params object untouched (no spurious copies/casts)."""
    assert tiny_cfg.compute_dtype == "float32"
    assert _to_compute_dtype(params, jnp.float32) is params


def test_bf16_compute_keeps_logits_and_master_grads_float32(tiny_cfg):
    """Mixed-precision contract: with bf16 compute the matmul weights run in bf16,
    but the output logits stay float32 (stability) and gradients land on the
    float32 master weights (the optimizer never sees bf16)."""
    cfg = replace(tiny_cfg, compute_dtype="bfloat16")
    params = init_edit_params(cfg, key=jax.random.PRNGKey(0))
    assert params.blocks[0].attn.w_q.dtype == jnp.float32  # master weights fp32

    token_ids = jax.random.randint(jax.random.PRNGKey(1), (1, 8), 1, 100)
    loss_mask = jnp.ones((1, 8))

    logits = forward(params, token_ids, cfg)
    assert logits.dtype == jnp.float32

    grads = jax.grad(lambda p: ar_loss(p, token_ids, loss_mask, cfg)[0])(params)
    assert grads.blocks[0].attn.w_q.dtype == jnp.float32
    assert grads.blocks[0].mlp_down.dtype == jnp.float32
