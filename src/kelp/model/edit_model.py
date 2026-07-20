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

"""Autoregressive edit-prediction model for tree diffusion.

A causal transformer that predicts single program edits: given the current
program as context, it autoregressively generates a position token (WHERE to
edit) followed by replacement tokens (WHAT to insert).

Key differences from the old D3PM model:
1. Causal attention (standard LLM) instead of bidirectional
2. No timestep embedding (tree diffusion doesn't use a fixed schedule)
3. Predicts one edit at a time, not all tokens simultaneously

The transformer backbone (blocks, attention, MLP, norms, RoPE) is identical
to Grug and can be initialized directly from pretrained LLM weights.
"""

import dataclasses
import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from einops import rearrange
from jax import random
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, Int, PRNGKeyArray
from levanter.grug.attention import (
    AttentionMask,
)
from levanter.grug.attention import (
    apply_rotary_embedding as grug_apply_rotary,
)
from levanter.grug.attention import (
    attention as grug_attention,
)

from kelp.model.config import EditModelConfig
from kelp.model.layers import (
    AttentionParams,
    TransformerBlockParams,
    init_weight,
    rms_norm,
    swiglu_mlp,
)

logger = logging.getLogger(__name__)


@register_dataclass
@dataclass(frozen=True)
class EditModelParams:
    """Parameters for the AR edit-prediction model.

    Contains token embeddings, output projection, transformer blocks, and
    final layer norm. No timestep conditioning -- tree diffusion uses
    iterative edits rather than a fixed noise schedule.
    """

    token_embed: jax.Array
    output_proj: jax.Array
    blocks: tuple[TransformerBlockParams, ...]
    final_norm: jax.Array


def init_edit_params(cfg: EditModelConfig, *, key: PRNGKeyArray) -> EditModelParams:
    """Initialize edit model parameters.

    Args:
        cfg: Model configuration.
        key: PRNG key.

    Returns:
        Initialized EditModelParams.
    """
    head_dim = cfg.head_dim
    key, embed_key, out_key = random.split(key, 3)
    layer_keys = random.split(key, cfg.num_layers)

    token_embed = init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std)
    output_proj = init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std)
    final_norm = jnp.ones((cfg.hidden_dim,), dtype=jnp.float32)

    blocks: list[TransformerBlockParams] = []
    D, N, M, H, I = cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, head_dim, cfg.intermediate_dim  # noqa: E741 -- matrix dims match D/N/M/H single-letter convention

    for i in range(cfg.num_layers):
        k_q, k_k, k_v, k_o, k_gate, k_up, k_down = random.split(layer_keys[i], 7)

        attn = AttentionParams(
            w_q=init_weight(k_q, (D, N * H), cfg.initializer_std),
            w_k=init_weight(k_k, (D, M * H), cfg.initializer_std),
            w_v=init_weight(k_v, (D, M * H), cfg.initializer_std),
            w_o=init_weight(k_o, (N * H, D), cfg.initializer_std),
        )

        blocks.append(
            TransformerBlockParams(
                attn=attn,
                rms_attn=jnp.ones((D,), dtype=jnp.float32),
                rms_mlp=jnp.ones((D,), dtype=jnp.float32),
                mlp_gate=init_weight(k_gate, (D, I), cfg.initializer_std),
                mlp_up=init_weight(k_up, (D, I), cfg.initializer_std),
                mlp_down=init_weight(k_down, (I, D), cfg.initializer_std),
            )
        )

    return EditModelParams(
        token_embed=token_embed,
        output_proj=output_proj,
        blocks=tuple(blocks),
        final_norm=final_norm,
    )


def _to_compute_dtype(params: EditModelParams, dtype: jnp.dtype) -> EditModelParams:
    """Cast the matmul weights (attention + MLP projections) to ``dtype``.

    This is the mechanism that makes ``compute_dtype='bfloat16'`` actually run
    bf16 matmuls: the master weights stay float32 (the optimizer needs them), but
    the forward/backward matmul *operands* are cast here. Without this, a bf16
    activation times an fp32 weight promotes back to fp32 (JAX's type-promotion
    lattice), so the MXU never sees a bf16xbf16 matmul and the flag is a no-op.

    Normalization weights and the output projection are deliberately left in
    float32 -- ``rms_norm`` accumulates in float32 and the vocab logits stay
    float32 for numerical stability (both are a negligible share of FLOPs). When
    ``dtype`` is float32 every cast is an identity XLA elides, so the float32
    path is byte-for-byte unchanged. Gradients flow back through the casts to the
    float32 master weights (the cast's transpose upcasts them).
    """
    if dtype == jnp.float32:
        return params

    def cast_block(b: TransformerBlockParams) -> TransformerBlockParams:
        return TransformerBlockParams(
            attn=AttentionParams(
                w_q=b.attn.w_q.astype(dtype),
                w_k=b.attn.w_k.astype(dtype),
                w_v=b.attn.w_v.astype(dtype),
                w_o=b.attn.w_o.astype(dtype),
            ),
            rms_attn=b.rms_attn,
            rms_mlp=b.rms_mlp,
            mlp_gate=b.mlp_gate.astype(dtype),
            mlp_up=b.mlp_up.astype(dtype),
            mlp_down=b.mlp_down.astype(dtype),
        )

    return dataclasses.replace(params, blocks=tuple(cast_block(b) for b in params.blocks))


def forward(
    params: EditModelParams,
    token_ids: Int[Array, "B S"],
    cfg: EditModelConfig,
    *,
    fused_attention: bool = False,
) -> Float[Array, "B S V"]:
    """Causal AR forward pass for edit prediction.

    Args:
        params: Model parameters.
        token_ids: Input token IDs. The sequence is
            [context..., SOS, POS, replacement..., EOS, PAD...].
        cfg: Model configuration.
        fused_attention: Select the attention path. False (default) builds a
            dense causal+padding mask and takes grug's reference attention --
            correct on any backend/sequence length with no mesh context, used by
            inference (variable bucketed seq, single device). True expresses the
            mask as a structured ``AttentionMask`` so grug picks the fused TPU
            splash kernel (O(seq) memory instead of materializing the O(seq^2)
            score matrix). Splash requires a JAX mesh context and seq % 128 == 0,
            both of which hold on the training path; the two paths are
            numerically identical at real (non-pad) positions.

    Returns:
        Logits of shape (batch, seq, vocab). For training, shift by 1
        to predict the next token at each position.
    """
    compute_dtype = jnp.dtype(cfg.compute_dtype)
    head_dim = cfg.head_dim
    _batch_size, seq_len = token_ids.shape

    # Cast matmul weights to the compute dtype so the MXU sees genuine
    # bf16xbf16 matmuls (see _to_compute_dtype). No-op when compute_dtype is
    # float32.
    params = _to_compute_dtype(params, compute_dtype)
    hidden = params.token_embed[token_ids].astype(compute_dtype)

    if fused_attention:
        # Structured mask -> fused TPU splash kernel. Segment IDs: real tokens
        # share segment 1, PAD tokens segment 0, so real tokens attend only
        # within the real segment (and causally). PAD query rows produce junk but
        # are zeroed by the loss mask downstream.
        segment_ids = (token_ids != cfg.pad_token_id).astype(jnp.int32)
        attn_mask: AttentionMask | jax.Array = AttentionMask.causal().with_segment_ids(segment_ids, segment_ids)
    else:
        # Dense (B, S, S) mask -> reference attention: causal AND both positions
        # non-padding. Materializes the score matrix but needs no mesh and works
        # for any seq length.
        causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        not_pad = token_ids != cfg.pad_token_id
        attn_mask = causal[None, :, :] & not_pad[:, None, :] & not_pad[:, :, None]

    def _block_fn(hidden, block):
        attn_in = rms_norm(hidden, block.rms_attn, cfg.layer_norm_eps)

        q = rearrange(
            jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_q),
            "b s (n d) -> b s n d",
            d=head_dim,
        )
        k = rearrange(
            jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_k),
            "b s (m d) -> b s m d",
            d=head_dim,
        )
        v = rearrange(
            jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_v),
            "b s (m d) -> b s m d",
            d=head_dim,
        )

        q, k = grug_apply_rotary(q, k, seq_len=seq_len, head_dim=head_dim, rope=cfg.rope)

        # Causal attention with padding mask.
        attn_out = grug_attention(q, k, v, mask=attn_mask)
        attn_out = rearrange(attn_out, "b s n d -> b s (n d)")
        attn_out = jnp.einsum("bsh,hd->bsd", attn_out, block.attn.w_o)

        hidden = hidden + attn_out

        mlp_in = rms_norm(hidden, block.rms_mlp, cfg.layer_norm_eps)
        mlp_out = swiglu_mlp(mlp_in, block.mlp_gate, block.mlp_up, block.mlp_down)
        hidden = hidden + mlp_out
        return hidden

    block_fn = jax.checkpoint(_block_fn) if cfg.gradient_checkpointing else _block_fn

    for block in params.blocks:
        hidden = block_fn(hidden, block)

    # Project to vocab logits in float32 for numerical stability.
    hidden = rms_norm(hidden, params.final_norm, cfg.layer_norm_eps)
    logits = jnp.einsum("bsh,hv->bsv", hidden.astype(jnp.float32), params.output_proj)
    return logits


def ar_loss(
    params: EditModelParams,
    token_ids: Int[Array, "B S"],
    loss_mask: Float[Array, "B S"],
    cfg: EditModelConfig,
    *,
    fused_attention: bool = False,
) -> tuple[Float[Array, ""], dict]:
    """Compute AR cross-entropy loss on edit predictions.

    Standard next-token prediction loss, but only on the edit portion
    of the sequence (position token + replacement tokens + EOS),
    controlled by loss_mask.

    Args:
        params: Model parameters.
        token_ids: Full sequence [context, SOS, POS, replacement, EOS, PAD...].
        loss_mask: Float mask, 1.0 for tokens that contribute to loss
            (POS, replacement, EOS), 0.0 for context and padding.
        cfg: Model configuration.
        fused_attention: Passed through to :func:`forward` -- True on the
            training path to select the fused TPU splash-attention kernel.

    Returns:
        Tuple of (scalar_loss, metrics_dict).
    """
    logits = forward(params, token_ids, cfg, fused_attention=fused_attention)

    # Shift: predict token at position i+1 from logits at position i.
    shifted_logits = logits[:, :-1, :]
    shifted_targets = token_ids[:, 1:]
    shifted_mask = loss_mask[:, 1:]

    log_probs = jax.nn.log_softmax(shifted_logits, axis=-1)
    target_log_probs = jnp.take_along_axis(log_probs, shifted_targets[..., None], axis=-1).squeeze(-1)

    masked_loss = -target_log_probs * shifted_mask
    num_loss_tokens = jnp.sum(shifted_mask)
    loss = jnp.sum(masked_loss) / jnp.maximum(num_loss_tokens, 1.0)

    # Metrics.
    predictions = jnp.argmax(shifted_logits, axis=-1)
    correct = (predictions == shifted_targets).astype(jnp.float32) * shifted_mask
    accuracy = jnp.sum(correct) / jnp.maximum(num_loss_tokens, 1.0)

    metrics = {
        "loss": loss,
        "accuracy": accuracy,
        "perplexity": jnp.exp(loss),
        "num_loss_tokens": num_loss_tokens,
    }

    return loss, metrics
