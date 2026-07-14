# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Kelp model: transformer config, layers, the assembled autoregressive
edit-prediction model, and checkpoint I/O."""

from kelp.model.checkpointing import find_best_checkpoint, load_checkpoint, save_checkpoint
from kelp.model.config import EditModelConfig
from kelp.model.edit_model import EditModelParams, ar_loss, forward, init_edit_params
from kelp.model.layers import (
    AttentionParams,
    TransformerBlockParams,
    init_weight,
    rms_norm,
    swiglu_mlp,
)

__all__ = [
    "EditModelConfig",
    "EditModelParams",
    "init_edit_params",
    "forward",
    "ar_loss",
    "AttentionParams",
    "TransformerBlockParams",
    "rms_norm",
    "swiglu_mlp",
    "init_weight",
    "save_checkpoint",
    "load_checkpoint",
    "find_best_checkpoint",
]
