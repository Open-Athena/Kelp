# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Kelp training: the autoregressive edit-model training engine.

Builds training examples from the corrupt/repair forward process, then trains
the causal edit-prediction model to predict the minimal edit back toward a
clean program.
"""

from kelp.training.engine import (
    EditTrainingConfig,
    EditTrainingState,
    create_edit_data_iter,
    create_edit_optimizer,
    generate_training_example,
    make_edit_train_step,
    train_edit_model,
)
from kelp.training.presets import PRESETS, ModelPreset, get_preset

__all__ = [
    "EditTrainingConfig",
    "EditTrainingState",
    "create_edit_optimizer",
    "make_edit_train_step",
    "generate_training_example",
    "create_edit_data_iter",
    "train_edit_model",
    "ModelPreset",
    "PRESETS",
    "get_preset",
]
