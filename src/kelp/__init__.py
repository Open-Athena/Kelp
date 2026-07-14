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

"""Kelp: tree diffusion for program repair.

Kelp repairs corrupted Python by predicting structured edits at the AST level,
so every intermediate state stays syntactically valid. The package is layered:

- ``kelp.tree``      — AST representation and the corruption (forward) process
- ``kelp.model``     — the causal edit-prediction transformer + checkpoint I/O
- ``kelp.inference`` — iterative repair (best-of-N, beam search, reranking)
- ``kelp.training``  — the training engine
- ``kelp.cli``       — command-line entry points

See docs/kelp.md for the full design document.
"""

# Top-level public API. Importing kelp gives access to the whole pipeline —
# build/augment a bank, corrupt a program, train a model, and repair a program —
# without needing to know the internal module layout. The layered subpackages
# (kelp.tree, kelp.model, kelp.inference, kelp.training) remain the place to
# reach for anything not re-exported here.
from kelp.inference import beam_search, best_of_n, rerank_candidates
from kelp.model import (
    EditModelConfig,
    EditModelParams,
    ar_loss,
    forward,
    init_edit_params,
    load_checkpoint,
    save_checkpoint,
)
from kelp.training import EditTrainingConfig, train_edit_model
from kelp.tree import (
    EditTokenizer,
    Mutation,
    SubtreeBank,
    augment_bank,
    corrupt_program,
    tree_diff,
)

__all__ = [
    # kelp.tree — representation & corruption
    "SubtreeBank",
    "Mutation",
    "corrupt_program",
    "tree_diff",
    "augment_bank",
    "EditTokenizer",
    # kelp.model — architecture & checkpoints
    "EditModelConfig",
    "EditModelParams",
    "init_edit_params",
    "forward",
    "ar_loss",
    "save_checkpoint",
    "load_checkpoint",
    # kelp.inference — repair
    "best_of_n",
    "beam_search",
    "rerank_candidates",
    # kelp.training
    "EditTrainingConfig",
    "train_edit_model",
]
