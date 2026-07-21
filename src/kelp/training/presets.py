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

"""Model size and resource presets for Kelp tree diffusion.

Provides pre-configured model sizes targeting different compute environments.
"""

from dataclasses import dataclass

from fray.cluster import ResourceConfig

from kelp.model.config import EditModelConfig


@dataclass(frozen=True)
class ModelPreset:
    """A preset combining model config with training resources."""

    name: str
    """Human-readable name."""

    config: EditModelConfig
    """Model configuration."""

    resource: ResourceConfig
    """Compute resources for training."""

    batch_size: int
    """Global batch size."""

    learning_rate: float
    """Base learning rate."""

    description: str = ""
    """Description of use case."""


# Default vocab size (LLaMA-3 tokenizer)
DEFAULT_VOCAB_SIZE = 128256


def toy_preset() -> ModelPreset:
    """Tiny preset for unit testing (~0.2M params)."""
    return ModelPreset(
        name="toy",
        config=EditModelConfig(
            vocab_size=DEFAULT_VOCAB_SIZE,
            hidden_dim=64,
            intermediate_dim=256,
            num_layers=2,
            num_heads=2,
            num_kv_heads=2,
            max_seq_len=128,
        ),
        resource=ResourceConfig.with_cpu(cpu=2),
        batch_size=2,
        learning_rate=1e-3,
        description="Toy model for testing infrastructure",
    )


def overnight_cpu_preset() -> ModelPreset:
    """Preset optimized for overnight CPU training (~4.6M params).

    Designed to complete ~30k+ steps in 8 hours on a laptop CPU.
    """
    return ModelPreset(
        name="overnight_cpu",
        config=EditModelConfig(
            vocab_size=256,  # Byte-level tokenizer for faster training
            hidden_dim=256,
            intermediate_dim=1024,
            num_layers=4,
            num_heads=4,
            num_kv_heads=4,
            max_seq_len=512,
        ),
        resource=ResourceConfig.with_cpu(cpu=8),
        batch_size=16,
        learning_rate=1e-3,
        description="For overnight CPU training runs",
    )


def laptop_preset() -> ModelPreset:
    """Small preset for laptop development (~27M params)."""
    return ModelPreset(
        name="laptop",
        config=EditModelConfig(
            vocab_size=DEFAULT_VOCAB_SIZE,
            hidden_dim=512,
            intermediate_dim=2048,
            num_layers=6,
            num_heads=8,
            num_kv_heads=8,
            max_seq_len=1024,
        ),
        resource=ResourceConfig.with_cpu(cpu=8),
        batch_size=4,
        learning_rate=3e-4,
        description="For CPU/laptop iteration",
    )


def single_gpu_preset() -> ModelPreset:
    """Medium preset for single GPU (~117M params)."""
    return ModelPreset(
        name="single_gpu",
        config=EditModelConfig(
            vocab_size=DEFAULT_VOCAB_SIZE,
            hidden_dim=768,
            intermediate_dim=3072,
            num_layers=12,
            num_heads=12,
            num_kv_heads=12,
            max_seq_len=2048,
        ),
        resource=ResourceConfig.with_gpu("a100-40gb", count=1),
        batch_size=16,
        learning_rate=3e-4,
        description="For single A100 training",
    )


def tpu_smoke_preset() -> ModelPreset:
    """Smallest TPU slice (v6e-4) with a tiny model for validating the launch path.

    Cheap end-to-end check of distributed init + data-parallel sharding + GCS
    checkpointing on real hardware -- not a research-scale run. batch_size is
    divisible by the 4 chips.
    """
    return ModelPreset(
        name="tpu_smoke",
        config=EditModelConfig(
            vocab_size=256,  # byte-level; overridden to tokenizer vocab in train.py
            hidden_dim=256,
            intermediate_dim=1024,
            num_layers=4,
            num_heads=4,
            num_kv_heads=4,
            max_seq_len=256,
        ),
        resource=ResourceConfig.with_tpu("v6e-4"),
        batch_size=8,
        learning_rate=1e-3,
        description="Smallest TPU slice (v6e-4) for smoke-testing the launch path",
    )


def tpu_vet_preset() -> ModelPreset:
    """~115M model on the cheapest validated TPU slice (v6e-4) for data-scaling vet runs.

    The research-credible middle between ``tpu_smoke`` (a ~5M toy for path
    validation) and ``tpu_v4_8`` (~1.6B). At ~115M it is ~15x the ~5-10M model
    whose capacity starved under the v5 corpus scale-up, yet small enough to
    train cheaply (a 30K-step run is a few chip-hours). The count is dominated
    by the 12 transformer blocks: the byte + AST-position ``EditTokenizer`` has
    only ~1.3K tokens, so the embedding/output layers add ~2M, not the ~197M a
    128K-vocab model of these dims would. Pair with ``--data-loader streaming``
    + prompt conditioning to push scaling into data variance rather than
    parameters. batch_size divides the 4 chips.
    """
    return ModelPreset(
        name="tpu_vet",
        config=EditModelConfig(
            vocab_size=DEFAULT_VOCAB_SIZE,
            hidden_dim=768,
            intermediate_dim=3072,
            num_layers=12,
            num_heads=12,
            num_kv_heads=12,
            max_seq_len=1024,
            # bf16 matmuls on the MXU: ~2-4x throughput and ~half the activation
            # memory vs float32, with sensitive reductions (rmsnorm, logits) kept
            # in float32. This is the biggest MFU lever (issue #132); float32 at
            # these dims measured ~3.5% MFU. Override with --compute-dtype.
            compute_dtype="bfloat16",
        ),
        resource=ResourceConfig.with_tpu("v6e-4"),
        # 64 global (16/chip). This fit the v6e-4's ~31GB HBM/chip even in
        # float32 at seq 1024; bf16 halves activation memory, so there is now
        # headroom to raise this (issue #132 lever 3) -- do so with --batch-size
        # after confirming HBM use, rather than baking a larger default in blind.
        batch_size=64,
        learning_rate=3e-4,
        description="~115M model on v6e-4 for cheap data-scaling experiments",
    )


def tpu_vet_300m_preset() -> ModelPreset:
    """~300M model on the same v6e-4 slice as ``tpu_vet``, for the exp10 capacity test.

    The bf16 + fused-attention landing (issue #132) roughly halved activation
    memory and 2-4x'd throughput, so ~2.6x the ``tpu_vet`` parameters (115M ->
    ~305M) now fits the v6e-4 and trains at ~similar wall-clock. Same width-to-
    depth balance as ``tpu_vet`` scaled up: hidden 768->1024, layers 12->18, with
    head_dim held at 64 (16 heads). MHA (num_kv_heads == num_heads) to keep the
    capacity comparison against the 115M ``tpu_vet`` clean. batch_size and LR are
    held at the ``tpu_vet`` values on purpose so exp10 varies *only* capacity
    (and the single-edit task) -- see scripts/train_exp10_300m.sh.
    """
    return ModelPreset(
        name="tpu_vet_300m",
        config=EditModelConfig(
            vocab_size=DEFAULT_VOCAB_SIZE,
            hidden_dim=1024,
            intermediate_dim=4096,
            num_layers=18,
            num_heads=16,
            num_kv_heads=16,
            max_seq_len=1024,
            compute_dtype="bfloat16",
            # ~305M at batch 64 / seq 1024 OOMs the v6e-4's HBM even in bf16
            # (observed: RESOURCE_EXHAUSTED after step 0). Gradient checkpointing
            # recomputes block activations in the backward pass instead of holding
            # all 18 layers' activations at once -- the dominant memory term --
            # for ~33% more compute (well within the bf16 throughput headroom). It
            # is numerically identical (same gradients), so batch 64 and the
            # capacity comparison vs the 115M tpu_vet stay clean.
            gradient_checkpointing=True,
        ),
        resource=ResourceConfig.with_tpu("v6e-4"),
        batch_size=64,
        learning_rate=3e-4,
        description="~305M model on v6e-4 (bf16 + grad checkpointing) for the exp10 capacity experiment",
    )


def tpu_v4_8_preset() -> ModelPreset:
    """Large preset for v4-8 TPU (~1.6B params)."""
    return ModelPreset(
        name="tpu_v4_8",
        config=EditModelConfig(
            vocab_size=DEFAULT_VOCAB_SIZE,
            hidden_dim=2048,
            intermediate_dim=8192,
            num_layers=24,
            num_heads=16,
            num_kv_heads=16,
            max_seq_len=4096,
        ),
        resource=ResourceConfig.with_tpu("v4-8"),
        batch_size=64,
        learning_rate=1e-4,
        description="For v4-8 TPU pod training",
    )


def tpu_v5p_8_preset() -> ModelPreset:
    """Largest preset for v5p-8 TPU pod (~7B params, Marin-8b-class architecture)."""
    return ModelPreset(
        name="tpu_v5p_8",
        config=EditModelConfig(
            vocab_size=DEFAULT_VOCAB_SIZE,
            hidden_dim=4096,
            intermediate_dim=14336,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            max_seq_len=8192,
        ),
        resource=ResourceConfig.with_tpu("v5p-8"),
        batch_size=128,
        learning_rate=5e-5,
        description="For v5p-8 TPU pod, 8B scale",
    )


PRESETS = {
    "toy": toy_preset,
    "overnight_cpu": overnight_cpu_preset,
    "laptop": laptop_preset,
    "single_gpu": single_gpu_preset,
    "tpu_smoke": tpu_smoke_preset,
    "tpu_vet": tpu_vet_preset,
    "tpu_vet_300m": tpu_vet_300m_preset,
    "tpu_v4_8": tpu_v4_8_preset,
    "tpu_v5p_8": tpu_v5p_8_preset,
}


def get_preset(name: str) -> ModelPreset:
    """Get a preset by name.

    Args:
        name: Preset name.

    Returns:
        ModelPreset instance.

    Raises:
        ValueError: If preset name is unknown.
    """
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name]()
