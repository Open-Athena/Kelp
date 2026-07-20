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

"""Training pipeline for tree diffusion with TreeDiff supervision.

Generates training data by:
1. Corrupting clean programs via AST subtree replacement (forward process)
2. Computing TreeDiff edit paths from corrupted back to clean
3. Picking a random step along the path as the training target
4. Training the AR model to predict that edit (position + replacement tokens)
"""

import logging
import math
import os
import random as pyrandom
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from etils import epath
from jax.tree_util import register_dataclass
from jaxtyping import Array

from kelp.model.checkpointing import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from kelp.model.config import EditModelConfig
from kelp.model.edit_model import (
    EditModelParams,
    ar_loss,
    init_edit_params,
)
from kelp.model.layers import (
    AttentionParams,
    TransformerBlockParams,
)
from kelp.training.generation import GenerationConfig, generate_example
from kelp.training.sharding import (
    data_parallel_size,
    make_data_parallel_mesh,
    replicate,
    shard_batch,
)
from kelp.training.throughput import (
    ThroughputTracker,
    count_params,
    device_peak_flops,
)
from kelp.tree.subtree_bank import SubtreeBank
from kelp.tree.tokenizer import EditTokenizer

logger = logging.getLogger(__name__)


@register_dataclass
@dataclass(frozen=True)
class EditTrainingState:
    """Training state for the AR edit model."""

    step: int
    params: EditModelParams
    opt_state: optax.OptState
    key: jax.Array


@dataclass(frozen=True)
class EditTrainingConfig:
    """Configuration for tree diffusion training."""

    model: EditModelConfig
    """Model configuration."""

    max_seq_len: int = 512
    """Maximum sequence length for tokenized training examples."""

    learning_rate: float = 3e-4
    """Base learning rate."""

    weight_decay: float = 0.01
    """Weight decay for AdamW."""

    warmup_steps: int = 100
    """Number of warmup steps."""

    total_steps: int = 10000
    """Total training steps."""

    batch_size: int = 8
    """Global batch size."""

    log_interval: int = 10
    """Steps between logging."""

    checkpoint_interval: int = 1000
    """Steps between checkpoints."""

    seed: int = 42
    """Random seed."""

    output_dir: str = "checkpoints/kelp-edit"
    """Output directory for checkpoints."""

    max_corruption_steps: int = 5
    """Maximum number of AST mutations for corruption (paper's s_max)."""

    max_edit_stmts: int = 3
    """Maximum statement count per edit."""

    p_random: float = 0.2
    """Probability of using a random program instead of forward diffusion
    (paper's rho). Provides exposure to diverse starting points."""

    corruption_curriculum: str = "constant"
    """Schedule for ramping corruption difficulty during training.
    'constant': use max_corruption_steps throughout (original behavior).
    'linear': linearly ramp from 1 to max_corruption_steps over warmup phase.
    'cosine': cosine ramp from 1 to max_corruption_steps over warmup phase."""

    curriculum_warmup_fraction: float = 0.3
    """Fraction of total_steps over which the curriculum ramps from 1 to
    max_corruption_steps. Only used when corruption_curriculum != 'constant'."""

    p_prompt: float = 0.5
    """Probability of including a docstring prompt when one is available.
    Only effective when the model config has prompt_tokens=True."""

    p_near_miss: float = 0.0
    """Probability of corrupting with an e-graph near-miss (realistic in-context
    operator-flip bug) instead of a bank subtree swap, on the forward-diffusion
    branch. Falls back to bank swap when no near-miss is available. 0.0 = original
    bank-swap corruption."""

    allow_bank_swap: bool = True
    """When False, drop programs with no realistic corruption instead of injecting
    an out-of-context bank subtree swap (realistic-or-drop training, zero alien
    grafts). Set via --no-bank-swap-fallback."""

    wandb_entity: str | None = "open-athena"
    """W&B entity (team/user). Defaults to open-athena."""

    wandb_project: str | None = "kelp"
    """W&B project name. If set, enables W&B logging."""

    wandb_run_name: str | None = None
    """W&B run name."""

    def effective_max_corruption_steps(self, step: int) -> int:
        """Compute the effective max corruption steps at a given training step.

        During the warmup phase, ramps from 1 to max_corruption_steps according
        to the chosen schedule. After warmup, always returns max_corruption_steps.
        """
        if self.corruption_curriculum == "constant" or self.max_corruption_steps <= 1:
            return self.max_corruption_steps

        warmup_end = int(self.total_steps * self.curriculum_warmup_fraction)
        if step >= warmup_end or warmup_end == 0:
            return self.max_corruption_steps

        progress = step / warmup_end  # 0.0 to 1.0

        if self.corruption_curriculum == "linear":
            fraction = progress
        elif self.corruption_curriculum == "cosine":
            fraction = 0.5 * (1.0 - math.cos(math.pi * progress))
        else:
            return self.max_corruption_steps

        # Interpolate from 1 to max_corruption_steps.
        return max(1, round(1 + fraction * (self.max_corruption_steps - 1)))


def _edit_weight_decay_mask(params: EditModelParams) -> EditModelParams:
    """Weight decay mask for EditModelParams.

    Excludes embeddings and normalization weights from weight decay.
    """
    masked_blocks = tuple(
        TransformerBlockParams(
            attn=AttentionParams(
                w_q=True,
                w_k=True,
                w_v=True,
                w_o=True,
            ),
            rms_attn=False,
            rms_mlp=False,
            mlp_gate=True,
            mlp_up=True,
            mlp_down=True,
        )
        for _ in params.blocks
    )
    return EditModelParams(
        token_embed=False,
        output_proj=True,
        blocks=masked_blocks,
        final_norm=False,
    )


def create_edit_optimizer(config: EditTrainingConfig) -> optax.GradientTransformation:
    """Create optimizer with warmup, cosine decay, and weight decay masking."""
    warmup_steps = min(config.warmup_steps, max(config.total_steps // 2, 1))
    decay_steps = max(config.total_steps, warmup_steps + 1)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=config.learning_rate * 0.1,
    )

    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=schedule,
            weight_decay=config.weight_decay,
            mask=_edit_weight_decay_mask,
        ),
    )


def make_edit_train_step(
    config: EditModelConfig,
    optimizer: optax.GradientTransformation,
):
    """Create a JIT-compiled training step for the AR edit model."""

    def loss_fn(params, token_ids, loss_mask):
        # Training opts into the fused splash-attention path (fixed seq length,
        # under a mesh context -- see train_edit_model). Inference keeps the
        # default dense/reference path.
        return ar_loss(params, token_ids, loss_mask, config, fused_attention=True)

    def train_step(
        state: EditTrainingState,
        batch: dict[str, Array],
    ) -> tuple[EditTrainingState, dict]:
        key, _step_key = jax.random.split(state.key)

        (_loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params,
            batch["token_ids"],
            batch["loss_mask"],
        )

        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)

        grad_norm = optax.global_norm(grads)
        metrics["grad_norm"] = grad_norm

        new_state = EditTrainingState(
            step=state.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            key=key,
        )

        return new_state, metrics

    return jax.jit(train_step)


def generate_training_example(
    clean_source: str,
    corpus: list[str],
    bank: SubtreeBank,
    tokenizer: EditTokenizer,
    max_seq_len: int,
    config: EditTrainingConfig,
    rng: pyrandom.Random,
    step: int = 0,
) -> tuple[list[int], list[int]] | None:
    """Generate a single training example from a clean program (inline path).

    Thin wrapper over the pure :func:`kelp.training.generation.generate_example`:
    it resolves the curriculum-scheduled corruption difficulty for ``step`` and
    the per-example knobs from ``config``, then returns just
    ``(token_ids, loss_mask)`` for backward compatibility. The metadata the pure
    core also produces is dropped here (the inline loop does not need it).

    Args:
        step: Current training step, used for curriculum scheduling of
            corruption difficulty.

    Returns None if no valid training example could be generated.
    """
    example = generate_example(
        clean_source,
        corpus,
        bank,
        tokenizer,
        max_corruption_steps=config.effective_max_corruption_steps(step),
        gen_cfg=GenerationConfig.from_training_config(config),
        rng=rng,
        max_seq_len=max_seq_len,
    )
    if example is None:
        return None
    return example.token_ids, example.loss_mask


def create_edit_data_iter(
    corpus: list[str],
    bank: SubtreeBank,
    tokenizer: EditTokenizer,
    config: EditTrainingConfig,
    seed: int = 42,
    start_step: int = 0,
) -> Iterator[dict[str, Array]]:
    """Create a training data iterator for tree diffusion.

    Yields batches of (token_ids, loss_mask) arrays. Each batch is tagged
    with the current step so generate_training_example can use
    curriculum-based corruption difficulty. ``start_step`` aligns that
    curriculum when resuming a run (the data sequence itself is not replayed).
    """
    rng = pyrandom.Random(seed)
    step = start_step

    while True:
        batch_token_ids: list[list[int]] = []
        batch_loss_masks: list[list[int]] = []

        while len(batch_token_ids) < config.batch_size:
            clean_source = rng.choice(corpus)
            example = generate_training_example(
                clean_source=clean_source,
                corpus=corpus,
                bank=bank,
                tokenizer=tokenizer,
                max_seq_len=config.max_seq_len,
                config=config,
                rng=rng,
                step=step,
            )
            if example is None:
                continue

            token_ids, loss_mask = example
            batch_token_ids.append(token_ids)
            batch_loss_masks.append(loss_mask)

        # Pad to max_seq_len.
        padded_ids = jnp.zeros((config.batch_size, config.max_seq_len), dtype=jnp.int32)
        padded_masks = jnp.zeros((config.batch_size, config.max_seq_len), dtype=jnp.float32)

        for i in range(config.batch_size):
            seq_len = len(batch_token_ids[i])
            padded_ids = padded_ids.at[i, :seq_len].set(jnp.array(batch_token_ids[i]))
            padded_masks = padded_masks.at[i, :seq_len].set(jnp.array(batch_loss_masks[i]))

        yield {"token_ids": padded_ids, "loss_mask": padded_masks}
        step += 1


LogCallback = Callable[[int, dict], None]


def train_edit_model(
    config: EditTrainingConfig,
    data_iter: Iterator[dict[str, Array]],
    initial_params: EditModelParams | None = None,
    initial_state: "EditTrainingState | None" = None,
    log_callback: LogCallback | None = None,
    mesh: "jax.sharding.Mesh | None" = None,
) -> EditModelParams:
    """Train a tree diffusion edit model.

    Args:
        config: Training configuration.
        data_iter: Iterator yielding batches with 'token_ids' and 'loss_mask'.
        initial_params: Optional initial parameters (for transfer learning).
        initial_state: Optional full training state to resume from (params +
            optimizer + step + rng), e.g. from :func:`load_resume_state`.
            Training continues from ``initial_state.step``. Mutually exclusive
            with ``initial_params``.
        log_callback: Optional callback for logging metrics.
        mesh: Optional device mesh for data-parallel training. Defaults to a
            1-D mesh over all local devices. On a single device this is a
            no-op; on an N-chip host the batch is sharded across chips and the
            JITted step runs SPMD with automatic gradient all-reduce.

    Returns:
        Trained EditModelParams.
    """
    # Multi-host is not yet supported: the data iterators yield a per-process
    # batch, but the mesh spans all hosts, so a plain device_put mis-assembles
    # the global array (assembling per-process batches into a global batch is
    # unimplemented; tracked in issue #5). Fail loudly rather than train
    # silently-wrong across hosts. Single-host multi-chip slices (v6e-4, v5p-8)
    # run process_count()==1 and are fine.
    if jax.process_count() > 1:
        raise NotImplementedError(
            f"Multi-host training ({jax.process_count()} processes) is not yet supported: "
            "per-process batches are not assembled into a global batch (see issue #5). "
            "Use a single-host slice for now."
        )

    if mesh is None:
        mesh = make_data_parallel_mesh()
    dp_size = data_parallel_size(mesh)
    if config.batch_size % dp_size != 0:
        raise ValueError(
            f"batch_size ({config.batch_size}) must be divisible by the "
            f"data-parallel size ({dp_size} devices) for even sharding."
        )
    logger.info(f"Data-parallel mesh: {dp_size} device(s), batch/device={config.batch_size // dp_size}")

    key = jax.random.PRNGKey(config.seed)
    optimizer = create_edit_optimizer(config)

    # Initialize W&B if configured.
    wandb_run = None
    if config.wandb_project is not None:
        try:
            from dataclasses import asdict

            import wandb

            wandb_run = wandb.init(
                entity=config.wandb_entity,
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=asdict(config),
            )
            logger.info(f"W&B logging enabled: {wandb_run.url}")
        except ImportError:
            logger.warning("wandb not installed; skipping W&B logging")

    if initial_state is not None:
        state = initial_state
        start_step = int(state.step)
        logger.info(f"Resuming training from step {start_step}")
    else:
        if initial_params is None:
            key, init_key = jax.random.split(key)
            params = init_edit_params(config.model, key=init_key)
        else:
            params = initial_params
        opt_state = optimizer.init(params)
        state = EditTrainingState(step=0, params=params, opt_state=opt_state, key=key)
        start_step = 0

    # Replicate the training state across all devices; the JITted step then
    # runs data-parallel over batch-sharded inputs.
    state = replicate(state, mesh)

    train_step = make_edit_train_step(config.model, optimizer)

    # MFU / throughput accounting. Peak is the chip's bf16 FLOP/s; unknown
    # hardware (e.g. CPU/CI) yields None and we log throughput without MFU.
    peak = device_peak_flops(jax.devices()[0].device_kind)
    tracker = ThroughputTracker(
        param_count=count_params(state.params),
        tokens_per_step=config.batch_size * config.max_seq_len,
        num_devices=dp_size,
        peak_flops_per_device=peak,
    )
    logger.info(
        f"Throughput accounting: N={tracker.param_count:,} params, "
        f"{tracker.tokens_per_step:,} tokens/step, {dp_size} device(s), "
        f"peak={'%.0f TFLOP/s/chip' % (peak / 1e12) if peak else 'unknown (no MFU)'}"
    )
    # Baselined on the first log so compilation time is excluded from throughput.
    last_log_time: float | None = None
    last_log_step = start_step

    logger.info(f"Starting edit model training: steps {start_step}..{config.total_steps}")

    # Run the step calls under the mesh context so the fused TPU splash-attention
    # kernel can resolve the device mesh (it reads the abstract mesh via
    # haliax._get_mesh; without an active context it raises "requires a JAX mesh
    # context" and attention silently loses the fused path). No-op on the size-1
    # CPU/single-device mesh used off-cluster.
    with jax.set_mesh(mesh):
        for step in range(start_step, config.total_steps):
            batch = shard_batch(next(data_iter), mesh)
            state, metrics = train_step(state, batch)

            if step % config.log_interval == 0:
                # _log_edit_metrics reads float(loss), which blocks on this step's
                # async dispatch -- so the clock below reflects real completion.
                _log_edit_metrics(step, metrics)
                now = time.perf_counter()
                perf: dict[str, float] = {}
                if last_log_time is not None and step > last_log_step:
                    perf = tracker.metrics(now - last_log_time, step - last_log_step)
                    if perf:
                        _log_throughput(step, perf)
                last_log_time, last_log_step = now, step

                if wandb_run is not None:
                    wandb_run.log(
                        {**{k: float(v) for k, v in metrics.items()}, **perf},
                        step=step,
                    )
                if log_callback is not None:
                    log_callback(step, {**metrics, **perf})

            if config.output_dir and config.checkpoint_interval > 0 and (step + 1) % config.checkpoint_interval == 0:
                ckpt_dir = epath.Path(config.output_dir) / f"step-{step + 1:06d}"
                save_training_checkpoint(state.params, state.opt_state, state.step, state.key, config.model, ckpt_dir)

    # Save final checkpoint (resumable, like the interval checkpoints).
    if config.output_dir:
        ckpt_dir = epath.Path(config.output_dir) / f"step-{config.total_steps:06d}"
        save_training_checkpoint(state.params, state.opt_state, state.step, state.key, config.model, ckpt_dir)

    if wandb_run is not None:
        wandb_run.finish()

    logger.info("Training complete")
    return state.params


def load_resume_state(ckpt_dir: str | os.PathLike, config: EditTrainingConfig) -> "EditTrainingState":
    """Assemble an :class:`EditTrainingState` from a resumable checkpoint.

    Rebuilds the optimizer from ``config`` (needed to restore the optimizer
    state's structure) and returns the full state to hand to
    :func:`train_edit_model` as ``initial_state``.
    """
    optimizer = create_edit_optimizer(config)
    params, opt_state, step, key, _model_config = load_training_checkpoint(ckpt_dir, optimizer)
    return EditTrainingState(step=step, params=params, opt_state=opt_state, key=key)


def _log_edit_metrics(step: int, metrics: dict) -> None:
    """Log training metrics."""
    loss = float(metrics["loss"])
    acc = float(metrics["accuracy"])
    ppl = float(metrics["perplexity"])
    grad_norm = float(metrics["grad_norm"])
    num_tokens = float(metrics["num_loss_tokens"])

    logger.info(
        f"step={step:06d} loss={loss:.4f} acc={acc:.4f} "
        f"ppl={ppl:.2f} grad_norm={grad_norm:.4f} loss_tokens={num_tokens:.0f}"
    )


def _log_throughput(step: int, perf: dict) -> None:
    """Log throughput / MFU for the interval ending at ``step``."""
    parts = [
        f"steps/s={perf['perf/steps_per_s']:.2f}",
        f"tok/s={perf['perf/tokens_per_s']:,.0f}",
        f"TFLOP/s/chip={perf['perf/tflops_per_device']:.1f}",
    ]
    if "perf/mfu" in perf:
        parts.append(f"MFU={perf['perf/mfu'] * 100:.1f}%")
    logger.info(f"step={step:06d} " + " ".join(parts))
