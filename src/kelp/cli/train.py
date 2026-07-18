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

"""Main training script for Kelp tree diffusion edit models.

Trains an AR edit-prediction model on Python source code using the tree
diffusion pipeline: corrupt programs via AST subtree replacement, compute
TreeDiff edit paths, and train the model to predict single edits.

Usage:
    # Train on toy corpus (laptop)
    uv run python -m kelp.cli.train --preset toy --steps 1000

    # Train overnight on CPU
    uv run python -m kelp.cli.train --preset overnight_cpu --steps 30000

    # With W&B logging
    uv run python -m kelp.cli.train --preset laptop --wandb-project kelp
"""

import argparse
import logging
import os
import random
from dataclasses import replace

from kelp.cli._logging import configure_logging
from kelp.corpus import TOY_CORPUS, load_corpus
from kelp.model.checkpointing import find_best_checkpoint
from kelp.training.distributed import bootstrap_distributed
from kelp.training.engine import (
    EditTrainingConfig,
    create_edit_data_iter,
    load_resume_state,
    train_edit_model,
)
from kelp.training.presets import PRESETS, get_preset
from kelp.tree.augmentation import augment_bank
from kelp.tree.subtree_bank import SubtreeBank
from kelp.tree.tokenizer import EditTokenizer

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a Kelp tree diffusion edit model")
    parser.add_argument("--preset", type=str, default="toy", choices=list(PRESETS.keys()), help="Model preset")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (uses preset default if not set)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (uses preset default if not set)")
    parser.add_argument("--output-dir", type=str, default="checkpoints/kelp-edit", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-interval", type=int, default=10, help="Steps between logging")
    parser.add_argument("--wandb-entity", type=str, default="open-athena", help="W&B entity (team/user)")
    parser.add_argument("--wandb-project", type=str, default="kelp", help="W&B project name (enables W&B logging)")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument(
        "--corpus-file",
        type=str,
        default=None,
        help="Path to corpus file (programs separated by '# ---' sentinel lines; see corpus.CORPUS_SEPARATOR)",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=1000, help="Steps between checkpoints")
    parser.add_argument(
        "--augment", action="store_true", help="Augment subtree bank with renamed/perturbed/synthetic variants"
    )
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.set_defaults(augment=False)
    parser.add_argument(
        "--max-corruption-steps",
        type=int,
        default=5,
        help="Maximum AST mutations per corruption (default: 5)",
    )
    parser.add_argument(
        "--corruption-curriculum",
        type=str,
        default="constant",
        choices=["constant", "linear", "cosine"],
        help="Schedule for ramping corruption difficulty: constant (default), linear, or cosine",
    )
    parser.add_argument(
        "--curriculum-warmup-fraction",
        type=float,
        default=0.3,
        help="Fraction of training over which to ramp corruption difficulty (default: 0.3)",
    )
    parser.add_argument(
        "--prompt-conditioning",
        action="store_true",
        help="Enable prompt conditioning (adds PROMPT_START/PROMPT_END tokens, uses docstrings as prompts)",
    )
    parser.add_argument(
        "--p-prompt",
        type=float,
        default=0.5,
        help="Probability of including a docstring prompt when available (default: 0.5)",
    )
    parser.add_argument(
        "--p-near-miss",
        type=float,
        default=0.0,
        help="Probability of corrupting with an e-graph near-miss (realistic operator-flip bug) "
        "instead of a bank subtree swap; falls back to bank swap when unavailable (default: 0.0). "
        "IGNORED when --no-bank-swap-fallback is set (the cascade is then always attempted).",
    )
    parser.add_argument(
        "--no-bank-swap-fallback",
        dest="allow_bank_swap",
        action="store_false",
        help="Drop programs with no realistic corruption instead of injecting an out-of-context "
        "bank subtree swap (realistic-or-drop training). Always attempts the realistic cascade, "
        "so it overrides --p-near-miss (which is then ignored).",
    )
    parser.add_argument(
        "--data-loader",
        type=str,
        default="inline",
        choices=["inline", "streaming"],
        help="Data pipeline: 'inline' (single-process, default) or 'streaming' "
        "(concurrent seed-driven generation; keeps fast accelerators fed)",
    )
    parser.add_argument(
        "--gen-workers",
        type=int,
        default=None,
        help="Streaming: CPU generation processes (default: cpu_count-2). Ignored for inline.",
    )
    parser.add_argument(
        "--reuse-factor",
        type=int,
        default=1,
        help="Streaming: times each generated example is fed before eviction (default: 1)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1024,
        help="Streaming: shuffle-buffer capacity in examples (default: 1024)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from the latest checkpoint in --output-dir if one exists (default: on). "
        "This makes preemptible runs recover automatically.",
    )
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Always start from step 0.")
    return parser.parse_args()


def main():
    """Main entry point."""
    configure_logging()  # force a stdout handler that survives absl/JAX (Iris-observable)
    args = parse_args()

    if not args.allow_bank_swap and 0.0 < args.p_near_miss < 1.0:
        logger.warning(
            "--p-near-miss=%.2f is ignored because --no-bank-swap-fallback is set: the realistic "
            "cascade is always attempted. Drop --no-bank-swap-fallback to honor the fraction.",
            args.p_near_miss,
        )

    # Bring up JAX distributed from Iris job metadata before any device use.
    # No-op off-cluster (laptop/CI/single host).
    bootstrap_distributed()

    preset = get_preset(args.preset)
    model_config = preset.config
    if args.prompt_conditioning:
        model_config = replace(model_config, prompt_tokens=True)
    lr = args.lr or preset.learning_rate
    batch_size = args.batch_size or preset.batch_size

    # Load corpus.
    if args.corpus_file:
        corpus = load_corpus(args.corpus_file)
        logger.info(f"Loaded {len(corpus)} programs from {args.corpus_file}")
    else:
        corpus = TOY_CORPUS
        logger.info(f"Using toy corpus ({len(corpus)} programs)")

    # Build subtree bank and tokenizer.
    bank = SubtreeBank.from_corpus(corpus)
    if args.augment:
        rng = random.Random(args.seed)
        bank = augment_bank(bank, rng, n_renamed=2, n_perturbed=2, synthetic_count=50)
    tokenizer = EditTokenizer(max_seq_len=model_config.max_seq_len, prompt_tokens=model_config.prompt_tokens)
    logger.info(f"Subtree bank: {bank.total_entries} entries across {len(bank.entries)} node types")

    # Override model config vocab_size to match tokenizer.
    model_config = replace(model_config, vocab_size=tokenizer.vocab_size)

    train_cfg = EditTrainingConfig(
        model=model_config,
        max_seq_len=model_config.max_seq_len,
        learning_rate=lr,
        total_steps=args.steps,
        batch_size=batch_size,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        output_dir=args.output_dir,
        seed=args.seed,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project or None,  # empty string disables W&B
        wandb_run_name=args.wandb_run_name,
        max_corruption_steps=args.max_corruption_steps,
        corruption_curriculum=args.corruption_curriculum,
        curriculum_warmup_fraction=args.curriculum_warmup_fraction,
        p_prompt=args.p_prompt,
        p_near_miss=args.p_near_miss,
        allow_bank_swap=args.allow_bank_swap,
    )

    # Resume from the latest checkpoint in output_dir if one exists (e.g. after a
    # preemption) -- full state (params + optimizer + step + rng) is restored.
    initial_state = None
    start_step = 0
    if args.resume and args.output_dir:
        latest = find_best_checkpoint(args.output_dir)
        if latest is not None:
            logger.info(f"Resuming from checkpoint: {latest}")
            initial_state = load_resume_state(latest, train_cfg)
            start_step = int(initial_state.step)

    if args.data_loader == "streaming":
        from kelp.training.streaming import create_streaming_data_iter

        workers = args.gen_workers if args.gen_workers is not None else max(1, (os.cpu_count() or 2) - 2)
        logger.info(f"Streaming dataloader: {workers} gen workers, reuse={args.reuse_factor}")
        data_iter = create_streaming_data_iter(
            corpus=corpus,
            bank=bank,
            tokenizer=tokenizer,
            config=train_cfg,
            seed=args.seed,
            num_workers=workers,
            reuse_factor=args.reuse_factor,
            buffer_size=args.buffer_size,
            start_step=start_step,
        )
    else:
        data_iter = create_edit_data_iter(
            corpus=corpus,
            bank=bank,
            tokenizer=tokenizer,
            config=train_cfg,
            seed=args.seed,
            start_step=start_step,
        )

    logger.info(f"Training config: {train_cfg}")
    logger.info(f"Model config: {model_config}")

    train_edit_model(
        config=train_cfg,
        data_iter=data_iter,
        initial_state=initial_state,
    )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
