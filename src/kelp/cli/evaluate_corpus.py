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

"""Held-out corpus evaluation for Kelp tree diffusion edit models.

Evaluates repair quality on programs sampled from the training corpus itself.
This measures whether the model can repair programs from its own training
distribution — the most direct test of what it learned.

Since corpus programs don't have test cases, we measure:
- Syntactic validity: does the repair parse as Python?
- Exact match: does the repair exactly match the original clean program?
- Normalized match: does the repair match after whitespace normalization?

The subtree bank is built from the sampled eval programs (not the full corpus),
keeping corruption difficulty proportional to the eval set.

Usage:
    uv run python -m kelp.cli.evaluate_corpus \\
        --checkpoint-dir checkpoints/kelp-edit-v5 \\
        --corpus-file corpus.txt

    uv run python -m kelp.cli.evaluate_corpus \\
        --checkpoint-dir checkpoints/kelp-edit-v5 --checkpoint step-010000 \\
        --corpus-file corpus.txt --num-tasks 100
"""

import argparse
import json
import logging
import random
import sys
import textwrap
import time

import jax
from etils import epath

from kelp.cli._eval_resume import eval_fingerprint, load_completed, shard_dir, write_result
from kelp.cli._logging import configure_logging
from kelp.corpus import extract_docstring, is_valid_python, load_corpus
from kelp.inference.beam_search import best_of_n
from kelp.model.checkpointing import find_best_checkpoint, load_checkpoint
from kelp.model.config import EditModelConfig
from kelp.model.edit_model import EditModelParams
from kelp.tree.mutation import corrupt_program
from kelp.tree.subtree_bank import SubtreeBank
from kelp.tree.tokenizer import EditTokenizer

logger = logging.getLogger(__name__)


def normalize_whitespace(source: str) -> str:
    """Normalize whitespace for fuzzy matching.

    Strips trailing whitespace per line, dedents, and strips leading/trailing
    blank lines. This catches cases where the repair is semantically correct
    but has different indentation from corruption/repair cycles.
    """
    return textwrap.dedent(source).strip()


def evaluate_corpus_program(
    clean: str,
    params: EditModelParams,
    config: EditModelConfig,
    tokenizer: EditTokenizer,
    bank: SubtreeBank,
    key: jax.Array,
    num_corruptions: int = 5,
    corruption_steps: int = 3,
    max_depth: int = 10,
    n_best_of: int = 16,
) -> dict:
    """Evaluate repair on a single corpus program."""
    rng = random.Random(42)

    total_valid = 0
    total_exact = 0
    total_normalized = 0
    total_candidates = 0
    total_trials = 0
    best_overall_score = -1.0
    best_overall_candidate = clean

    clean_normalized = normalize_whitespace(clean)

    # Extract docstring from the clean program for prompt conditioning.
    prompt = extract_docstring(clean) if tokenizer.prompt_tokens else None

    for _trial in range(num_corruptions):
        key, search_key = jax.random.split(key)

        corrupted, _mutations = corrupt_program(
            clean,
            num_steps=corruption_steps,
            bank=bank,
            rng=rng,
        )
        if corrupted == clean:
            continue

        total_trials += 1

        candidates = best_of_n(
            params=params,
            source=corrupted,
            cfg=config,
            tokenizer=tokenizer,
            key=search_key,
            n=n_best_of,
            max_depth=max_depth,
            temperature=0.8,
            prompt=prompt,
        )

        trial_best_score = -1.0
        trial_best_candidate = corrupted

        for c in candidates:
            total_candidates += 1
            valid = is_valid_python(c.source)
            if valid:
                total_valid += 1

            exact = c.source.strip() == clean.strip()
            if exact:
                total_exact += 1

            normalized = normalize_whitespace(c.source) == clean_normalized
            if normalized:
                total_normalized += 1

            # Score: prefer exact match, then normalized match, then model score.
            if exact:
                score = 2.0
            elif normalized:
                score = 1.0
            elif valid:
                score = c.score
            else:
                score = -1.0

            if score > trial_best_score:
                trial_best_score = score
                trial_best_candidate = c.source

        if trial_best_score > best_overall_score:
            best_overall_score = trial_best_score
            best_overall_candidate = trial_best_candidate

    return {
        "num_trials": total_trials,
        "total_candidates": total_candidates,
        "valid_rate": total_valid / max(total_candidates, 1),
        "exact_match_rate": total_exact / max(total_candidates, 1),
        "normalized_match_rate": total_normalized / max(total_candidates, 1),
        "best_candidate": best_overall_candidate.strip(),
        "clean": clean.strip(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Kelp on held-out corpus programs")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/kelp-edit",
        help="Directory containing step-XXXXXX subdirectories",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint subdirectory (e.g., step-012000)",
    )
    parser.add_argument(
        "--corpus-file",
        type=str,
        required=True,
        help="Training corpus file to sample eval programs from",
    )
    parser.add_argument("--num-tasks", type=int, default=50, help="Number of programs to evaluate")
    parser.add_argument("--num-corruptions", type=int, default=5, help="Corruption trials per program")
    parser.add_argument("--corruption-steps", type=int, default=3, help="AST mutations per corruption")
    parser.add_argument("--n-best-of", type=int, default=16, help="Number of independent rollouts")
    parser.add_argument("--max-depth", type=int, default=10, help="Maximum edit depth")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    return parser.parse_args()


def main():
    configure_logging()  # force stdout handler so INFO logs survive JAX/absl's root handler (visible in iris logs)
    args = parse_args()
    checkpoint_base = epath.Path(args.checkpoint_dir)

    if args.checkpoint:
        ckpt_dir = checkpoint_base / args.checkpoint
    else:
        ckpt_dir = find_best_checkpoint(checkpoint_base)
        if ckpt_dir is None:
            logger.error(f"No checkpoints found in {checkpoint_base}")
            return 1

    logger.info(f"Evaluating checkpoint: {ckpt_dir}")

    params, config = load_checkpoint(ckpt_dir)
    tokenizer = EditTokenizer(max_seq_len=config.max_seq_len, prompt_tokens=config.prompt_tokens)

    # Load and sample from corpus.
    corpus = load_corpus(args.corpus_file)
    logger.info(f"Loaded corpus: {len(corpus)} programs")

    # Filter to programs that fit in max_seq_len (the model can't handle longer).
    eligible = [p for p in corpus if len(p) <= config.max_seq_len - 50 and is_valid_python(p)]
    logger.info(f"Eligible programs (fit in seq_len, valid Python): {len(eligible)}")

    rng = random.Random(args.seed)
    eval_programs = rng.sample(eligible, min(args.num_tasks, len(eligible)))
    logger.info(f"Sampled {len(eval_programs)} programs for evaluation")

    # Build bank from the eval programs (controlled difficulty).
    bank = SubtreeBank.from_corpus(eval_programs)
    logger.info(f"Subtree bank: {bank.total_entries} entries across {len(bank.entries)} node types")

    key = jax.random.PRNGKey(args.seed)

    logger.info(f"Running corpus evaluation: {len(eval_programs)} programs, {args.num_corruptions} corruptions each")
    logger.info(f"Inference: best-of-{args.n_best_of}, max_depth={args.max_depth}")
    logger.info("")

    # Durable per-program results (fingerprinted by config) so a preemption
    # resumes instead of restarting from program 0.
    output_path = args.output or str(ckpt_dir / "corpus_eval_results.json")
    shards = shard_dir(
        output_path,
        eval_fingerprint(
            {
                "checkpoint": str(ckpt_dir),
                "seed": args.seed,
                "num_tasks": args.num_tasks,  # changes which programs are sampled per index
                "num_corruptions": args.num_corruptions,
                "corruption_steps": args.corruption_steps,
                "n_best_of": args.n_best_of,
                "max_depth": args.max_depth,
                "corpus_file": args.corpus_file,
            }
        ),
    )
    shards.mkdir(parents=True, exist_ok=True)
    completed = load_completed(shards, "index")
    if completed:
        logger.info(f"Resuming: {len(completed)} of {len(eval_programs)} programs already complete in {shards}")

    start_time = time.time()

    for i, prog in enumerate(eval_programs):
        if i in completed:
            continue
        # Fold the sample index into the base key so each program's randomness
        # is order-independent -- identical whether fresh or resumed.
        prog_key = jax.random.fold_in(key, i)
        # Use first line as the program name (usually 'def func_name(...):').
        first_line = prog.strip().split("\n")[0][:60]
        logger.info(f"[{i + 1}/{len(eval_programs)}] {first_line}")

        result = evaluate_corpus_program(
            clean=prog,
            params=params,
            config=config,
            tokenizer=tokenizer,
            bank=bank,
            key=prog_key,
            num_corruptions=args.num_corruptions,
            corruption_steps=args.corruption_steps,
            max_depth=args.max_depth,
            n_best_of=args.n_best_of,
        )
        result["index"] = i
        write_result(shards, result, "index")
        completed[i] = result

        logger.info(
            f"  valid={result['valid_rate']:.1%} exact={result['exact_match_rate']:.1%} "
            f"norm={result['normalized_match_rate']:.1%} "
            f"trials={result['num_trials']} cands={result['total_candidates']}"
        )

    elapsed = time.time() - start_time

    # Aggregate over all completed programs (freshly computed + resumed), in sample order.
    all_results = [completed[i] for i in range(len(eval_programs)) if i in completed]

    # Only include programs that had at least one corruption trial.
    tasks_with_trials = [r for r in all_results if r["num_trials"] > 0]

    if not tasks_with_trials:
        logger.warning("No programs could be corrupted — bank may be too small")
        return 1

    avg_valid = sum(r["valid_rate"] for r in tasks_with_trials) / len(tasks_with_trials)
    avg_exact = sum(r["exact_match_rate"] for r in tasks_with_trials) / len(tasks_with_trials)
    avg_normalized = sum(r["normalized_match_rate"] for r in tasks_with_trials) / len(tasks_with_trials)
    avg_candidates = sum(r["total_candidates"] for r in tasks_with_trials) / len(tasks_with_trials)

    logger.info("")
    logger.info("=" * 70)
    logger.info("Corpus Evaluation Summary")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {ckpt_dir}")
    logger.info(f"Programs evaluated: {len(tasks_with_trials)} (of {len(eval_programs)} sampled)")
    logger.info(f"Corruptions/program: {args.num_corruptions}")
    logger.info(f"Inference: best-of-{args.n_best_of}, max_depth={args.max_depth}")
    logger.info(f"Subtree bank: {bank.total_entries} entries / {len(bank.entries)} node types (from eval programs)")
    logger.info(f"Time: {elapsed:.1f}s")
    logger.info("")
    logger.info(f"{'Metric':<35} {'Value':>10}")
    logger.info("-" * 47)
    logger.info(f"{'Syntactic validity rate':<35} {avg_valid:>10.1%}")
    logger.info(f"{'Exact match rate':<35} {avg_exact:>10.1%}")
    logger.info(f"{'Normalized match rate':<35} {avg_normalized:>10.1%}")
    logger.info(f"{'Avg candidates per program':<35} {avg_candidates:>10.1f}")
    logger.info("=" * 70)

    # Save the aggregated summary (output_path computed above for the shard dir).
    results_data = {
        "checkpoint": str(ckpt_dir),
        "config": {
            "num_tasks": args.num_tasks,
            "num_corruptions": args.num_corruptions,
            "corruption_steps": args.corruption_steps,
            "n_best_of": args.n_best_of,
            "max_depth": args.max_depth,
            "seed": args.seed,
            "corpus_file": args.corpus_file,
        },
        "aggregate": {
            "programs_evaluated": len(tasks_with_trials),
            "programs_sampled": len(eval_programs),
            "syntactic_validity": avg_valid,
            "exact_match": avg_exact,
            "normalized_match": avg_normalized,
            "avg_candidates": avg_candidates,
        },
        "per_program": tasks_with_trials,
        "elapsed_seconds": elapsed,
    }

    epath.Path(output_path).write_text(json.dumps(results_data, indent=2))
    logger.info(f"Results saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
