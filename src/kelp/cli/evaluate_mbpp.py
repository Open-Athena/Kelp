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

"""MBPP-based evaluation for Kelp tree diffusion edit models.

Uses held-out MBPP programs (with real test cases) to evaluate program repair.
This provides a more representative and contamination-free eval compared to the
hand-crafted EVAL_TASKS in evaluate.py.

Pipeline:
1. Load MBPP programs and their assert-based test cases
2. Build subtree bank from training corpus (not eval programs)
3. For each program: corrupt via AST mutation, repair with best-of-N, test
4. Report per-program and aggregate metrics

Usage:
    uv run python -m kelp.cli.evaluate_mbpp \\
        --checkpoint-dir checkpoints/kelp-edit-v3 \\
        --corpus-file corpus.txt
"""

import argparse
import ast
import json
import logging
import random
import signal
import sys
import time
from contextlib import contextmanager

import jax
from etils import epath

from kelp.cli._eval_resume import eval_fingerprint, load_completed, shard_dir, write_result
from kelp.cli._logging import configure_logging
from kelp.corpus import is_valid_python, load_corpus
from kelp.inference.beam_search import best_of_n
from kelp.model.checkpointing import find_best_checkpoint, load_checkpoint
from kelp.model.config import EditModelConfig
from kelp.model.edit_model import EditModelParams
from kelp.tree.corruption import corrupt_realistic
from kelp.tree.subtree_bank import SubtreeBank
from kelp.tree.tokenizer import EditTokenizer

logger = logging.getLogger(__name__)


def load_mbpp_eval_tasks(max_length: int = 512, max_tasks: int = 0) -> list[dict]:
    """Load MBPP programs as eval tasks with executable test cases.

    Returns list of dicts with keys: task_id, text, clean, tests, setup_code.
    """
    from datasets import load_dataset

    tasks: list[dict] = []
    for split in ["train", "validation", "test", "prompt"]:
        try:
            ds = load_dataset("google-research-datasets/mbpp", "full", split=split, trust_remote_code=True)
        except Exception:
            continue

        for item in ds:
            code = item.get("code", "")
            test_list = item.get("test_list", [])
            setup_code = item.get("test_setup_code", "")

            if not code or not test_list:
                continue
            if len(code) > max_length or len(code) < 20:
                continue
            try:
                ast.parse(code)
            except SyntaxError:
                continue

            tasks.append(
                {
                    "task_id": item.get("task_id", len(tasks)),
                    "text": item.get("text", ""),
                    "clean": code,
                    "tests": test_list,
                    "setup_code": setup_code,
                }
            )

    if max_tasks > 0:
        tasks = tasks[:max_tasks]

    logger.info(f"Loaded {len(tasks)} MBPP eval tasks")
    return tasks


class _TestTimeout(Exception):
    """Raised when a generated candidate exceeds the per-test wall-clock limit."""


@contextmanager
def _time_limit(seconds: float):
    """Best-effort wall-clock limit for executing generated code.

    Guards against a non-terminating candidate (e.g. ``while True``) hanging the
    whole eval -- exactly the vet-cond-v2 failure where one task's runaway
    candidate stalled the run and 17/50 tasks were lost. Uses SIGALRM, which is
    main-thread + Unix only; off the main thread (or if unavailable) it degrades
    to no limit rather than erroring. A C-level busy loop can still ignore the
    signal, but model-generated MBPP code is pure Python and interruptible.
    """
    if seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise _TestTimeout()

    try:
        old = signal.signal(signal.SIGALRM, _handler)
    except ValueError:
        # Not the main thread -> cannot arm SIGALRM; run without a limit.
        yield
        return

    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def run_mbpp_test(program: str, test_assert: str, setup_code: str = "", timeout_s: float = 5.0) -> bool:
    """Execute an MBPP assert-based test case against a program.

    ``timeout_s`` bounds execution so a non-terminating candidate fails the test
    instead of hanging the eval (0 disables the limit). See :func:`_time_limit`.
    """
    try:
        namespace: dict = {}
        with _time_limit(timeout_s):
            if setup_code:
                exec(setup_code, namespace)
            exec(program, namespace)
            exec(test_assert, namespace)
        return True
    except Exception:
        # Includes _TestTimeout (a non-terminating candidate) -> a failed test.
        return False


def evaluate_mbpp_task(
    task: dict,
    params: EditModelParams,
    config: EditModelConfig,
    tokenizer: EditTokenizer,
    bank: SubtreeBank,
    key: jax.Array,
    num_corruptions: int = 5,
    corruption_steps: int = 3,
    p_near_miss: float = 0.0,
    allow_bank_swap: bool = True,
    n_best_of: int = 16,
    max_depth: int = 10,
    constrain_position: bool = False,
    test_timeout: float = 5.0,
) -> dict:
    """Evaluate a single MBPP task across multiple corruption/repair trials.

    ``p_near_miss`` selects the corruption distribution via the shared
    :func:`kelp.tree.corruption.corrupt_realistic` policy: set it to the training
    run's value to measure the *trained* task (matched eval), or 0.0 for the
    original out-of-context bank-swap (unmatched / generalization eval).
    """
    clean = task["clean"]
    tests = task["tests"]
    setup_code = task.get("setup_code", "")
    prompt = task.get("text") if tokenizer.prompt_tokens else None
    rng = random.Random(task["task_id"])

    total_valid = 0
    total_exact_match = 0
    total_test_pass_rate = 0.0
    total_candidates = 0
    total_trials = 0
    best_overall_pass_rate = 0.0
    best_overall_candidate = clean

    for _trial in range(num_corruptions):
        key, _corrupt_key, search_key = jax.random.split(key, 3)

        corrupted, _mode = corrupt_realistic(
            clean,
            num_steps=corruption_steps,
            bank=bank,
            rng=rng,
            p_near_miss=p_near_miss,
            allow_bank_swap=allow_bank_swap,
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
            constrain_position=constrain_position,
        )

        for c in candidates:
            total_candidates += 1
            if is_valid_python(c.source):
                total_valid += 1
            if c.source.strip() == clean.strip():
                total_exact_match += 1

        # Test all candidates and pick the one that passes the most tests.
        # This is execution-guided reranking: the model generates diverse
        # candidates and we select by functional correctness.
        if candidates:
            best_trial_pass_rate = 0.0
            best_trial_candidate = candidates[0].source
            for c in candidates:
                c_passed = sum(1 for t in tests if run_mbpp_test(c.source, t, setup_code, timeout_s=test_timeout))
                c_rate = c_passed / len(tests) if tests else 0.0
                if c_rate > best_trial_pass_rate:
                    best_trial_pass_rate = c_rate
                    best_trial_candidate = c.source
            total_test_pass_rate += best_trial_pass_rate

            if best_trial_pass_rate > best_overall_pass_rate:
                best_overall_pass_rate = best_trial_pass_rate
                best_overall_candidate = best_trial_candidate

    return {
        "task_id": task["task_id"],
        "text": task["text"][:100],
        "num_trials": total_trials,
        "total_candidates": total_candidates,
        "valid_rate": total_valid / max(total_candidates, 1),
        "exact_match_rate": total_exact_match / max(total_candidates, 1),
        "avg_test_pass_rate": total_test_pass_rate / max(total_trials, 1),
        "best_test_pass_rate": best_overall_pass_rate,
        "best_candidate": best_overall_candidate.strip()[:200],
        "clean": clean.strip()[:200],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Kelp checkpoint on held-out MBPP programs")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/kelp-edit-v3",
        help="Directory containing step-XXXXXX subdirectories",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint subdirectory (uses latest if not set)",
    )
    parser.add_argument(
        "--corpus-file",
        type=str,
        default=None,
        help="Training corpus file for building subtree bank (recommended for realistic corruption)",
    )
    parser.add_argument("--num-corruptions", type=int, default=5, help="Corruption trials per task")
    parser.add_argument("--corruption-steps", type=int, default=3, help="AST mutations per corruption")
    parser.add_argument(
        "--p-near-miss",
        type=float,
        default=0.0,
        help="Fraction of trials corrupted with realistic in-context bugs (matches training's "
        "--p-near-miss). 0.0 = original bank-swap corruption (unmatched/generalization eval). "
        "IGNORED when --no-bank-swap-fallback is set (the cascade is then always attempted).",
    )
    parser.add_argument(
        "--no-bank-swap-fallback",
        dest="allow_bank_swap",
        action="store_false",
        help="Skip trials where no realistic corruption applies instead of falling back to a "
        "bank swap (matches training's --no-bank-swap-fallback). Always attempts the realistic "
        "cascade, so it overrides --p-near-miss (which is then ignored).",
    )
    parser.add_argument("--n-best-of", type=int, default=16, help="Number of independent rollouts")
    parser.add_argument("--max-depth", type=int, default=10, help="Maximum edit depth")
    parser.add_argument(
        "--constrain-position",
        action="store_true",
        default=False,
        help="Experimental: mask the edit-position token to valid AST boundaries at decode time. "
        "Off by default -- it raises edit VALIDITY (23%%->90%%) but barely moves functional repair "
        "(realistic best-of-16 26.7%% vs 27.3%% unconstrained; hard 8.0%% vs 5.3%%), because valid != "
        "correct: the model's position CHOICE, not validity, is the bottleneck. Research toggle.",
    )
    parser.add_argument("--max-tasks", type=int, default=50, help="Max MBPP tasks to evaluate (0=all)")
    parser.add_argument(
        "--test-timeout",
        type=float,
        default=5.0,
        help="Per-test wall-clock limit (seconds) for executing a generated candidate; a "
        "non-terminating candidate fails the test instead of hanging the eval (0 disables).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for results")
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project to log eval metrics to (empty/unset = no W&B logging).",
    )
    parser.add_argument("--wandb-entity", type=str, default="open-athena", help="W&B entity (team/user).")
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="Resume/append to this W&B run id (e.g. the training run), so eval points overlay the loss "
        "curve. All checkpoints logged to the same run form the repair-vs-step curve.",
    )
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name (default: eval-<checkpoint>).")
    return parser.parse_args()


def _checkpoint_step(ckpt_dir: epath.Path) -> int | None:
    """Parse the training step from a ``step-XXXXXX`` checkpoint dir name."""
    name = ckpt_dir.name
    suffix = name[len("step-") :] if name.startswith("step-") else ""
    return int(suffix) if suffix.isdigit() else None


def _log_eval_to_wandb(args: argparse.Namespace, ckpt_dir: epath.Path, metrics: dict) -> None:
    """Log aggregate eval metrics to W&B, keyed by checkpoint_step so repeated
    evals (per checkpoint) draw a repair-vs-step curve independent of log order."""
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed; skipping W&B eval logging")
        return

    step = _checkpoint_step(ckpt_dir)
    run = wandb.init(
        entity=args.wandb_entity or None,
        project=args.wandb_project,
        name=args.wandb_run_name or f"eval-{ckpt_dir.name}",
        id=args.wandb_run_id or None,
        resume="allow" if args.wandb_run_id else None,
    )
    # checkpoint_step is the x-axis for eval/* so out-of-order checkpoint evals
    # still land at the right place on the curve.
    wandb.define_metric("checkpoint_step")
    wandb.define_metric("eval/*", step_metric="checkpoint_step")
    payload = {f"eval/{k}": v for k, v in metrics.items()}
    if step is not None:
        payload["checkpoint_step"] = step
    run.log(payload)
    run.finish()
    logger.info(f"Logged eval metrics to W&B: {run.url}")


def _eval_fingerprint(args: argparse.Namespace, ckpt_dir: epath.Path) -> str:
    """Fingerprint of the config that determines a task's result, so re-runs
    with different parameters get a fresh shard dir instead of stale results.
    ``max_tasks`` is excluded: it changes which tasks run, not any task's result,
    so a wider run can reuse a narrower one's shards."""
    return eval_fingerprint(
        {
            "checkpoint": str(ckpt_dir),
            "seed": args.seed,
            "num_corruptions": args.num_corruptions,
            "corruption_steps": args.corruption_steps,
            "p_near_miss": args.p_near_miss,
            "allow_bank_swap": args.allow_bank_swap,
            "n_best_of": args.n_best_of,
            "max_depth": args.max_depth,
            "corpus_file": args.corpus_file,
            "constrain_position": args.constrain_position,
            "test_timeout": args.test_timeout,
        }
    )


def main():
    configure_logging()  # force stdout handler so INFO logs survive JAX/absl's root handler (visible in iris logs)
    args = parse_args()

    if not args.allow_bank_swap and args.p_near_miss < 1.0:
        logger.warning(
            "--p-near-miss=%.2f is ignored because --no-bank-swap-fallback is set: the realistic "
            "cascade is always attempted (equivalent to p_near_miss=1.0). Drop --no-bank-swap-fallback "
            "to honor the fraction.",
            args.p_near_miss,
        )

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

    # Load MBPP eval tasks.
    eval_tasks = load_mbpp_eval_tasks(max_tasks=args.max_tasks)
    if not eval_tasks:
        logger.error("No MBPP tasks loaded")
        return 1

    # Build subtree bank from training corpus if provided (issue #52),
    # otherwise fall back to building from eval programs.
    if args.corpus_file:
        corpus = load_corpus(args.corpus_file)
        logger.info(f"Building subtree bank from training corpus: {len(corpus)} programs")
        bank = SubtreeBank.from_corpus(corpus)
    else:
        eval_programs = [t["clean"] for t in eval_tasks]
        logger.info(f"Building subtree bank from {len(eval_programs)} eval programs (no --corpus-file)")
        bank = SubtreeBank.from_corpus(eval_programs)
    logger.info(f"Subtree bank: {bank.total_entries} entries across {len(bank.entries)} node types")

    key = jax.random.PRNGKey(args.seed)

    logger.info(f"Running MBPP evaluation: {len(eval_tasks)} tasks, {args.num_corruptions} corruptions each")
    logger.info(f"Inference: best-of-{args.n_best_of}, max_depth={args.max_depth}")
    logger.info("")

    # Durable per-task results so a preemption resumes instead of restarting.
    # The shard dir is fingerprinted by the eval config so a re-run with
    # different parameters doesn't reuse stale results.
    output_path = args.output or str(ckpt_dir / "mbpp_eval_results.json")
    tasks_dir = shard_dir(output_path, _eval_fingerprint(args, ckpt_dir))
    tasks_dir.mkdir(parents=True, exist_ok=True)
    completed = load_completed(tasks_dir, "task_id")
    if completed:
        logger.info(f"Resuming: {len(completed)} of {len(eval_tasks)} tasks already complete in {tasks_dir}")

    start_time = time.time()

    for i, task in enumerate(eval_tasks):
        tid = int(task["task_id"])
        if tid in completed:
            continue
        # Fold the task_id into the base key so each task's randomness is
        # independent of iteration order -- results are identical whether the
        # run is fresh or resumed after a preemption.
        task_key = jax.random.fold_in(key, tid)
        logger.info(f"[{i + 1}/{len(eval_tasks)}] task_id={tid}: {task['text'][:60]}")

        result = evaluate_mbpp_task(
            task=task,
            params=params,
            config=config,
            tokenizer=tokenizer,
            bank=bank,
            key=task_key,
            num_corruptions=args.num_corruptions,
            corruption_steps=args.corruption_steps,
            p_near_miss=args.p_near_miss,
            allow_bank_swap=args.allow_bank_swap,
            n_best_of=args.n_best_of,
            max_depth=args.max_depth,
            constrain_position=args.constrain_position,
            test_timeout=args.test_timeout,
        )
        write_result(tasks_dir, result, "task_id")
        completed[tid] = result

        logger.info(
            f"  valid={result['valid_rate']:.1%} exact={result['exact_match_rate']:.1%} "
            f"test_pass={result['avg_test_pass_rate']:.1%} best_pass={result['best_test_pass_rate']:.1%}"
        )

    elapsed = time.time() - start_time

    # Aggregate over all completed tasks (freshly computed + resumed), in task order.
    all_results = [completed[int(t["task_id"])] for t in eval_tasks if int(t["task_id"]) in completed]

    # Aggregate metrics.
    tasks_with_trials = [r for r in all_results if r["num_trials"] > 0]
    if not tasks_with_trials:
        logger.warning("No tasks had successful corruptions — cannot compute metrics")
        return 1

    avg_valid = sum(r["valid_rate"] for r in tasks_with_trials) / len(tasks_with_trials)
    avg_exact = sum(r["exact_match_rate"] for r in tasks_with_trials) / len(tasks_with_trials)
    avg_test_pass = sum(r["avg_test_pass_rate"] for r in tasks_with_trials) / len(tasks_with_trials)
    avg_best_pass = sum(r["best_test_pass_rate"] for r in tasks_with_trials) / len(tasks_with_trials)

    logger.info("")
    logger.info("=" * 70)
    logger.info("MBPP Evaluation Summary")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {ckpt_dir}")
    logger.info(f"Tasks evaluated: {len(tasks_with_trials)} (of {len(eval_tasks)} loaded)")
    logger.info(f"Corruptions/task: {args.num_corruptions}")
    logger.info(f"Inference: best-of-{args.n_best_of}, max_depth={args.max_depth}")
    logger.info(f"Subtree bank: {'training corpus' if args.corpus_file else 'eval programs only'}")
    logger.info(f"Time: {elapsed:.1f}s")
    logger.info("")
    logger.info(f"{'Metric':<30} {'Value':>10}")
    logger.info("-" * 42)
    logger.info(f"{'Syntactic validity rate':<30} {avg_valid:>10.1%}")
    logger.info(f"{'Exact match rate':<30} {avg_exact:>10.1%}")
    logger.info(f"{'Avg test pass rate':<30} {avg_test_pass:>10.1%}")
    logger.info(f"{'Best test pass rate':<30} {avg_best_pass:>10.1%}")
    logger.info("=" * 70)

    if args.wandb_project:
        _log_eval_to_wandb(
            args,
            ckpt_dir,
            {
                "mbpp_avg_pass_rate": avg_test_pass,
                "mbpp_best_pass_rate": avg_best_pass,
                "syntactic_validity": avg_valid,
                "exact_match": avg_exact,
                "tasks_evaluated": len(tasks_with_trials),
            },
        )

    # Save the aggregated summary (output_path computed above for tasks_dir).
    results_data = {
        "checkpoint": str(ckpt_dir),
        "config": {
            "num_corruptions": args.num_corruptions,
            "corruption_steps": args.corruption_steps,
            "p_near_miss": args.p_near_miss,
            "allow_bank_swap": args.allow_bank_swap,
            "n_best_of": args.n_best_of,
            "max_depth": args.max_depth,
            "max_tasks": args.max_tasks,
            "corpus_file": args.corpus_file,
            "test_timeout": args.test_timeout,
        },
        "aggregate": {
            "tasks_evaluated": len(tasks_with_trials),
            "tasks_loaded": len(eval_tasks),
            "syntactic_validity": avg_valid,
            "exact_match": avg_exact,
            "avg_test_pass_rate": avg_test_pass,
            "best_test_pass_rate": avg_best_pass,
        },
        "per_task": all_results,
        "elapsed_seconds": elapsed,
    }

    epath.Path(output_path).write_text(json.dumps(results_data, indent=2))
    logger.info(f"Results saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
