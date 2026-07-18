# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render training examples for a human gut-check before a training run.

Shows, for a sample of corpus programs, exactly what the model will be trained
on under a given corruption config: the clean program, the corrupted input (what
the model sees), the docstring prompt, and the target edit (position + what to
insert). Use it to sanity-check corruption realism and catch error classes
before spending compute on a retrain.

Usage:
    uv run python -m kelp.cli.inspect_data \\
        --corpus-file gs://marin-us-east5/kelp/corpus/stack_edu_python_vet.txt \\
        --num 20 --p-near-miss 1.0 --max-corruption-steps 1
"""

import argparse
import logging
import random
import sys

from kelp.corpus import extract_docstring, load_corpus
from kelp.tree.corruption import corrupt_realistic
from kelp.tree.subtree_bank import SubtreeBank
from kelp.tree.tree_diff import find_path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render training examples for a gut-check")
    p.add_argument("--corpus-file", type=str, required=True, help="Corpus file (local or gs://)")
    p.add_argument("--num", type=int, default=20, help="Number of examples to render")
    p.add_argument("--p-near-miss", type=float, default=0.0, help="Fraction corrupted via e-graph near-miss")
    p.add_argument("--max-corruption-steps", type=int, default=1, help="Max mutations per corruption")
    p.add_argument("--max-edit-stmts", type=int, default=3, help="Max statements per bank-swap edit")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    return p.parse_args()


def _corrupt(clean: str, bank: SubtreeBank, rng: random.Random, args: argparse.Namespace) -> tuple[str, str]:
    """Return (corrupted, mode) via the same shared policy train/eval use."""
    steps = rng.randint(1, args.max_corruption_steps)
    return corrupt_realistic(
        clean,
        num_steps=steps,
        bank=bank,
        rng=rng,
        p_near_miss=args.p_near_miss,
        max_edit_stmts=args.max_edit_stmts,
    )


def main() -> int:
    # The rendered output goes through print(); silence the INFO flood from the
    # subtree bank build and (loudly) from egglog's equality-saturation engine.
    logging.disable(logging.INFO)
    args = parse_args()
    corpus = load_corpus(args.corpus_file)
    bank = SubtreeBank.from_corpus(corpus)
    rng = random.Random(args.seed)
    sample = rng.sample(corpus, min(args.num, len(corpus)))

    counts = {"op-flip": 0, "var-swap": 0, "near-miss": 0, "bank-swap": 0, "no-path": 0}
    for i, clean in enumerate(sample):
        corrupted, mode = _corrupt(clean, bank, rng, args)
        # The training target: the edit path from corrupted back to clean.
        path = find_path(corrupted, clean, max_edit_stmts=args.max_edit_stmts)
        counts[mode] += 1
        if not path:
            counts["no-path"] += 1
        prompt = extract_docstring(clean)
        edit = path[0] if path else None
        print("=" * 78)
        print(f"[{i + 1}/{len(sample)}] corruption={mode}  path_len={len(path)}  prompt={'yes' if prompt else 'no'}")
        if prompt:
            print(f"  PROMPT: {prompt[:100]}")
        print("  --- CLEAN (target) ---")
        print("    " + clean.rstrip().replace("\n", "\n    "))
        print("  --- CORRUPTED (model input) ---")
        print("    " + corrupted.rstrip().replace("\n", "\n    "))
        if edit is not None:
            print(
                f"  --- FIRST EDIT (model must produce): "
                f"@pos {edit.start}..{edit.end}  ->  {edit.replacement!r}"
            )
    print("=" * 78)
    print(f"summary: {counts}  (no-path = corruption find_path failed -> example would be dropped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
