# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Precompute the (augmented) subtree bank as a corpus-prep artifact.

E-graph augmentation is minutes of CPU-only work with no checkpointing: on a
preemptible slice it re-runs from zero on every restart and can starve a
training job of its first checkpoint entirely (exp12 burned 29 attempts this
way without training a single step). Build the bank offline, next to the
corpus and spec sidecar; training loads it in seconds via ``--bank-file``.

Usage:
    uv run python -m kelp.cli.build_bank \\
        --corpus-file curated_v3.txt --output curated_v3_bank.json.gz --augment
"""

import argparse
import logging
import random
import sys
import time

from kelp.cli._logging import configure_logging
from kelp.corpus import load_corpus
from kelp.tree.augmentation import augment_bank
from kelp.tree.subtree_bank import SubtreeBank

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute the (augmented) subtree bank for a corpus")
    parser.add_argument("--corpus-file", type=str, required=True, help="Corpus file (local or gs://)")
    parser.add_argument("--output", type=str, required=True, help="Output bank .json.gz (local or gs://)")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply the training-time augmentation (renamed/perturbed/synthetic/e-graph) before saving. "
        "Match the training run's intent: exp-style runs use --augment.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Augmentation RNG seed (recorded in the artifact name)")
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()

    corpus = load_corpus(args.corpus_file)
    logger.info(f"Building bank from {len(corpus)} programs ({args.corpus_file})")
    start = time.time()
    bank = SubtreeBank.from_corpus(corpus)
    logger.info(f"Base bank: {bank.total_entries} entries ({time.time() - start:.0f}s)")

    if args.augment:
        bank = augment_bank(bank, random.Random(args.seed), n_renamed=2, n_perturbed=2, synthetic_count=50)
        logger.info(f"Augmented bank: {bank.total_entries} entries ({time.time() - start:.0f}s total)")

    bank.save(args.output)
    logger.info(f"Wrote {bank.total_entries} entries to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
