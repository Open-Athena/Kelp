# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthesize executable assert specs for a corpus (kelp_v2.md M2, exp12).

Offline corpus-preparation step: runs the layered spec synthesizer
(:mod:`kelp.spec_synthesis`) over every program, writes a JSONL sidecar keyed
by content hash, and prints the coverage report that gates the exp12 training
run (target: >=~50% of programs with a non-empty spec).

Usage:
    uv run python -m kelp.cli.synthesize_specs \\
        --corpus-file gs://.../curated_v2.txt --output specs_v2.jsonl
"""

import argparse
import logging
import sys
import time

from etils import epath

from kelp.cli._logging import configure_logging
from kelp.cli._test_runner import default_runner
from kelp.corpus import load_corpus
from kelp.spec_synthesis import sidecar_record, synthesize_spec

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize executable assert specs for a corpus")
    parser.add_argument("--corpus-file", type=str, required=True, help="Corpus file (local or gs://)")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL sidecar (local or gs://)")
    parser.add_argument("--max-programs", type=int, default=0, help="Limit programs processed (0=all)")
    parser.add_argument("--max-asserts", type=int, default=4, help="Max assert lines per spec")
    parser.add_argument("--max-attempts", type=int, default=24, help="Max fuzz calls per program")
    parser.add_argument("--timeout", type=float, default=2.0, help="Per-execution timeout (seconds)")
    parser.add_argument("--report-every", type=int, default=500, help="Progress log interval (programs)")
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()

    corpus = load_corpus(args.corpus_file)
    if args.max_programs > 0:
        corpus = corpus[: args.max_programs]
    logger.info(f"Synthesizing specs for {len(corpus)} programs from {args.corpus_file}")

    runner = default_runner()
    start = time.time()
    n_any = n_doctest_only = n_fuzz_only = n_both = total_asserts = 0
    misses: dict[str, int] = {}
    lines: list[str] = []

    for i, source in enumerate(corpus):
        result = synthesize_spec(
            source,
            runner=runner,
            max_asserts=args.max_asserts,
            max_attempts=args.max_attempts,
            timeout_s=args.timeout,
        )
        if result.spec is None:
            # Attribute the miss: an explicit skip, a program that doesn't even
            # exec standalone (missing library imports), or fuzzing that found
            # no passing call. One extra probe only on misses.
            reason = result.skip_reason
            if reason is None:
                reason = (
                    "no_pool_hit" if runner.run(source, "assert True", timeout_s=args.timeout) else "not_selfcontained"
                )
            misses[reason] = misses.get(reason, 0) + 1
        if result.spec is not None:
            n_any += 1
            total_asserts += len(result.asserts)
            if result.n_doctest and result.n_fuzz:
                n_both += 1
            elif result.n_doctest:
                n_doctest_only += 1
            else:
                n_fuzz_only += 1
            lines.append(sidecar_record(source, result))
        if args.report_every and (i + 1) % args.report_every == 0:
            logger.info(f"[{i + 1}/{len(corpus)}] coverage so far: {n_any / (i + 1):.1%} ({time.time() - start:.0f}s)")

    epath.Path(args.output).write_text("\n".join(lines) + ("\n" if lines else ""))

    n = len(corpus)
    logger.info("=" * 60)
    logger.info("Spec synthesis coverage report")
    logger.info("=" * 60)
    logger.info(f"Programs:                {n}")
    logger.info(f"With spec (coverage):    {n_any} ({n_any / max(n, 1):.1%})")
    logger.info(f"  doctest-only:          {n_doctest_only}")
    logger.info(f"  fuzz-only:             {n_fuzz_only}")
    logger.info(f"  both layers:           {n_both}")
    logger.info(f"Avg asserts per spec:    {total_asserts / max(n_any, 1):.2f}")
    logger.info("Miss breakdown:")
    for reason, count in sorted(misses.items(), key=lambda kv: -kv[1]):
        logger.info(f"  {reason:<22} {count} ({count / max(n, 1):.1%})")
    logger.info(f"Elapsed:                 {time.time() - start:.0f}s")
    logger.info(f"Sidecar written:         {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
