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

"""Prepare a large Python function corpus for Kelp tree diffusion training.

Combines multiple sources to maximize diversity:
1. Local Python repos: extract functions from source trees
2. MBPP: curated coding problems from Google Research (~374 programs)
3. HumanEval: OpenAI's coding benchmark (~164 programs)
4. codeparrot/github-code: streaming Python from GitHub (~5,000+ functions)
5. HuggingFaceTB/stack-edu: educational Python code filtered from The Stack v2

Eval task decontamination: programs matching kelp.eval_tasks.EVAL_TASKS
are automatically excluded from the training corpus to prevent train/test leakage.

Output format: one program per block, separated by '# ---' sentinel lines,
compatible with train.py --corpus-file flag.

Usage:
    uv run python -m kelp.cli.prepare_corpus --output corpus.txt
    uv run python -m kelp.cli.prepare_corpus --output corpus.txt --source-dirs /path/to/repos
    uv run python -m kelp.cli.prepare_corpus --output corpus.txt --no-github  # offline mode
    uv run python -m kelp.cli.prepare_corpus --output corpus_v7.txt --stack-edu-max 50000
"""

import argparse
import ast
import io
import json
import logging
import random
import sys
import textwrap
from pathlib import Path

from etils import epath

from kelp.corpus import CORPUS_SEPARATOR, extract_docstring, normalize_program
from kelp.eval_tasks import EVAL_SIGNATURES
from kelp.tree.corruption import has_corruptible_content

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

EXCLUDE_DIRS = {
    ".venv",
    "__pycache__",
    ".git",
    "node_modules",
    "checkpoints",
    ".eggs",
    "build",
    "dist",
    "test",
    "tests",
    # Nested third-party/vendored trees: a bundled site-packages (e.g. under a
    # stdlib install) or a _vendor dir pulls in code of unknown/mixed provenance.
    # Matched relative to source_dir, so this never blocks a library pointed at
    # directly (its own path lives *under* an ancestor site-packages).
    "site-packages",
    "dist-packages",
    "_vendor",
    "vendored",
}

# Eval-task decontamination signatures live in kelp.eval_tasks (derived from
# EVAL_TASKS), so they can never drift out of sync with the eval set.


def extract_functions_from_file(source: str, max_length: int, *, require_docstring: bool = False) -> list[str]:
    """Extract individual function definitions from a Python source file.

    Returns dedented function source strings that are parseable and under
    max_length. When ``require_docstring`` is set, only functions that carry a
    docstring are kept -- used to build a prompt-conditioning corpus where the
    docstring supplies the intent signal.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    functions = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        func_src = ast.get_source_segment(source, node)
        if func_src is None:
            continue
        # Dedent in case the function is nested inside a class.
        func_src = textwrap.dedent(func_src)
        if len(func_src) > max_length:
            continue
        if len(func_src) < 20:
            continue
        # Verify it parses standalone.
        try:
            ast.parse(func_src)
        except SyntaxError:
            continue
        if require_docstring and extract_docstring(func_src) is None:
            continue
        functions.append(func_src)
    return functions


def extract_local_functions(source_dir: Path, max_length: int, *, require_docstring: bool = False) -> list[str]:
    """Extract Python functions from a local directory tree.

    When ``require_docstring`` is set, only functions carrying a docstring are
    kept -- used to build a prompt-conditioning corpus from library source (where
    the docstring is the intent signal).
    """
    logger.info(f"Extracting functions from {source_dir}...")
    functions = []
    file_count = 0

    for py_file in source_dir.rglob("*.py"):
        # Match EXCLUDE_DIRS against the path *relative to* source_dir, so an
        # explicitly-requested library under .venv/site-packages is scanned
        # (only nested build/test/venv dirs within the tree are skipped).
        if any(part in EXCLUDE_DIRS for part in py_file.relative_to(source_dir).parts):
            continue
        file_count += 1
        try:
            source = py_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        functions.extend(extract_functions_from_file(source, max_length, require_docstring=require_docstring))

    logger.info(f"  Scanned {file_count} files, extracted {len(functions)} functions")
    return functions


def load_mbpp_programs() -> list[str]:
    """Load programs from Google's MBPP benchmark."""
    try:
        from datasets import load_dataset

        programs = []
        for split in ["train", "validation", "test", "prompt"]:
            try:
                ds = load_dataset("google-research-datasets/mbpp", "full", split=split, trust_remote_code=True)
                for item in ds:
                    code = item.get("code", "")
                    if code:
                        programs.append(code)
            except Exception:
                continue
        logger.info(f"  MBPP: loaded {len(programs)} programs")
        return programs
    except ImportError:
        logger.warning("  MBPP: 'datasets' library not installed, skipping")
        return []
    except Exception as e:
        logger.warning(f"  MBPP: failed to load: {e}")
        return []


def load_humaneval_programs() -> list[str]:
    """Load programs from OpenAI's HumanEval benchmark."""
    try:
        from datasets import load_dataset

        ds = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)
        programs = []
        for item in ds:
            prompt = item.get("prompt", "")
            solution = item.get("canonical_solution", "")
            full = prompt + solution
            if full.strip():
                programs.append(full)
        logger.info(f"  HumanEval: loaded {len(programs)} programs")
        return programs
    except ImportError:
        logger.warning("  HumanEval: 'datasets' library not installed, skipping")
        return []
    except Exception as e:
        logger.warning(f"  HumanEval: failed to load: {e}")
        return []


def stream_github_code(max_functions: int, max_length: int) -> list[str]:
    """Stream Python functions from codeparrot/github-code.

    Iterates over Python files from GitHub, extracting individual functions
    until we reach the target count.
    """
    try:
        from datasets import load_dataset

        logger.info(f"  Streaming github-code (target: {max_functions} functions)...")
        ds = load_dataset(
            "codeparrot/github-code",
            streaming=True,
            split="train",
            languages=["Python"],
            trust_remote_code=True,
        )

        functions = []
        files_scanned = 0
        for item in ds:
            files_scanned += 1
            code = item.get("code", "")
            funcs = extract_functions_from_file(code, max_length)
            functions.extend(funcs)
            if files_scanned % 1000 == 0:
                logger.info(f"    Scanned {files_scanned} files, {len(functions)} functions so far...")
            if len(functions) >= max_functions:
                functions = functions[:max_functions]
                break

        logger.info(f"  github-code: extracted {len(functions)} functions from {files_scanned} files")
        return functions
    except ImportError:
        logger.warning("  github-code: 'datasets' library not installed, skipping")
        return []
    except Exception as e:
        logger.warning(f"  github-code: failed to stream: {e}")
        return []


def stream_stack_edu(max_functions: int, max_length: int) -> list[str]:
    """Stream educational Python functions from HuggingFaceTB/stack-edu.

    Stack Edu contains Python code filtered from The Stack v2 by an
    educational quality classifier. This is a high-quality source with
    strong docstring coverage, ideal for prompt-conditioned training.
    """
    try:
        from datasets import load_dataset

        logger.info(f"  Streaming stack-edu Python (target: {max_functions} functions)...")
        ds = load_dataset(
            "HuggingFaceTB/stack-edu",
            "Python",
            streaming=True,
            split="train",
            trust_remote_code=True,
        )

        functions: list[str] = []
        files_scanned = 0
        for item in ds:
            files_scanned += 1
            code = item.get("content", "")
            if not code:
                continue
            funcs = extract_functions_from_file(code, max_length)
            functions.extend(funcs)
            if files_scanned % 5000 == 0:
                logger.info(f"    Scanned {files_scanned} files, {len(functions)} functions so far...")
            if len(functions) >= max_functions:
                functions = functions[:max_functions]
                break

        logger.info(f"  stack-edu: extracted {len(functions)} functions from {files_scanned} files")
        return functions
    except ImportError:
        logger.warning("  stack-edu: 'datasets' library not installed, skipping")
        return []
    except Exception as e:
        logger.warning(f"  stack-edu: failed to stream: {e}")
        return []


# Marin's Stack Edu Python mirror, colocated with the us-east5 training slice.
DEFAULT_STACK_EDU_GCS = "gs://marin-us-east5/documents/stack_edu/Python-4da2c7/train"


def stream_stack_edu_gcs(
    gcs_path: str, max_functions: int, max_length: int, *, require_docstring: bool = False
) -> list[str]:
    """Stream educational Python functions from Marin's Stack Edu mirror in GCS.

    Reads dolma-format JSONL shards (zstd-compressed) straight from a Marin GCS
    bucket -- no HuggingFace token or rate limits, and zero-egress when the
    bucket is colocated with the training slice (e.g. us-east5). Each record's
    ``text`` field holds a raw Python source file; individual function
    definitions are extracted from it, matching stream_stack_edu's contract.

    ``gcs_path`` is a prefix (a directory of ``*.jsonl.zst`` shards); shards are
    read in sorted order so a given max_functions is reproducible.
    """
    try:
        import gcsfs
        import zstandard
    except ImportError:
        logger.warning("  stack-edu-gcs: gcsfs/zstandard not installed, skipping")
        return []

    logger.info(f"  Streaming stack-edu from GCS {gcs_path} (target: {max_functions} functions)...")
    try:
        fs = gcsfs.GCSFileSystem()
        shards = sorted(fs.glob(gcs_path.rstrip("/") + "/**/*.jsonl.zst"))
    except Exception as e:
        logger.warning(f"  stack-edu-gcs: failed to list shards under {gcs_path}: {e}")
        return []
    if not shards:
        logger.warning(f"  stack-edu-gcs: no .jsonl.zst shards under {gcs_path}")
        return []

    functions: list[str] = []
    files_scanned = 0
    dctx = zstandard.ZstdDecompressor()
    for shard in shards:
        if len(functions) >= max_functions:
            break
        try:
            with fs.open(shard, "rb") as raw, dctx.stream_reader(raw) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                for line in text_stream:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        code = json.loads(line).get("text", "")
                    except json.JSONDecodeError:
                        continue
                    if not code:
                        continue
                    files_scanned += 1
                    functions.extend(extract_functions_from_file(code, max_length, require_docstring=require_docstring))
                    if files_scanned % 5000 == 0:
                        logger.info(f"    Scanned {files_scanned} files, {len(functions)} functions so far...")
                    if len(functions) >= max_functions:
                        break
        except Exception as e:
            logger.warning(f"  stack-edu-gcs: failed reading {shard}: {e}")
            continue

    functions = functions[:max_functions]
    logger.info(f"  stack-edu-gcs: extracted {len(functions)} functions from {files_scanned} files")
    return functions


def deduplicate_and_filter(programs: list[str], max_length: int) -> list[str]:
    """Remove duplicates, blocklisted eval programs, and filter.

    Keeps only programs that:
    - Parse as valid Python
    - Are under max_length characters
    - Are not trivially short (<20 chars)
    - Are unique (exact match dedup)
    - Do NOT match any EVAL_SIGNATURES entry (train/test decontamination)
    """
    seen = set()
    filtered = []
    blocked = 0

    for prog in programs:
        prog = prog.rstrip()
        if not prog.endswith("\n"):
            prog = prog + "\n"

        if len(prog) > max_length or len(prog) < 20:
            continue

        # Normalize whitespace for dedup but keep original.
        key = prog.strip()
        if key in seen:
            continue
        seen.add(key)

        # Decontamination: exclude programs containing eval task signatures.
        # Catches both direct matches and indirect leakage (e.g., test fixtures
        # that embed eval programs as string literals).
        if any(sig in prog for sig in EVAL_SIGNATURES):
            blocked += 1
            continue

        try:
            ast.parse(prog)
        except SyntaxError:
            continue

        filtered.append(prog)

    if blocked:
        logger.info(f"  Decontamination: blocked {blocked} programs matching eval tasks")

    return filtered


def write_corpus(programs: list[str], output_path: str | Path) -> None:
    """Write programs to a local or ``gs://`` corpus file, '# ---'-separated.

    This separator allows programs to contain internal blank lines,
    unlike the previous blank-line-separated format.
    """
    parts: list[str] = []
    for i, prog in enumerate(programs):
        if i > 0:
            parts.append(CORPUS_SEPARATOR + "\n")
        parts.append(prog.rstrip("\n") + "\n")
    epath.Path(output_path).write_text("".join(parts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a Python function corpus for Kelp training")
    parser.add_argument("--output", type=str, required=True, help="Output corpus file path")
    parser.add_argument(
        "--source-dirs",
        type=str,
        nargs="+",
        default=None,
        help="Directories to scan for Python files (default: auto-detect marin repo root)",
    )
    parser.add_argument("--max-length", type=int, default=512, help="Maximum function length in characters")
    parser.add_argument("--max-github", type=int, default=5000, help="Max functions to stream from github-code")
    parser.add_argument("--no-github", action="store_true", help="Skip github-code streaming (offline mode)")
    parser.add_argument("--no-hf", action="store_true", help="Skip all HuggingFace datasets (fully offline)")
    parser.add_argument(
        "--holdout-mbpp",
        action="store_true",
        default=True,
        help="Exclude all MBPP programs from training corpus (default: True, since evaluate_mbpp.py uses all splits)",
    )
    parser.add_argument(
        "--include-mbpp",
        dest="holdout_mbpp",
        action="store_false",
        help="Include MBPP programs in training (WARNING: contaminates evaluate_mbpp.py eval)",
    )
    parser.add_argument(
        "--stack-edu-max",
        type=int,
        default=0,
        help="Max functions to stream from Stack Edu Python (0=skip, e.g. 50000)",
    )
    parser.add_argument(
        "--stack-edu-gcs",
        type=str,
        nargs="?",
        const=DEFAULT_STACK_EDU_GCS,
        default=None,
        help="Stream Stack Edu from Marin's GCS mirror (jsonl.zst) instead of HuggingFace, using "
        f"--stack-edu-max as the target count. Bare flag uses the default us-east5 path ({DEFAULT_STACK_EDU_GCS}); "
        "no HF token needed and zero-egress when colocated with the training slice.",
    )
    parser.add_argument(
        "--no-local",
        action="store_true",
        help="Skip extracting functions from local source directories (Stack Edu / HF only).",
    )
    parser.add_argument(
        "--require-docstring",
        action="store_true",
        help="Keep only functions that carry a docstring. Builds a corpus for prompt conditioning, "
        "where the docstring is the intent signal (function-level docstring coverage is otherwise low).",
    )
    parser.add_argument(
        "--require-corruptible",
        action="store_true",
        help="Keep only functions that admit a realistic in-context corruption (a flippable operator "
        "or >=2 in-scope names). Drops trivial functions (abstract stubs, one-line wrappers) whose "
        "only corruption is an alien bank swap. Write to a NEW output path -- never overwrite an "
        "existing corpus. Off by default; the training-time --no-bank-swap-fallback drops these "
        "at load time regardless, so regenerating the corpus is optional.",
    )
    parser.add_argument(
        "--require-spec",
        action="store_true",
        help="Keep only functions for which an executable assert spec can be synthesized "
        "(sandbox-validated doctests or deterministic value fuzzing; see kelp.spec_synthesis). "
        "The spec-conditioning corpus filter (kelp_v2.md M2): standalone, executable functions "
        "only -- class methods and I/O-bound functions are dropped. Write specs with --spec-output.",
    )
    parser.add_argument(
        "--spec-output",
        type=str,
        default=None,
        help="With --require-spec: write the synthesized specs as a JSONL sidecar (content-hash "
        "keyed, the format kelp-train --spec-file consumes) to this path (local or gs://).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    # Resolve source directories.
    if args.source_dirs:
        source_dirs = [Path(d) for d in args.source_dirs]
    else:
        source_dirs = [Path(__file__).resolve().parents[2]]
    logger.info(f"Source directories: {[str(d) for d in source_dirs]}")

    all_programs: list[str] = []

    # Source 1: Local Python functions from source directories.
    if args.no_local:
        logger.info("  Local extraction: skipped (--no-local)")
    else:
        for source_dir in source_dirs:
            local_funcs = extract_local_functions(source_dir, args.max_length, require_docstring=args.require_docstring)
            all_programs.extend(local_funcs)

    # Source: Stack Edu from Marin GCS (works offline from HuggingFace).
    if args.stack_edu_max > 0 and args.stack_edu_gcs:
        all_programs.extend(
            stream_stack_edu_gcs(
                args.stack_edu_gcs, args.stack_edu_max, args.max_length, require_docstring=args.require_docstring
            )
        )

    if not args.no_hf:
        # Source 2: MBPP (skip if held out for eval).
        if not args.holdout_mbpp:
            mbpp = load_mbpp_programs()
            all_programs.extend(mbpp)
        else:
            logger.info("  MBPP: held out for evaluation (--holdout-mbpp)")

        # Source 3: HumanEval.
        humaneval = load_humaneval_programs()
        all_programs.extend(humaneval)

        # Source 4: codeparrot/github-code.
        if not args.no_github:
            github = stream_github_code(args.max_github, args.max_length)
            all_programs.extend(github)

        # Source 5: HuggingFaceTB/stack-edu (educational Python code).
        # Skipped when the Marin GCS mirror is used (handled above) to avoid double-pulling.
        if args.stack_edu_max > 0 and not args.stack_edu_gcs:
            stack_edu = stream_stack_edu(args.stack_edu_max, args.max_length)
            all_programs.extend(stack_edu)

    # Deduplicate and filter.
    logger.info(f"Total raw programs: {len(all_programs)}")
    filtered = deduplicate_and_filter(all_programs, args.max_length)
    logger.info(f"After dedup/filter: {len(filtered)} programs")

    # Optional: drop programs with no realistic corruption (trivial functions).
    if args.require_corruptible:
        before = len(filtered)
        filtered = [p for p in filtered if has_corruptible_content(p)]
        logger.info(f"After --require-corruptible: {len(filtered)} programs ({before - len(filtered)} dropped)")

    # Normalize each program to exactly what load_corpus will return at train
    # time (per-line rstrip). Content-hash-keyed sidecars are built from this
    # text; without it, a trailing space on any internal line silently orphans
    # the program's spec at training (the sidecar-key mismatch bug).
    filtered = [normalize_program(p) for p in filtered]

    # Optional: keep only functions with a synthesizable executable spec
    # (kelp_v2.md M2). Executes every candidate in the sandboxed worker; on
    # curated_v2/stack-edu this drops ~90% (methods, I/O-bound), so aim the
    # raw pull ~10x above the target corpus size.
    spec_lines: list[str] = []
    if args.require_spec:
        from kelp.spec_synthesis import sidecar_record, synthesize_spec

        before = len(filtered)
        kept: list[str] = []
        for p in filtered:
            result = synthesize_spec(p)
            if result.spec is not None:
                kept.append(p)
                spec_lines.append(sidecar_record(p, result))
        filtered = kept
        logger.info(f"After --require-spec: {len(filtered)} programs ({before - len(filtered)} dropped)")

    # Shuffle for training diversity.
    rng.shuffle(filtered)

    # Write output (local or gs://).
    output_path = epath.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_corpus(filtered, output_path)

    if spec_lines and args.spec_output:
        spec_path = epath.Path(args.spec_output)
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text("\n".join(spec_lines) + "\n")
        logger.info(f"Wrote {len(spec_lines)} specs to {spec_path}")

    # Report stats.
    total_chars = sum(len(p) for p in filtered)
    avg_len = total_chars / max(len(filtered), 1)
    num_with_docstrings = sum(1 for p in filtered if extract_docstring(p) is not None)
    logger.info(f"Wrote {len(filtered)} programs to {output_path}")
    logger.info(f"Total characters: {total_chars:,}")
    logger.info(f"Average length: {avg_len:.0f} chars")
    logger.info(f"With docstrings: {num_with_docstrings} ({100 * num_with_docstrings / max(len(filtered), 1):.1f}%)")


if __name__ == "__main__":
    main()
