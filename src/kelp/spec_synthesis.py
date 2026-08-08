# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Executable spec synthesis for corpus functions (kelp_v2.md M2, exp12).

Produces a block of ``assert f(...) == literal`` lines for a clean corpus
program by *executing it* in the sandboxed worker, layered by quality-per-cost:

1. **Doctests** (highest quality, human-written): the doctest-derived asserts
   from :func:`kelp.corpus.extract_spec_asserts`, kept only if they actually
   pass -- stale doctests are silently dropped rather than trained on.
2. **Value fuzzing**: candidate inputs are literals mined from the function's
   own AST plus a small pool of canonical values, applied to the function's
   signature. A call becomes an assert only if it is deterministic (the worker
   evaluates twice and compares), pure enough to reproduce, and both the call
   and its result repr round-trip through ``ast.literal_eval``.

Everything is deterministic per program (sampling order is seeded from the
source hash), so re-running synthesis over a corpus reproduces the sidecar
byte-for-byte. Designed to run OFFLINE in corpus preparation -- never in the
streaming training path, where per-example execution would crater throughput.

Hypothesis strategies (``st.from_type``) are a possible third layer for
type-annotated functions if measured coverage comes in under target; the
acceptance filter here would apply to its draws unchanged.
"""

import ast
import hashlib
import logging
import random
from dataclasses import dataclass, field

from kelp.cli._test_runner import SubprocessTestRunner, default_runner
from kelp.corpus import extract_spec_asserts

logger = logging.getLogger(__name__)

# Canonical values covering the common shapes of MBPP-style function arguments.
# Small on purpose: characteristic examples, not falsification.
_VALUE_POOL: list[object] = [0, 1, 2, 3, -1, 10, "", "a", "abc", [], [1, 2, 3], [3, 1, 2], (1, 2), {}, True, False]

# Functions whose AST mentions these names are skipped before spending any
# sandbox calls: they are I/O-bound or nondeterministic by construction. The
# worker's double-eval check would catch most anyway; this just saves attempts.
_IMPURE_NAMES = frozenset(
    {"open", "input", "random", "randint", "shuffle", "time", "sleep", "urandom", "getenv", "environ", "uuid4"}
)

_MAX_REPR_LEN = 80
"""Reject call/result reprs longer than this: specs compete with code for the
model's 1024-token budget, and a huge literal is a bad example anyway."""


@dataclass
class SpecResult:
    """Synthesized spec plus provenance counts for the coverage report."""

    asserts: list[str] = field(default_factory=list)
    n_doctest: int = 0
    n_fuzz: int = 0
    skip_reason: str | None = None
    """Why fuzzing was not attempted (method / wide_signature / impure /
    no_function), for the coverage report's miss breakdown. None when fuzzing
    ran (regardless of hits)."""

    @property
    def spec(self) -> str | None:
        return "\n".join(self.asserts) if self.asserts else None


def _first_function(source: str) -> ast.FunctionDef | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node
    return None


def _mentions_impure_names(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return True
    return any(isinstance(n, ast.Name | ast.Attribute) and _node_name(n) in _IMPURE_NAMES for n in ast.walk(tree))


def _node_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _mined_literals(fn: ast.FunctionDef) -> list[object]:
    """Constants appearing in the function body -- likely-meaningful inputs
    (thresholds, sentinel strings) that a generic pool would never guess."""
    out: list[object] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float | str | bool):
            v = node.value
            if len(repr(v)) <= 20 and v not in out:
                out.append(v)
    return out


def _literal_ok(text: str) -> bool:
    if len(text) > _MAX_REPR_LEN:
        return False
    try:
        ast.literal_eval(text)
        return True
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return False


def synthesize_spec(
    source: str,
    runner: SubprocessTestRunner | None = None,
    max_asserts: int = 4,
    max_attempts: int = 24,
    timeout_s: float = 2.0,
) -> SpecResult:
    """Synthesize an executable assert spec for the first function in ``source``.

    Returns a :class:`SpecResult`; ``result.spec`` is None when nothing
    qualified (the caller then trains that example unconditioned, exactly like
    the spec-dropout path).
    """
    runner = runner or default_runner()
    result = SpecResult()

    # Impurity gates BOTH layers: a nondeterministic function's doctest can
    # pass on a lucky roll just as a fuzz call can, and a flaky assert in the
    # sidecar is exactly what this module promises to prevent.
    impure = _mentions_impure_names(source)

    # Layer 1: doctests, kept only if they execute green TWICE -- the same
    # determinism standard the fuzz layer gets from the worker's double-eval.
    if not impure:
        doctest_spec = extract_spec_asserts(source, max_asserts=max_asserts)
        if doctest_spec:
            for line in doctest_spec.splitlines():
                if runner.run(source, line, timeout_s=timeout_s) and runner.run(source, line, timeout_s=timeout_s):
                    result.asserts.append(line)
                    result.n_doctest += 1

    fn = _first_function(source)
    if fn is None:
        result.skip_reason = "no_function"
        return result
    if len(result.asserts) >= max_asserts:
        return result

    # Layer 2: value fuzzing over the signature. Only REQUIRED args are fuzzed
    # -- defaults cover the rest, which is how these functions are actually
    # called and vastly widens the hit surface on library code. Skip methods
    # (an unbound `self` can't be faked with pool values), *args/**kwargs,
    # wide signatures (combinatorics), and impure functions.
    arg_names = [a.arg for a in fn.args.args]
    n_required = len(arg_names) - len(fn.args.defaults)
    if arg_names and arg_names[0] in ("self", "cls"):
        result.skip_reason = "method"
        return result
    if fn.args.vararg or fn.args.kwarg or n_required > 4:
        result.skip_reason = "wide_signature"
        return result
    if impure:
        result.skip_reason = "impure"
        return result

    candidates = _mined_literals(fn) + _VALUE_POOL
    rng = random.Random(hashlib.sha1(source.encode()).hexdigest())
    seen_calls: set[str] = set()
    for _ in range(max_attempts):
        if len(result.asserts) >= max_asserts:
            break
        args = [rng.choice(candidates) for _ in range(n_required)]
        call = f"{fn.name}({', '.join(repr(a) for a in args)})"
        if call in seen_calls or len(call) > _MAX_REPR_LEN:
            continue
        seen_calls.add(call)
        out_repr = runner.eval_expr(source, call, timeout_s=timeout_s)
        if out_repr is None or not _literal_ok(out_repr):
            continue
        line = f"assert {call} == {out_repr}"
        if line not in result.asserts:
            result.asserts.append(line)
            result.n_fuzz += 1

    # An all-None spec (side-effect-style function) constrains nothing about
    # behavior -- worthless as a conditioning signal, so report no spec at all.
    if result.asserts and all(a.endswith("== None") for a in result.asserts):
        result.asserts = []
        result.n_doctest = result.n_fuzz = 0
        result.skip_reason = "all_none_outputs"
    return result


def corpus_spec_key(source: str) -> str:
    """Stable sidecar key for a corpus program (content hash, order-independent)."""
    return hashlib.sha1(source.encode()).hexdigest()[:16]


def sidecar_record(source: str, result: SpecResult) -> str:
    """One JSONL sidecar line for a program's spec -- the single source of
    truth for the record schema, shared by every sidecar writer (prepare_corpus
    --require-spec and kelp.cli.synthesize_specs) so they cannot drift."""
    import json

    return json.dumps(
        {
            "key": corpus_spec_key(source),
            "spec": result.spec,
            "n_doctest": result.n_doctest,
            "n_fuzz": result.n_fuzz,
        }
    )


def load_spec_map(path: str) -> dict[str, str]:
    """Load a JSONL spec sidecar (local or gs://) into a {key: spec} dict."""
    import json

    from etils import epath

    spec_map: dict[str, str] = {}
    for line in epath.Path(path).read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            spec_map[record["key"]] = record["spec"]
    logger.info(f"Loaded {len(spec_map)} specs from {path}")
    return spec_map
