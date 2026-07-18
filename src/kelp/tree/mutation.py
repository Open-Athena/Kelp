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

"""AST-based mutation for tree diffusion forward process.

Corrupts Python programs by replacing random AST subtrees with alternatives
drawn from a SubtreeBank. Every intermediate program produced by this process
is syntactically valid, following the key invariant of Tree Diffusion
(Kapur et al., 2024).
"""

import ast
import logging
import random
from dataclasses import dataclass

from kelp.tree.ast_positions import PositionedNode, iter_editable_nodes, node_source_span
from kelp.tree.subtree_bank import STATEMENT_TYPES, SubtreeBank

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Mutation:
    """A single tree edit: replace a span of source code.

    Character offsets are into the source string at the time the mutation
    was computed. After applying a mutation, all subsequent offsets shift.
    """

    start: int
    """Character start offset in source."""

    end: int
    """Character end offset in source (exclusive)."""

    replacement: str
    """Replacement source code string."""

    node_type: str
    """AST node type that was replaced (e.g., 'If', 'Call')."""

    original: str
    """Original source code that was replaced."""

    def apply(self, source: str) -> str:
        """Apply this mutation to a source string."""
        return source[: self.start] + self.replacement + source[self.end :]


def _is_string_expr(node: ast.AST) -> bool:
    """True if ``node`` is a bare string-literal statement (a docstring or a
    stray string expression).

    Bank-swapping such a node makes the repair *target* a long verbatim string
    the model must reproduce token-for-token -- synthesis, not localized repair
    -- and, since the docstring is also fed to the model as the prompt, the
    example is either degenerate (copyable from the prompt) or impossible. Both
    are excluded from bank-swap candidates. See the 2026-07-17 gut-check in
    docs/vet-cond-v1-failure-analysis.md.
    """
    if not isinstance(node, ast.Expr):
        return False
    value = node.value
    return isinstance(value, ast.JoinedStr) or (isinstance(value, ast.Constant) and isinstance(value.value, str))


def _find_candidates(
    source: str,
    tree: ast.Module,
    max_edit_stmts: int,
    bank: SubtreeBank,
) -> list[PositionedNode]:
    """Find all AST nodes eligible for mutation.

    Starts from the shared editable-node set (extractable type, valid position,
    ``stmt_count <= max_edit_stmts``) and adds the mutation-specific filters:
    3. The bank has replacement candidates of the same type.
    5. Its source segment is non-trivial (>= 5 chars).
    6. It is not a root-level node (direct child of Module.body).
    7. It is not a bare string-literal statement (docstring); see
       :func:`_is_string_expr`.
    """
    # Skip root-level nodes to prevent catastrophic corruption that replaces
    # entire top-level definitions. For single-function programs (the common
    # case in our corpus), replacing the root FunctionDef effectively requires
    # the model to synthesize a new program from scratch rather than repair
    # local edits. See DIAGNOSTIC_REPORT.md Finding 1.
    root_node_ids = {id(node) for node in tree.body}

    candidates = []
    for pn in iter_editable_nodes(source, tree, max_edit_stmts):
        if id(pn.node) in root_node_ids:
            continue
        if not bank.has_type(pn.node_type):
            continue
        if pn.end - pn.start < 5:
            continue
        if _is_string_expr(pn.node):
            continue
        candidates.append(pn)

    return candidates


def random_mutation(
    source: str,
    bank: SubtreeBank,
    max_edit_stmts: int = 3,
    rng: random.Random | None = None,
    max_retries: int = 10,
) -> Mutation | None:
    """Select a random AST subtree and replace it with a bank sample.

    Following the paper's approach: pick a random eligible node, sample a
    replacement of the same AST type from the subtree bank.

    Args:
        source: Python source code.
        bank: Subtree bank to draw replacements from.
        max_edit_stmts: Maximum statement count for editable subtrees.
        rng: Random number generator.
        max_retries: Maximum attempts to find a valid mutation.

    Returns:
        A Mutation, or None if no valid mutation could be found.
    """
    if rng is None:
        rng = random.Random()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    candidates = _find_candidates(source, tree, max_edit_stmts, bank)
    if not candidates:
        return None

    # The paper's improved variant: sample a node type first (uniform over
    # types), then sample a node of that type. This avoids bias toward types
    # that appear many times in a single program.
    type_to_candidates: dict[str, list[PositionedNode]] = {}
    for c in candidates:
        if c.node_type not in type_to_candidates:
            type_to_candidates[c.node_type] = []
        type_to_candidates[c.node_type].append(c)

    for _ in range(max_retries):
        chosen_type = rng.choice(list(type_to_candidates.keys()))
        candidate = rng.choice(type_to_candidates[chosen_type])

        replacement_entry = bank.sample_with_size(
            candidate.node_type,
            max_stmts=max_edit_stmts,
            rng=rng,
        )
        if replacement_entry is None:
            continue

        original = source[candidate.start : candidate.end]

        # Skip if replacement is identical to the original.
        if replacement_entry.source == original:
            continue

        # For statements, we need to match the indentation of the original.
        replacement_source = _match_indentation(source, candidate.start, replacement_entry.source, candidate.node_type)

        # Verify the mutation produces valid Python.
        mutated = source[: candidate.start] + replacement_source + source[candidate.end :]
        try:
            ast.parse(mutated)
        except SyntaxError:
            continue

        return Mutation(
            start=candidate.start,
            end=candidate.end,
            replacement=replacement_source,
            node_type=candidate.node_type,
            original=original,
        )

    return None


def _leading_whitespace(s: str) -> str:
    """Return the run of leading spaces/tabs at the start of ``s``."""
    n = 0
    for ch in s:
        if ch in (" ", "\t"):
            n += 1
        else:
            break
    return s[:n]


def _match_indentation(source: str, insert_offset: int, replacement: str, node_type: str) -> str:
    """Adjust indentation of the replacement to match the insertion point.

    For statements, the replacement's indentation is normalized to match the
    indentation level at the insertion point in the source. Expressions don't
    need indentation adjustment.
    """
    if node_type not in STATEMENT_TYPES:
        return replacement

    # Find the indentation of the line at insert_offset.
    line_start = source.rfind("\n", 0, insert_offset) + 1
    target_indent = _leading_whitespace(source[line_start:insert_offset])

    # Determine the replacement's current base indentation (first line).
    repl_lines = replacement.split("\n")
    if not repl_lines:
        return replacement

    repl_base_indent = _leading_whitespace(repl_lines[0])

    # Re-indent: replace the base indentation with the target indentation.
    result_lines = []
    for i, line in enumerate(repl_lines):
        if i == 0:
            result_lines.append(target_indent + line.lstrip())
        elif line.strip():
            # Remove original base indent, add target indent + relative indent.
            if line.startswith(repl_base_indent):
                relative = line[len(repl_base_indent) :]
                result_lines.append(
                    target_indent + "    " + relative.lstrip()
                    if relative.startswith(" ") or relative.startswith("\t")
                    else target_indent + relative
                )
            else:
                result_lines.append(target_indent + "    " + line.lstrip())
        else:
            result_lines.append("")

    return "\n".join(result_lines)


# ---------------------------------------------------------------------------
# Direct operator-flip corruption (realistic single-operator bugs)
# ---------------------------------------------------------------------------
#
# Unlike the e-graph near-miss, which can only corrupt expressions whose leaves
# are bare names/literals (it must convert the whole expression into its PyExpr
# term language), an operator flip edits the operator *token in place* and leaves
# both operands untouched -- byte for byte. So it fires on real code where the
# operands are calls, subscripts, or attributes (``self.f(a) + len(xs)``,
# ``arr[i] > lo``), which dominate real corpora and which the e-graph cannot
# model. The result is a plausible single-operator bug over the program's own
# subexpressions, exactly the near-miss the eval measures.

# Operator symbol as it appears in source, plus the plausible wrong operators to
# flip it to. Same arity/precedence class so the splice stays syntactically
# valid without reparenthesizing.
_BINOP_FLIPS: dict[type, tuple[str, tuple[str, ...]]] = {
    ast.Add: ("+", ("-", "*")),
    ast.Sub: ("-", ("+",)),
    ast.Mult: ("*", ("+", "//")),
    ast.Div: ("/", ("*", "//")),
    ast.FloorDiv: ("//", ("/", "*")),
    ast.Mod: ("%", ("//",)),
    ast.Pow: ("**", ("*",)),
    ast.LShift: ("<<", (">>",)),
    ast.RShift: (">>", ("<<",)),
    ast.BitOr: ("|", ("&", "^")),
    ast.BitAnd: ("&", ("|", "^")),
    ast.BitXor: ("^", ("|", "&")),
}

_CMPOP_FLIPS: dict[type, tuple[str, tuple[str, ...]]] = {
    ast.Lt: ("<", ("<=", ">")),
    ast.LtE: ("<=", ("<", ">=")),
    ast.Gt: (">", (">=", "<")),
    ast.GtE: (">=", (">", "<=")),
    ast.Eq: ("==", ("!=",)),
    ast.NotEq: ("!=", ("==",)),
}

_BOOLOP_FLIPS: dict[type, tuple[str, tuple[str, ...]]] = {
    ast.And: ("and", ("or",)),
    ast.Or: ("or", ("and",)),
}


def _find_operator_offset(source: str, symbol: str, start: int, end: int) -> int:
    """Find ``symbol`` in ``source[start:end]``, skipping ``#`` comments.

    The gap between two operands contains only whitespace, line-continuations,
    and comments besides the operator itself, so a plain ``str.find`` can match
    an operator character *inside a comment* (e.g. ``a  # a+b\\n + b``) and edit
    the comment instead of the operator. Scanning past comment spans finds the
    real operator token. Returns the offset, or -1 if not found.
    """
    i = start
    while i < end:
        if source[i] == "#":
            nl = source.find("\n", i, end)
            if nl == -1:
                return -1  # comment runs to end of the search window
            i = nl + 1
            continue
        if source.startswith(symbol, i):
            return i
        i += 1
    return -1


def _operator_sites(source: str, tree: ast.AST) -> list[tuple[int, int, str, tuple[str, ...]]]:
    """Locate flippable operator tokens.

    Returns ``(op_start, op_end, old_symbol, alternatives)`` for each binary,
    single-op comparison, boolean, or augmented-assignment operator whose token
    can be found in the source gap between its operands. The operand spans are
    never touched, so operands of any shape are fine.
    """
    sites: list[tuple[int, int, str, tuple[str, ...]]] = []

    def locate(left_end: int, right_start: int, symbol: str, alts: tuple[str, ...]) -> None:
        idx = _find_operator_offset(source, symbol, left_end, right_start)
        if idx != -1:
            sites.append((idx, idx + len(symbol), symbol, alts))

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOP_FLIPS:
            lspan, rspan = node_source_span(source, node.left), node_source_span(source, node.right)
            if lspan and rspan:
                symbol, alts = _BINOP_FLIPS[type(node.op)]
                locate(lspan[1], rspan[0], symbol, alts)
        elif isinstance(node, ast.AugAssign) and type(node.op) in _BINOP_FLIPS:
            tspan, vspan = node_source_span(source, node.target), node_source_span(source, node.value)
            if tspan and vspan:
                symbol, alts = _BINOP_FLIPS[type(node.op)]
                locate(tspan[1], vspan[0], symbol + "=", tuple(a + "=" for a in alts))
        elif isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _CMPOP_FLIPS:
            lspan, rspan = node_source_span(source, node.left), node_source_span(source, node.comparators[0])
            if lspan and rspan:
                symbol, alts = _CMPOP_FLIPS[type(node.ops[0])]
                locate(lspan[1], rspan[0], symbol, alts)
        elif isinstance(node, ast.BoolOp) and type(node.op) in _BOOLOP_FLIPS:
            symbol, alts = _BOOLOP_FLIPS[type(node.op)]
            spans = [node_source_span(source, v) for v in node.values]
            for left, right in zip(spans, spans[1:], strict=False):
                if left and right:
                    locate(left[1], right[0], symbol, alts)

    return sites


def flip_one_operator(source: str, rng: random.Random) -> Mutation | None:
    """Flip a single operator token to a plausible wrong one, in place.

    Picks a random flippable operator and swaps only its token, keeping both
    operands verbatim. Returns None if the source doesn't parse, has no
    flippable operator, or no flip yields valid Python.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    sites = _operator_sites(source, tree)
    rng.shuffle(sites)
    for op_start, op_end, old_symbol, alts in sites:
        for new_symbol in rng.sample(alts, len(alts)):
            candidate = source[:op_start] + new_symbol + source[op_end:]
            try:
                ast.parse(candidate)
            except SyntaxError:
                continue
            if candidate == source:
                continue
            return Mutation(
                start=op_start,
                end=op_end,
                replacement=new_symbol,
                node_type="operator",
                original=old_symbol,
            )
    return None


def operator_flip_corrupt_program(
    source: str,
    num_steps: int,
    rng: random.Random,
) -> tuple[str, list[Mutation]]:
    """Corrupt a program by flipping ``num_steps`` operator tokens.

    Drop-in sibling of :func:`corrupt_program` that produces realistic
    single-operator bugs over the program's own subexpressions. Re-parses after
    each flip so offsets stay valid; stops early if it runs out of flippable
    operators.
    """
    current = source
    mutations: list[Mutation] = []
    for _ in range(num_steps):
        mutation = flip_one_operator(current, rng)
        if mutation is None:
            break
        current = mutation.apply(current)
        mutations.append(mutation)
    return current, mutations


# ---------------------------------------------------------------------------
# Variable-swap corruption (wrong-variable bugs)
# ---------------------------------------------------------------------------
#
# Operator flips only fire on code that has an operator. Real functions are full
# of operator-free code -- call chains, assignments, returns -- where the classic
# realistic bug is *the wrong variable*: ``return width`` where ``height`` was
# meant. A variable swap replaces one name read with another name already present
# in the program, so the corruption stays in-context (no alien tokens) and edits
# exactly one leaf, which find_path recovers as a single clean edit.


def swap_one_variable(source: str, rng: random.Random) -> Mutation | None:
    """Replace one name read with a different name already used in the program.

    Picks a random ``ast.Name`` in load context and swaps it for another name
    drawn from the program's own name pool. Returns None if the source doesn't
    parse, has fewer than two distinct names, or no swap yields valid Python.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    name_pool = sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)})
    if len(name_pool) < 2:
        return None

    loads = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)]
    rng.shuffle(loads)
    for node in loads:
        span = node_source_span(source, node)
        if span is None:
            continue
        start, end = span
        # Guard against comprehension/whitespace surprises: the span must be
        # exactly the identifier we think it is.
        if source[start:end] != node.id:
            continue
        alternatives = [name for name in name_pool if name != node.id]
        for replacement in rng.sample(alternatives, len(alternatives)):
            candidate = source[:start] + replacement + source[end:]
            try:
                ast.parse(candidate)
            except SyntaxError:
                continue
            if candidate == source:
                continue
            return Mutation(
                start=start,
                end=end,
                replacement=replacement,
                node_type="Name",
                original=node.id,
            )
    return None


def variable_swap_corrupt_program(
    source: str,
    num_steps: int,
    rng: random.Random,
) -> tuple[str, list[Mutation]]:
    """Corrupt a program by swapping ``num_steps`` name reads for other in-scope names.

    Sibling of :func:`operator_flip_corrupt_program` for operator-free code.
    Re-parses after each swap so offsets stay valid; stops early when no further
    swap is possible.
    """
    current = source
    mutations: list[Mutation] = []
    for _ in range(num_steps):
        mutation = swap_one_variable(current, rng)
        if mutation is None:
            break
        current = mutation.apply(current)
        mutations.append(mutation)
    return current, mutations


def corrupt_program(
    source: str,
    num_steps: int,
    bank: SubtreeBank,
    max_edit_stmts: int = 3,
    rng: random.Random | None = None,
) -> tuple[str, list[Mutation]]:
    """Apply multiple random mutations to corrupt a program.

    The forward (noise) process for tree diffusion. Each mutation replaces
    a random AST subtree with an alternative from the subtree bank, producing
    a valid but semantically different program.

    Args:
        source: Clean Python source code.
        num_steps: Number of mutations to apply.
        bank: Subtree bank to draw replacements from.
        max_edit_stmts: Maximum statement count per edit.
        rng: Random number generator.

    Returns:
        Tuple of (corrupted_source, list_of_mutations_applied).
        The mutations are in application order. If fewer than num_steps
        mutations could be found, fewer are returned.
    """
    if rng is None:
        rng = random.Random()

    current = source
    mutations: list[Mutation] = []

    for _ in range(num_steps):
        mutation = random_mutation(
            current,
            bank,
            max_edit_stmts=max_edit_stmts,
            rng=rng,
        )
        if mutation is None:
            break

        current = mutation.apply(current)
        mutations.append(mutation)

    return current, mutations
