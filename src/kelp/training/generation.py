# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure training-example generation (corruption -> TreeDiff -> encode).

This is the side-effect-free core of the data pipeline, extracted so it can run
on remote workers (Iris actors / Zephyr) as well as inline on the training host.
It is decoupled from the training loop in two ways:

- No training ``step``: corruption difficulty is passed in explicitly as
  ``max_corruption_steps``. The caller decides that value -- the inline loop and
  the streaming dataloader both derive it from the curriculum
  (``EditTrainingConfig.effective_max_corruption_steps(step)``), so difficulty
  stays a load-time knob rather than being baked into stored data.
- No ``EditTrainingConfig``: only the per-example knobs it needs travel in a
  small, hashable :class:`GenerationConfig`.

Everything here is picklable (module-level function + frozen-dataclass args +
picklable ``SubtreeBank`` / ``EditTokenizer``), so a worker closure over
``(generate_example, bank, tokenizer, gen_cfg)`` ships cleanly. Determinism is
the caller's job: seed a ``random.Random`` from ``(dataset_seed, index)`` and the
same index always yields the same example.
"""

import random as pyrandom
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kelp.corpus import extract_docstring, extract_spec_asserts
from kelp.spec_synthesis import corpus_spec_key, load_spec_map
from kelp.tree.corruption import corrupt_realistic
from kelp.tree.subtree_bank import SubtreeBank
from kelp.tree.tokenizer import EditTokenizer
from kelp.tree.tree_diff import find_path

if TYPE_CHECKING:
    from kelp.training.engine import EditTrainingConfig


@dataclass(frozen=True)
class GenerationConfig:
    """Per-example generation knobs (decoupled from the training config)."""

    max_edit_stmts: int = 3
    """Maximum statement count per edit."""

    p_random: float = 0.2
    """Probability of using a random corpus program instead of forward diffusion."""

    p_prompt: float = 0.5
    """Probability of including a docstring prompt when one is available."""

    p_spec: float = 0.5
    """Probability of including an executable assert spec when one is
    available (requires a spec_tokens tokenizer; kelp_v2.md M2, issue #147).
    Independent of ``p_prompt`` so the {none, NL, spec, NL+spec} ablation grid
    falls out of the two dropouts."""

    spec_map: dict[str, str] | None = None
    """Precomputed specs keyed by :func:`kelp.spec_synthesis.corpus_spec_key`
    (the --spec-file sidecar from prepare_corpus --require-spec). Preferred
    over inline doctest extraction: sidecar specs are sandbox-validated and
    include fuzz-synthesized asserts. None falls back to doctests only."""

    p_near_miss: float = 0.0
    """On the forward-diffusion branch, probability of corrupting with an e-graph
    near-miss (a realistic in-context operator-flip bug) instead of a bank
    subtree swap. Falls back to the bank swap when no near-miss is available for
    the program. 0.0 keeps the original bank-swap corruption."""

    allow_bank_swap: bool = True
    """When False, drop (return None for) programs with no realistic corruption
    instead of injecting an out-of-context bank subtree swap. Yields
    realistic-or-drop training data with zero alien grafts; the realistic cascade
    is always attempted regardless of ``p_near_miss``."""

    @staticmethod
    def from_training_config(config: "EditTrainingConfig") -> "GenerationConfig":
        """Build from an EditTrainingConfig (the inline training path's config)."""
        return GenerationConfig(
            max_edit_stmts=config.max_edit_stmts,
            p_random=config.p_random,
            p_prompt=config.p_prompt,
            p_spec=config.p_spec,
            spec_map=load_spec_map(config.spec_file) if config.spec_file else None,
            p_near_miss=config.p_near_miss,
            allow_bank_swap=config.allow_bank_swap,
        )


@dataclass(frozen=True)
class TrainingExample:
    """A single encoded example plus the metadata a data pipeline may key on.

    (A columnar/Arrow representation of a *batch* of these -- two ``list<int>``
    columns plus the three metadata columns -- is being evaluated in issue #9 as
    a potentially faster alternative to per-example Python objects.)
    """

    token_ids: list[int]
    loss_mask: list[int]
    corruption_steps: int
    """AST mutations applied on the forward (diffusion) branch; 0 for random-program examples."""
    is_random: bool
    """True if the p_random branch (random corpus program) was taken."""
    prompt_used: bool
    """True if a docstring prompt was prepended."""

    spec_used: bool = False
    """True if a doctest-derived assert spec was prepended (issue #147)."""


def generate_example(
    clean_source: str,
    corpus: list[str],
    bank: SubtreeBank,
    tokenizer: EditTokenizer,
    *,
    max_corruption_steps: int,
    gen_cfg: GenerationConfig,
    rng: pyrandom.Random,
    max_seq_len: int,
) -> TrainingExample | None:
    """Generate one training example from a clean program.

    Following the tree-diffusion forward process with edit path [0]:
    1. With probability ``p_random``, pick a random corpus program as the
       'corrupted' version; otherwise apply 1..``max_corruption_steps`` AST
       mutations to corrupt the clean program.
    2. Compute the TreeDiff path from corrupted back to clean.
    3. Pick a random step along the path as the intermediate/target split.
    4. Encode as ``(token_ids, loss_mask)``.

    Returns None if no valid training example could be generated (empty path,
    edit position overflow, or the encoding exceeds ``max_seq_len``).

    References:
        [0] Tree Diffusion (Kapur et al., 2024). https://arxiv.org/abs/2405.20519
    """
    # Step 1: Generate a corrupted version.
    is_random = False
    corruption_steps = 0
    if rng.random() < gen_cfg.p_random and len(corpus) > 1:
        # Use a random program from the corpus.
        is_random = True
        corrupted = rng.choice(corpus)
        while corrupted == clean_source and len(corpus) > 1:
            corrupted = rng.choice(corpus)
    else:
        # Apply the shared corruption policy (realistic in-context bugs with a
        # bank-swap fallback). Same function the evaluator uses, so training and
        # eval corrupt identically -- see kelp.tree.corruption.
        corruption_steps = rng.randint(1, max_corruption_steps)
        corrupted, _mode = corrupt_realistic(
            clean_source,
            num_steps=corruption_steps,
            bank=bank,
            rng=rng,
            p_near_miss=gen_cfg.p_near_miss,
            max_edit_stmts=gen_cfg.max_edit_stmts,
            allow_bank_swap=gen_cfg.allow_bank_swap,
        )
        # allow_bank_swap=False can decline to corrupt (trivial program); the
        # empty-path check below then drops the example.

    # Step 2: Compute TreeDiff path from corrupted to clean.
    path = find_path(corrupted, clean_source, max_edit_stmts=gen_cfg.max_edit_stmts)
    if not path:
        return None

    # Step 3: Pick a random step along the path.
    step_idx = rng.randrange(len(path))

    # Apply all mutations before step_idx to get the intermediate program.
    intermediate = corrupted
    for i in range(step_idx):
        intermediate = path[i].apply(intermediate)

    # The training target is path[step_idx]: the next edit to apply.
    target_mutation = path[step_idx]

    # Step 4: Encode as a training example.
    # The edit position is the character offset in the intermediate program,
    # which maps to a token index (1:1 for byte-level tokenizer).
    edit_token_idx = tokenizer.char_offset_to_token_index(intermediate, target_mutation.start)

    # Skip if the edit position overflows the tokenizer's position range.
    if edit_token_idx >= tokenizer.num_position_tokens:
        return None

    # Optionally include a docstring prompt from the *clean* source.
    prompt_source: str | None = None
    if tokenizer.prompt_tokens:
        docstring = extract_docstring(clean_source)
        if docstring and rng.random() < gen_cfg.p_prompt:
            prompt_source = docstring

    # Optionally include an executable assert spec for the *clean* source (the
    # goal-observation conditioning of kelp_v2.md M2). Sidecar specs (sandbox-
    # validated, fuzz-augmented) take precedence; inline doctest extraction is
    # the fallback. Independent draw from the prompt so the conditioning
    # ablation grid is trainable.
    spec_source: str | None = None
    if tokenizer.spec_tokens:
        spec = None
        if gen_cfg.spec_map is not None:
            spec = gen_cfg.spec_map.get(corpus_spec_key(clean_source))
        if spec is None:
            spec = extract_spec_asserts(clean_source)
        if spec and rng.random() < gen_cfg.p_spec:
            spec_source = spec

    token_ids, loss_mask = tokenizer.encode_training_example(
        context_source=intermediate,
        edit_position_token_idx=edit_token_idx,
        replacement_source=target_mutation.replacement,
        prompt_source=prompt_source,
        spec_source=spec_source,
    )

    # Skip if too long.
    if len(token_ids) > max_seq_len:
        return None

    return TrainingExample(
        token_ids=token_ids,
        loss_mask=loss_mask,
        corruption_steps=corruption_steps,
        is_random=is_random,
        prompt_used=prompt_source is not None,
        spec_used=spec_source is not None,
    )
