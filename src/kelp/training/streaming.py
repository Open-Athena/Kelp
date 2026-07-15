# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming, seed-driven training-data generation.

A drop-in replacement for ``create_edit_data_iter`` that *generates* examples on
the fly from the corpus + SubtreeBank instead of materializing them. Because the
generative space (programs x corruption seeds x subtree-bank samples x path
steps) vastly exceeds any stored set, this gives effectively unbounded diversity
and stores nothing but the corpus; a run reproduces from
``(corpus, seed, code)`` alone.

The benchmark on chainlink #118 showed that for the 8B/v5p target a handful of
CPU cores keep the TPU fed at reuse factor 1 (each example seen once); smaller,
faster-step models need either more cores or a reuse factor R>1 (each example
fed R times) to amortize generation cost. Both are knobs here.

Structure:
- A generation **source** yields ``TrainingExample`` objects. ``_inline_source``
  runs single-process (dev/CI, deterministic, no pool); ``_multiprocess_source``
  fans generation across CPU worker processes.
- A :class:`ReuseShuffleBuffer` provides shuffling, the reuse factor, and
  curriculum gating (see below).
- :func:`create_streaming_data_iter` ties them together and yields the same
  ``{"token_ids", "loss_mask"}`` batch contract as ``create_edit_data_iter``.

Curriculum: generation runs at the ceiling ``max_corruption_steps`` and each
example is tagged (via #119 metadata) with the corruption depth used. At step
``s`` the buffer only yields examples whose depth is within
``effective_max_corruption_steps(s)`` (random-branch examples are always
eligible). For the default ``constant`` curriculum this gate is a no-op. This is
an approximation of the exact online two-branch sampling -- it reproduces the
difficulty ramp but not the precise per-branch ratio; that refinement is noted
on #122/#123 if research fidelity ever requires it.

Multi-host: each process derives a disjoint seed stream from
``jax.process_index()`` so data-parallel hosts see different data. Batch size is
per-process (matching the single-host data-parallel path in #114); wiring
local->global batch assembly for multi-host is the coordination point with #114.

The Iris actor-pool backend (generation on hosts beyond the trainer, for the
small-model case) is a follow-up; this module implements the co-located pool.
"""

import logging
import multiprocessing
import random as pyrandom
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jaxtyping import Array

from kelp.training.generation import GenerationConfig, TrainingExample, generate_example
from kelp.tree.subtree_bank import SubtreeBank
from kelp.tree.tokenizer import EditTokenizer

if TYPE_CHECKING:
    from kelp.training.engine import EditTrainingConfig

logger = logging.getLogger(__name__)

_CHUNK = 64
"""Seeds submitted to the worker pool per batch of futures."""


def _seed_for(base: int, process_index: int, index: int) -> int:
    """Stable, disjoint-per-process seed for the ``index``-th example.

    Uses integer arithmetic only, so it is independent of PYTHONHASHSEED.
    """
    return (base * 1_000_003 + process_index) * 1_000_003 + index


def _generate_at(
    seed_value: int,
    corpus: list[str],
    bank: SubtreeBank,
    tokenizer: EditTokenizer,
    gen_cfg: GenerationConfig,
    max_corruption_steps: int,
    max_seq_len: int,
) -> TrainingExample | None:
    """Deterministically generate one example for a seed (mirrors create_edit_data_iter)."""
    rng = pyrandom.Random(seed_value)
    clean = rng.choice(corpus)
    return generate_example(
        clean,
        corpus,
        bank,
        tokenizer,
        max_corruption_steps=max_corruption_steps,
        gen_cfg=gen_cfg,
        rng=rng,
        max_seq_len=max_seq_len,
    )


# --- generation sources -----------------------------------------------------


def _inline_source(
    corpus: list[str],
    bank: SubtreeBank,
    tokenizer: EditTokenizer,
    gen_cfg: GenerationConfig,
    *,
    max_corruption_steps: int,
    max_seq_len: int,
    seed: int,
    process_index: int,
) -> Iterator[TrainingExample]:
    """Single-process infinite stream (deterministic; for dev/CI/laptop)."""
    index = 0
    while True:
        ex = _generate_at(
            _seed_for(seed, process_index, index), corpus, bank, tokenizer, gen_cfg, max_corruption_steps, max_seq_len
        )
        index += 1
        if ex is not None:
            yield ex


# Worker-side state, populated once per process by the pool initializer so the
# (large) bank/corpus are shipped a single time rather than per task.
_WORKER: dict = {}


def _init_worker(
    corpus: list[str],
    bank: SubtreeBank,
    tokenizer: EditTokenizer,
    gen_cfg: GenerationConfig,
    max_corruption_steps: int,
    max_seq_len: int,
) -> None:
    _WORKER.update(
        corpus=corpus,
        bank=bank,
        tokenizer=tokenizer,
        gen_cfg=gen_cfg,
        max_corruption_steps=max_corruption_steps,
        max_seq_len=max_seq_len,
    )


def _worker_generate(seed_value: int) -> TrainingExample | None:
    w = _WORKER
    return _generate_at(
        seed_value, w["corpus"], w["bank"], w["tokenizer"], w["gen_cfg"], w["max_corruption_steps"], w["max_seq_len"]
    )


def _multiprocess_source(
    corpus: list[str],
    bank: SubtreeBank,
    tokenizer: EditTokenizer,
    gen_cfg: GenerationConfig,
    *,
    max_corruption_steps: int,
    max_seq_len: int,
    seed: int,
    process_index: int,
    num_workers: int,
) -> Iterator[TrainingExample]:
    """Infinite stream fanned across ``num_workers`` CPU processes.

    Uses a spawn context so worker processes do not inherit the parent's
    initialized JAX/XLA state (fork-after-JAX is unsafe). Results are yielded in
    seed order, so the stream is deterministic given the seed.
    """
    ctx = multiprocessing.get_context("spawn")
    initargs = (corpus, bank, tokenizer, gen_cfg, max_corruption_steps, max_seq_len)
    with ProcessPoolExecutor(
        max_workers=num_workers, mp_context=ctx, initializer=_init_worker, initargs=initargs
    ) as pool:
        index = 0
        while True:
            seeds = [_seed_for(seed, process_index, index + k) for k in range(_CHUNK)]
            index += _CHUNK
            for ex in pool.map(_worker_generate, seeds):
                if ex is not None:
                    yield ex


# --- reuse + shuffle + curriculum gate --------------------------------------


class ReuseShuffleBuffer:
    """A bounded shuffle buffer that feeds each example up to ``reuse_factor`` times.

    ``draw(eligible)`` returns a random buffered example satisfying the
    ``eligible`` predicate, refilling from ``source`` as needed. Reuse factor 1
    means every example is fresh; R>1 amortizes generation cost by feeding each
    example R times before eviction.
    """

    def __init__(
        self,
        source: Iterator[TrainingExample],
        capacity: int,
        reuse_factor: int,
        rng: pyrandom.Random,
    ):
        if reuse_factor < 1:
            raise ValueError(f"reuse_factor must be >= 1, got {reuse_factor}")
        self._source = source
        self._capacity = capacity
        self._reuse = reuse_factor
        self._rng = rng
        self._items: list[list] = []  # each entry: [example, remaining_uses]

    def _fill(self, target: int) -> None:
        # Tolerates a finite source (real sources are infinite): stops filling
        # when the source is exhausted rather than propagating StopIteration.
        while len(self._items) < target:
            try:
                self._items.append([next(self._source), self._reuse])
            except StopIteration:
                break

    def draw(self, eligible: Callable[[TrainingExample], bool]) -> TrainingExample:
        """Return a random eligible example, or raise StopIteration if the source
        is finite and no eligible example remains."""
        self._fill(self._capacity)
        attempts = 0
        while True:
            if not self._items:
                self._fill(self._capacity)
                if not self._items:
                    raise StopIteration
            idx = self._rng.randrange(len(self._items))
            entry = self._items[idx]
            example: TrainingExample = entry[0]
            if eligible(example):
                entry[1] -= 1
                if entry[1] <= 0:  # spent: evict via swap-pop
                    self._items[idx] = self._items[-1]
                    self._items.pop()
                return example
            attempts += 1
            if attempts > 4 * max(self._capacity, 1):
                # Starved of eligible examples (e.g. a strict early curriculum
                # gate): pull a fresh wave. If the pool cannot grow, give up.
                before = len(self._items)
                self._fill(before + self._capacity)
                if len(self._items) == before:
                    raise StopIteration
                attempts = 0


def _all_eligible(ex: TrainingExample) -> bool:
    """Curriculum gate for the constant schedule: everything is eligible."""
    return True


def _within_difficulty(effective_max: int) -> Callable[[TrainingExample], bool]:
    """Gate admitting forward-diffusion examples up to ``effective_max`` depth;
    random-branch examples are always eligible."""

    def eligible(ex: TrainingExample) -> bool:
        return ex.is_random or ex.corruption_steps <= effective_max

    return eligible


def _stack_and_pad(examples: list[TrainingExample], batch_size: int, max_seq_len: int) -> dict[str, Array]:
    """Pad a list of examples into the {token_ids, loss_mask} batch contract."""
    padded_ids = jnp.zeros((batch_size, max_seq_len), dtype=jnp.int32)
    padded_masks = jnp.zeros((batch_size, max_seq_len), dtype=jnp.float32)
    for i, ex in enumerate(examples):
        n = len(ex.token_ids)
        padded_ids = padded_ids.at[i, :n].set(jnp.array(ex.token_ids, dtype=jnp.int32))
        padded_masks = padded_masks.at[i, :n].set(jnp.array(ex.loss_mask, dtype=jnp.float32))
    return {"token_ids": padded_ids, "loss_mask": padded_masks}


def create_streaming_data_iter(
    corpus: list[str],
    bank: SubtreeBank,
    tokenizer: EditTokenizer,
    config: "EditTrainingConfig",
    *,
    seed: int = 42,
    num_workers: int = 1,
    reuse_factor: int = 1,
    buffer_size: int = 1024,
    process_index: int | None = None,
    process_count: int | None = None,
) -> Iterator[dict[str, Array]]:
    """Stream training batches by generating examples on the fly.

    Drop-in for ``create_edit_data_iter``: yields ``{"token_ids", "loss_mask"}``
    with the same shapes/dtypes so ``train_edit_model`` is unchanged.

    Args:
        seed: Base seed. Combined with ``process_index`` for a disjoint,
            reproducible per-process stream.
        num_workers: CPU generation processes. ``<=1`` runs inline (no pool).
        reuse_factor: Times each example is fed before eviction (>=1).
        buffer_size: Shuffle-buffer capacity (examples).
        process_index / process_count: Data-parallel host coordinates; default to
            ``jax.process_index()`` / ``jax.process_count()``.
    """
    if process_index is None:
        process_index = jax.process_index()
    if process_count is None:
        process_count = jax.process_count()

    gen_cfg = GenerationConfig.from_training_config(config)
    ceiling = config.max_corruption_steps
    max_seq_len = config.max_seq_len

    common = dict(max_corruption_steps=ceiling, max_seq_len=max_seq_len, seed=seed, process_index=process_index)
    if num_workers <= 1:
        source = _inline_source(corpus, bank, tokenizer, gen_cfg, **common)
    else:
        source = _multiprocess_source(corpus, bank, tokenizer, gen_cfg, num_workers=num_workers, **common)
        logger.info("Streaming dataloader: %d gen workers, reuse=%d, buffer=%d", num_workers, reuse_factor, buffer_size)

    buffer = ReuseShuffleBuffer(source, capacity=buffer_size, reuse_factor=reuse_factor, rng=pyrandom.Random(seed))
    step = 0
    try:
        while True:
            effective_max = config.effective_max_corruption_steps(step)
            # Fast path: constant curriculum (effective==ceiling) admits everything.
            eligible = _all_eligible if effective_max >= ceiling else _within_difficulty(effective_max)

            batch = [buffer.draw(eligible) for _ in range(config.batch_size)]
            yield _stack_and_pad(batch, config.batch_size, max_seq_len)
            step += 1
    finally:
        # Close the generator source (and its worker pool, if any).
        close = getattr(source, "close", None)
        if close is not None:
            close()
