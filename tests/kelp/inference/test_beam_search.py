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

"""Tests for beam search inference."""

import jax
import pytest

from kelp.inference.beam_search import (
    BeamCandidate,
    _ar_generate_tokens,
    beam_search,
    best_of_n,
    generate_edit,
)
from kelp.model.config import EditModelConfig
from kelp.model.edit_model import init_edit_params
from kelp.tree.mutation import Mutation
from kelp.tree.tokenizer import EditTokenizer

MAX_SEQ_LEN = 128


@pytest.fixture
def tokenizer():
    return EditTokenizer(max_seq_len=MAX_SEQ_LEN)


@pytest.fixture
def model_cfg(tokenizer):
    return EditModelConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=64,
        intermediate_dim=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        max_seq_len=MAX_SEQ_LEN,
    )


@pytest.fixture
def params(model_cfg):
    return init_edit_params(model_cfg, key=jax.random.PRNGKey(0))


# --- BeamCandidate ---


def test_beam_candidate_creation():
    c = BeamCandidate(source="x = 1\n", score=-2.5, depth=1, edits=())
    assert c.source == "x = 1\n"
    assert c.score == -2.5
    assert c.depth == 1
    assert c.edits == ()


def test_beam_candidate_with_edits():
    m = Mutation(start=4, end=5, replacement="2", node_type="Constant", original="1")
    c = BeamCandidate(source="x = 2\n", score=-1.0, depth=1, edits=(m,))
    assert len(c.edits) == 1
    assert c.edits[0].replacement == "2"


# --- _ar_generate_tokens ---


def test_ar_generate_produces_tokens(params, model_cfg, tokenizer):
    """AR generation should produce at least one token."""
    context = tokenizer.encode_source("x = 1\n")
    generated, log_prob = _ar_generate_tokens(
        params=params,
        context_token_ids=context,
        cfg=model_cfg,
        tokenizer=tokenizer,
        key=jax.random.PRNGKey(42),
        temperature=1.0,
        max_new_tokens=10,
    )
    assert len(generated) > 0
    assert isinstance(log_prob, float)
    assert log_prob <= 0.0  # Log-probs are non-positive.


def test_ar_generate_respects_max_tokens(params, model_cfg, tokenizer):
    """Generation should not exceed max_new_tokens."""
    context = tokenizer.encode_source("x = 1\n")
    max_tokens = 5
    generated, _ = _ar_generate_tokens(
        params=params,
        context_token_ids=context,
        cfg=model_cfg,
        tokenizer=tokenizer,
        key=jax.random.PRNGKey(0),
        temperature=1.0,
        max_new_tokens=max_tokens,
    )
    assert len(generated) <= max_tokens


def test_ar_generate_stops_at_eos(params, model_cfg, tokenizer):
    """If EOS is generated, it should be the last token."""
    context = tokenizer.encode_source("x = 1\n")
    generated, _ = _ar_generate_tokens(
        params=params,
        context_token_ids=context,
        cfg=model_cfg,
        tokenizer=tokenizer,
        key=jax.random.PRNGKey(7),
        temperature=1.0,
        max_new_tokens=50,
    )
    if tokenizer.eos_token_id in generated:
        assert generated[-1] == tokenizer.eos_token_id


def test_ar_generate_greedy(params, model_cfg, tokenizer):
    """Greedy decoding (temperature=0) should be deterministic."""
    context = tokenizer.encode_source("x = 1\n")
    gen1, lp1 = _ar_generate_tokens(
        params=params,
        context_token_ids=context,
        cfg=model_cfg,
        tokenizer=tokenizer,
        key=jax.random.PRNGKey(0),
        temperature=0.0,
        max_new_tokens=10,
    )
    gen2, lp2 = _ar_generate_tokens(
        params=params,
        context_token_ids=context,
        cfg=model_cfg,
        tokenizer=tokenizer,
        key=jax.random.PRNGKey(99),  # Different key shouldn't matter.
        temperature=0.0,
        max_new_tokens=10,
    )
    assert gen1 == gen2
    assert abs(lp1 - lp2) < 1e-5


# --- generate_edit ---


def test_generate_edit_returns_mutation_or_none(params, model_cfg, tokenizer):
    """generate_edit should return a Mutation or None."""
    source = "x = 1 + 2\n"
    mutation, log_prob = generate_edit(
        params=params,
        source=source,
        cfg=model_cfg,
        tokenizer=tokenizer,
        key=jax.random.PRNGKey(42),
        temperature=1.0,
    )
    if mutation is not None:
        assert isinstance(mutation, Mutation)
        assert log_prob <= 0.0
        # Applying the mutation should produce valid Python.
        import ast

        ast.parse(mutation.apply(source))
    else:
        assert log_prob == float("-inf")


def test_generate_edit_with_various_seeds(params, model_cfg, tokenizer):
    """Try multiple seeds; at least some should produce valid edits or None gracefully."""
    source = "def f(x):\n    return x + 1\n"
    results = []
    for seed in range(20):
        mutation, log_prob = generate_edit(
            params=params,
            source=source,
            cfg=model_cfg,
            tokenizer=tokenizer,
            key=jax.random.PRNGKey(seed),
            temperature=1.0,
        )
        results.append((mutation, log_prob))

    # All results should be well-formed.
    for mutation, log_prob in results:
        if mutation is not None:
            assert isinstance(mutation, Mutation)
            assert log_prob <= 0.0
        else:
            assert log_prob == float("-inf")


# --- beam_search ---


def test_beam_search_returns_candidates(params, model_cfg, tokenizer):
    """Beam search should return a non-empty list of candidates."""
    programs = ["x = 1 + 2\n"]
    results = beam_search(
        params=params,
        initial_programs=programs,
        cfg=model_cfg,
        tokenizer=tokenizer,
        key=jax.random.PRNGKey(0),
        beam_size=4,
        expansions_per_beam=2,
        max_depth=2,
        temperature=1.0,
    )
    assert len(results) > 0
    assert all(isinstance(c, BeamCandidate) for c in results)


def test_beam_search_sorted_by_score(params, model_cfg, tokenizer):
    """Results should be sorted: edited candidates first, then by score."""
    results = beam_search(
        params=params,
        initial_programs=["x = 1 + 2\n", "y = 3\n"],
        cfg=model_cfg,
        tokenizer=tokenizer,
        key=jax.random.PRNGKey(1),
        beam_size=4,
        expansions_per_beam=2,
        max_depth=2,
    )
    keys = [(c.depth > 0, c.score) for c in results]
    assert keys == sorted(keys, reverse=True)


def test_beam_search_respects_beam_size(params, model_cfg, tokenizer):
    """Should not return more candidates than beam_size."""
    beam_size = 3
    results = beam_search(
        params=params,
        initial_programs=["x = 1\n", "y = 2\n", "z = 3\n", "w = 4\n"],
        cfg=model_cfg,
        tokenizer=tokenizer,
        key=jax.random.PRNGKey(2),
        beam_size=beam_size,
        expansions_per_beam=2,
        max_depth=2,
    )
    assert len(results) <= beam_size


def test_beam_search_no_duplicate_sources(params, model_cfg, tokenizer):
    """Beam should deduplicate candidates with the same source."""
    results = beam_search(
        params=params,
        initial_programs=["x = 1 + 2\n"],
        cfg=model_cfg,
        tokenizer=tokenizer,
        key=jax.random.PRNGKey(3),
        beam_size=8,
        expansions_per_beam=3,
        max_depth=3,
    )
    sources = [c.source for c in results]
    assert len(sources) == len(set(sources))


def test_beam_search_includes_initial_program(params, model_cfg, tokenizer):
    """The initial program should be among the candidates (it starts with score 0)."""
    source = "x = 1\n"
    results = beam_search(
        params=params,
        initial_programs=[source],
        cfg=model_cfg,
        tokenizer=tokenizer,
        key=jax.random.PRNGKey(4),
        beam_size=8,
        expansions_per_beam=2,
        max_depth=1,
    )
    sources = {c.source for c in results}
    assert source in sources


# --- best_of_n ---


def test_best_of_n_returns_candidates(params, model_cfg, tokenizer):
    """Best-of-N should return at least one candidate."""
    results = best_of_n(
        params=params,
        source="x = 1 + 2\n",
        cfg=model_cfg,
        tokenizer=tokenizer,
        key=jax.random.PRNGKey(0),
        n=4,
        max_depth=2,
        temperature=1.0,
    )
    assert len(results) > 0
    assert all(isinstance(c, BeamCandidate) for c in results)


def test_best_of_n_sorted_by_score(params, model_cfg, tokenizer):
    """Results should be sorted: edited candidates first, then by score."""
    results = best_of_n(
        params=params,
        source="x = 1 + 2\n",
        cfg=model_cfg,
        tokenizer=tokenizer,
        key=jax.random.PRNGKey(5),
        n=4,
        max_depth=2,
    )
    keys = [(c.depth > 0, c.score) for c in results]
    assert keys == sorted(keys, reverse=True)


def test_best_of_n_no_duplicate_sources(params, model_cfg, tokenizer):
    """Best-of-N should deduplicate by source."""
    results = best_of_n(
        params=params,
        source="x = 1 + 2\n",
        cfg=model_cfg,
        tokenizer=tokenizer,
        key=jax.random.PRNGKey(6),
        n=8,
        max_depth=2,
    )
    sources = [c.source for c in results]
    assert len(sources) == len(set(sources))


def test_best_of_n_respects_n(params, model_cfg, tokenizer):
    """Should not return more unique candidates than n."""
    n = 4
    results = best_of_n(
        params=params,
        source="x = 1\n",
        cfg=model_cfg,
        tokenizer=tokenizer,
        key=jax.random.PRNGKey(7),
        n=n,
        max_depth=2,
    )
    assert len(results) <= n


def test_bucketed_pad_len_picks_smallest_fitting_bucket():
    """AR decoding pads to the smallest bucket >= needed so short programs run
    cheaper forwards; falls back to max_len when nothing fits."""
    from kelp.inference.beam_search import _bucketed_pad_len

    assert _bucketed_pad_len(100, 1024) == 128
    assert _bucketed_pad_len(300, 1024) == 512
    assert _bucketed_pad_len(513, 1024) == 1024
    assert _bucketed_pad_len(2000, 1024) == 1024  # nothing fits -> max_len
    assert _bucketed_pad_len(100, 64) == 64  # max_len below every bucket -> max_len
