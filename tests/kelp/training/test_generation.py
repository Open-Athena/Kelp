# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure, worker-serializable example generator."""

import pickle
import random

import pytest

from kelp.training.generation import GenerationConfig, TrainingExample, generate_example
from kelp.tree.subtree_bank import SubtreeBank
from kelp.tree.tokenizer import EditTokenizer

CORPUS = [
    "def add(a, b):\n    return a + b\n",
    "def sub(a, b):\n    return a - b\n",
    "def mul(a, b):\n    return a * b\n",
    "def square(x):\n    return x * x\n",
    "def neg(x):\n    return -x\n",
]
MAX_SEQ_LEN = 128
DEFAULT_GEN_CFG = GenerationConfig()


@pytest.fixture
def bank():
    return SubtreeBank.from_corpus(CORPUS)


@pytest.fixture
def tokenizer():
    return EditTokenizer(max_seq_len=MAX_SEQ_LEN)


def _gen(bank, tokenizer, seed, *, gen_cfg=DEFAULT_GEN_CFG, max_corruption_steps=3):
    return generate_example(
        CORPUS[0],
        CORPUS,
        bank,
        tokenizer,
        max_corruption_steps=max_corruption_steps,
        gen_cfg=gen_cfg,
        rng=random.Random(seed),
        max_seq_len=MAX_SEQ_LEN,
    )


def test_deterministic_for_same_seed(bank, tokenizer):
    """A given seed always yields the identical example (reproducible pipelines)."""
    a = _gen(bank, tokenizer, 123)
    b = _gen(bank, tokenizer, 123)
    assert a == b  # frozen dataclass equality covers tokens, mask, and metadata


def test_picklable_closure_round_trips(bank, tokenizer):
    """The generator + its args survive pickling (needed for remote workers)."""
    payload = (generate_example, bank, tokenizer, GenerationConfig())
    fn, bank2, tok2, cfg2 = pickle.loads(pickle.dumps(payload))

    original = _gen(bank, tokenizer, 7)
    revived = fn(
        CORPUS[0],
        CORPUS,
        bank2,
        tok2,
        max_corruption_steps=3,
        gen_cfg=cfg2,
        rng=random.Random(7),
        max_seq_len=MAX_SEQ_LEN,
    )
    assert revived == original


def test_diffusion_branch_metadata(bank, tokenizer):
    """With p_random=0, examples come from forward diffusion: is_random False,
    corruption_steps within the allowed range."""
    cfg = GenerationConfig(p_random=0.0)
    found = [ex for s in range(20) if (ex := _gen(bank, tokenizer, s, gen_cfg=cfg, max_corruption_steps=3))]
    assert found, "expected at least one valid diffusion example"
    for ex in found:
        assert isinstance(ex, TrainingExample)
        assert ex.is_random is False
        assert 1 <= ex.corruption_steps <= 3


def test_random_branch_metadata(bank, tokenizer):
    """With p_random=1, examples come from a random corpus program: is_random
    True, corruption_steps 0 (no forward-diffusion depth)."""
    cfg = GenerationConfig(p_random=1.0)
    found = [ex for s in range(20) if (ex := _gen(bank, tokenizer, s, gen_cfg=cfg))]
    assert found, "expected at least one valid random-branch example"
    for ex in found:
        assert ex.is_random is True
        assert ex.corruption_steps == 0


def test_p_near_miss_threads_and_generates(bank, tokenizer):
    """The near-miss corruption knob plumbs through the config and still yields
    valid examples (forward-diffusion branch, always near-miss)."""
    cfg = GenerationConfig(p_random=0.0, p_near_miss=1.0)
    ex = _gen(bank, tokenizer, 1, gen_cfg=cfg)
    assert ex is not None and ex.is_random is False


def test_from_training_config_carries_p_near_miss():
    from kelp.training.engine import EditTrainingConfig

    tc = EditTrainingConfig(model=_model_cfg_for_test(), p_near_miss=0.7)
    assert GenerationConfig.from_training_config(tc).p_near_miss == 0.7


def _model_cfg_for_test():
    from kelp.model.config import EditModelConfig

    return EditModelConfig(
        vocab_size=300,
        hidden_dim=32,
        intermediate_dim=64,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=MAX_SEQ_LEN,
    )


def test_spec_conditioning_threads_through_generation(bank):
    """With a spec_tokens tokenizer and p_spec at the boundaries, a doctest'd
    clean program deterministically does/doesn't get a spec block, and the spec
    never carries loss (it is conditioning, not target)."""
    tok = EditTokenizer(max_seq_len=MAX_SEQ_LEN, prompt_tokens=True, spec_tokens=True)
    doctested = 'def add(a, b):\n    """Add.\n\n    >>> add(1, 2)\n    3\n    """\n    return a + b\n'
    corpus = [doctested] + CORPUS[1:]

    def gen(p_spec, seed):
        return generate_example(
            doctested,
            corpus,
            bank,
            tok,
            max_corruption_steps=3,
            gen_cfg=GenerationConfig(p_spec=p_spec, p_random=0.0),
            rng=random.Random(seed),
            max_seq_len=MAX_SEQ_LEN,
        )

    ex = next(e for e in (gen(1.0, s) for s in range(20)) if e is not None)
    assert ex.spec_used
    assert tok.spec_start_token_id in ex.token_ids
    assert ex.loss_mask[ex.token_ids.index(tok.spec_end_token_id)] == 0

    ex_off = next(e for e in (gen(0.0, s) for s in range(20)) if e is not None)
    assert not ex_off.spec_used and tok.spec_start_token_id not in ex_off.token_ids
