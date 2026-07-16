# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for resumable checkpointing (params + optimizer + step + rng).

The point of a resumable checkpoint (vs a params-only one) is that a preempted
run can continue exactly: the optimizer state, step counter, and RNG key must
all survive, and the training loop must pick up from the saved step.
"""

import jax
import jax.numpy as jnp

from kelp.model.checkpointing import (
    find_best_checkpoint,
    load_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)
from kelp.model.config import EditModelConfig
from kelp.model.edit_model import init_edit_params
from kelp.training.engine import (
    EditTrainingConfig,
    create_edit_data_iter,
    create_edit_optimizer,
    load_resume_state,
    train_edit_model,
)
from kelp.tree.subtree_bank import SubtreeBank
from kelp.tree.tokenizer import EditTokenizer

CORPUS = [
    "def add(a, b):\n    return a + b\n",
    "def sub(a, b):\n    return a - b\n",
    "def mul(a, b):\n    return a * b\n",
    "def square(x):\n    return x * x\n",
]
MAX_SEQ_LEN = 128


def _model_cfg(vocab):
    return EditModelConfig(
        vocab_size=vocab,
        hidden_dim=32,
        intermediate_dim=64,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=MAX_SEQ_LEN,
    )


def test_training_checkpoint_round_trips_full_state(tmp_path):
    """params, optimizer state, step, and rng all survive a save/load cycle;
    the same directory still loads params-only for eval."""
    tok = EditTokenizer(max_seq_len=MAX_SEQ_LEN)
    model = _model_cfg(tok.vocab_size)
    train_cfg = EditTrainingConfig(model=model, max_seq_len=MAX_SEQ_LEN, total_steps=10, wandb_project=None)

    params = init_edit_params(model, key=jax.random.PRNGKey(0))
    optimizer = create_edit_optimizer(train_cfg)
    # Take one real optimizer step so the state is non-trivial (non-zero moments).
    grads = jax.tree.map(lambda x: jnp.ones_like(x), params)
    updates, opt_state = optimizer.update(grads, optimizer.init(params), params)
    step, key = 123, jax.random.PRNGKey(7)

    ckpt = tmp_path / "step-000123"
    save_training_checkpoint(params, opt_state, step, key, model, ckpt)
    p2, opt2, step2, key2, cfg2 = load_training_checkpoint(ckpt, optimizer)

    assert step2 == 123
    assert cfg2 == model
    assert jnp.array_equal(key2, key)
    for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(p2), strict=True):
        assert jnp.array_equal(a, b)
    for a, b in zip(jax.tree.leaves(opt_state), jax.tree.leaves(opt2), strict=True):
        assert jnp.array_equal(a, b)

    # Eval path (params only, no optimizer needed) still works on the same dir.
    p3, _ = load_checkpoint(ckpt)
    assert jnp.array_equal(p3.token_embed, params.token_embed)


def test_train_resumes_from_latest_checkpoint(tmp_path):
    """Training a run to a checkpoint, then resuming, continues from the saved
    step rather than restarting at 0."""
    tok = EditTokenizer(max_seq_len=MAX_SEQ_LEN)
    model = _model_cfg(tok.vocab_size)
    bank = SubtreeBank.from_corpus(CORPUS)

    def cfg(total):
        return EditTrainingConfig(
            model=model,
            max_seq_len=MAX_SEQ_LEN,
            total_steps=total,
            batch_size=2,
            warmup_steps=1,
            log_interval=1,
            checkpoint_interval=2,
            output_dir=str(tmp_path),
            wandb_project=None,
        )

    def data():
        return create_edit_data_iter(corpus=CORPUS, bank=bank, tokenizer=tok, config=cfg(2))

    # Phase 1: train 2 steps -> checkpoint at step-000002.
    train_edit_model(config=cfg(2), data_iter=data())
    latest = find_best_checkpoint(tmp_path)
    assert latest is not None and latest.name == "step-000002"

    # Phase 2: resume with a higher step budget; it must start at step 2.
    resume_cfg = cfg(4)
    initial_state = load_resume_state(latest, resume_cfg)
    assert int(initial_state.step) == 2

    logged_steps: list[int] = []
    train_edit_model(
        config=resume_cfg,
        data_iter=create_edit_data_iter(corpus=CORPUS, bank=bank, tokenizer=tok, config=resume_cfg, start_step=2),
        initial_state=initial_state,
        log_callback=lambda s, _m: logged_steps.append(s),
    )
    # The resumed run covers steps 2 and 3 (not 0,1).
    assert min(logged_steps) == 2
    assert max(logged_steps) == 3
