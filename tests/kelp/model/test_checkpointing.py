# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Orbax-backed checkpoint save/load."""

import jax
import jax.numpy as jnp

from kelp.model.checkpointing import find_best_checkpoint, load_checkpoint, save_checkpoint
from kelp.model.config import EditModelConfig
from kelp.model.edit_model import init_edit_params


def _cfg():
    return EditModelConfig(
        vocab_size=300,
        hidden_dim=32,
        intermediate_dim=64,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=64,
    )


def test_save_load_round_trip(tmp_path):
    """Params and config survive a save/load cycle unchanged and typed."""
    cfg = _cfg()
    params = init_edit_params(cfg, key=jax.random.PRNGKey(0))

    ckpt_dir = tmp_path / "step-000001"
    save_checkpoint(params, cfg, ckpt_dir)
    loaded_params, loaded_cfg = load_checkpoint(ckpt_dir)

    # Config restored faithfully (including the nested RoPE dataclass).
    assert loaded_cfg == cfg

    # Every parameter array is bit-identical and the dataclass tree is rebuilt
    # (not a plain dict), so attribute access still works.
    orig_leaves = jax.tree.leaves(params)
    new_leaves = jax.tree.leaves(loaded_params)
    assert len(orig_leaves) == len(new_leaves)
    for a, b in zip(orig_leaves, new_leaves, strict=True):
        assert jnp.array_equal(a, b)
    assert loaded_params.token_embed.shape == (cfg.vocab_size, cfg.hidden_dim)
    assert len(loaded_params.blocks) == cfg.num_layers


def test_find_best_checkpoint_picks_highest_step(tmp_path):
    cfg = _cfg()
    params = init_edit_params(cfg, key=jax.random.PRNGKey(0))
    for step in (1, 50, 7):
        save_checkpoint(params, cfg, tmp_path / f"step-{step:06d}")

    best = find_best_checkpoint(tmp_path)
    assert best is not None
    assert best.name == "step-000050"


def test_find_best_checkpoint_empty(tmp_path):
    assert find_best_checkpoint(tmp_path) is None
    assert find_best_checkpoint(tmp_path / "does-not-exist") is None


def test_find_best_checkpoint_skips_incomplete(tmp_path):
    """A mid-write preemption leaves a step dir without Orbax's commit marker;
    resume must fall back to the newest COMPLETE checkpoint instead of
    crashing on the partial one (the failure that killed exp12 at step 5500)."""
    from kelp.model.checkpointing import PARAMS_SUBDIR, find_best_checkpoint

    complete = tmp_path / "step-000500" / PARAMS_SUBDIR
    complete.mkdir(parents=True)
    # Interrupted local write: params still tmp-named (never renamed into place).
    partial = tmp_path / "step-001000" / f"{PARAMS_SUBDIR}.orbax-checkpoint-tmp-99"
    partial.mkdir(parents=True)

    best = find_best_checkpoint(tmp_path)
    assert best is not None and best.name == "step-000500"

    partial.rename(tmp_path / "step-001000" / PARAMS_SUBDIR)  # commit -> newest wins
    assert find_best_checkpoint(tmp_path).name == "step-001000"

    import shutil

    shutil.rmtree(tmp_path / "step-000500")
    shutil.rmtree(tmp_path / "step-001000")
    assert find_best_checkpoint(tmp_path) is None
