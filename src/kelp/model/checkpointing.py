# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint save/load for Kelp tree diffusion models.

Backed by Orbax, which stores arrays via TensorStore in the Zarr format. This
replaces the previous pickle format so checkpoints can be written directly to
GCS (``gs://...``) -- required because Iris job containers are ephemeral, so a
local path vanishes when the job ends.

A checkpoint directory contains:
- ``config.json``: the ``EditModelConfig`` (written by process 0 only)
- ``params/``: model parameters, an Orbax/TensorStore (Zarr) tree

Paths may be local or ``gs://`` (handled transparently via ``etils.epath``).
Saving is a multi-host collective: every process calls :func:`save_checkpoint`
(Orbax coordinates which host writes which shard); only process 0 writes the
JSON sidecar.

Note: saving is synchronous here (durable on return). Making it asynchronous --
overlapping the GCS write with subsequent training steps -- needs a checkpointer
that persists across training steps rather than these per-call functions, and is
tracked in issue #7.
"""

import json
import logging
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from etils import epath
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from levanter.grug.attention import RotaryConfig

from kelp.model.config import EditModelConfig
from kelp.model.edit_model import EditModelParams, init_edit_params

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "config.json"
PARAMS_SUBDIR = "params"
# Resumable extras (optimizer state, step, RNG key) live alongside params/ so
# load_checkpoint stays params-only for eval while resume reads both.
TRAIN_STATE_SUBDIR = "train_state"


def _absolute(ckpt_dir: str | os.PathLike) -> epath.Path:
    """Return an absolute path. Orbax rejects relative paths; cloud URIs
    (``gs://``) are already absolute and are left untouched."""
    raw = os.fspath(ckpt_dir)
    if "://" not in raw:
        raw = os.path.abspath(raw)
    return epath.Path(raw)


def _replicated_current_sharding() -> NamedSharding:
    """A replicated sharding over the *current* devices, so a restore reshards
    the saved arrays onto whatever topology we load on (e.g. a checkpoint saved
    across TPU chips loads on a single CPU for eval, or back onto a TPU mesh)."""
    devices = jax.devices()
    mesh = Mesh(np.asarray(devices).reshape(len(devices)), ("dp",))
    return NamedSharding(mesh, PartitionSpec())


def _with_sharding(abstract_tree, sharding: NamedSharding):
    """Attach ``sharding`` to every ShapeDtypeStruct leaf of an abstract pytree."""
    return jax.tree.map(lambda s: jax.ShapeDtypeStruct(s.shape, s.dtype, sharding=sharding), abstract_tree)


def _config_to_json(model_config: EditModelConfig) -> dict:
    """Serialize an EditModelConfig to a JSON-safe dict (RoPE flattened)."""
    from dataclasses import asdict

    config_dict = asdict(model_config)
    if hasattr(model_config.rope, "__dict__"):
        config_dict["rope"] = asdict(model_config.rope)
    return config_dict


def _config_from_json(config_dict: dict) -> EditModelConfig:
    """Reconstruct an EditModelConfig from its JSON dict (rebuilding RoPE)."""
    rope_val = config_dict.pop("rope", None)
    if isinstance(rope_val, dict):
        config_dict["rope"] = RotaryConfig(**rope_val)
    elif rope_val is not None:
        config_dict["rope"] = rope_val
    return EditModelConfig(**config_dict)


def save_checkpoint(params: EditModelParams, model_config: EditModelConfig, ckpt_dir: str | os.PathLike) -> None:
    """Save model parameters and config to a checkpoint directory.

    Args:
        params: Model parameters to save.
        model_config: Model configuration to save.
        ckpt_dir: Destination directory. May be local or ``gs://``. Created if
            absent.
    """
    path = _absolute(ckpt_dir)
    path.mkdir(parents=True, exist_ok=True)

    # JSON sidecar: a single writer avoids concurrent writes on multi-host.
    if jax.process_index() == 0:
        (path / CONFIG_FILENAME).write_text(json.dumps(_config_to_json(model_config), indent=2))

    # Array tree: a collective save across all processes (Orbax handles sharded
    # writes and finalization). force=True so a retried step overwrites cleanly.
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(path / PARAMS_SUBDIR, params, force=True)
    ckptr.wait_until_finished()

    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(ckpt_dir: str | os.PathLike) -> tuple[EditModelParams, EditModelConfig]:
    """Load model parameters and config from a checkpoint directory.

    Args:
        ckpt_dir: Directory containing ``config.json`` and ``params/``. May be
            local or ``gs://``.

    Returns:
        Tuple of (params, config).
    """
    path = _absolute(ckpt_dir)
    config_path = path / CONFIG_FILENAME
    params_path = path / PARAMS_SUBDIR

    if not config_path.exists() or not params_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    config = _config_from_json(json.loads(config_path.read_text()))

    # Restore into the exact EditModelParams structure. eval_shape gives the
    # per-leaf shapes/dtypes (and pytree structure) without allocating, so
    # Orbax rebuilds the registered dataclass tree rather than plain dicts.
    abstract_params = jax.eval_shape(lambda: init_edit_params(config, key=jax.random.PRNGKey(0)))
    abstract_params = _with_sharding(abstract_params, _replicated_current_sharding())

    ckptr = ocp.StandardCheckpointer()
    params = ckptr.restore(params_path, target=abstract_params)

    logger.info(f"Loaded checkpoint from {ckpt_dir} (vocab_size={config.vocab_size})")
    return params, config


def save_training_checkpoint(
    params: EditModelParams,
    opt_state: optax.OptState,
    step: int,
    key: jax.Array,
    model_config: EditModelConfig,
    ckpt_dir: str | os.PathLike,
) -> None:
    """Save a *resumable* checkpoint: model params (also readable by
    :func:`load_checkpoint` for eval) plus the optimizer state, step, and RNG
    key needed to resume training exactly (e.g. after a preemption).

    The params live in ``params/`` (as :func:`save_checkpoint` writes them) and
    the resume extras in ``train_state/``.
    """
    path = _absolute(ckpt_dir)
    save_checkpoint(params, model_config, path)  # config.json + params/

    extras = {"opt_state": opt_state, "step": jnp.asarray(step, dtype=jnp.int32), "key": key}
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(path / TRAIN_STATE_SUBDIR, extras, force=True)
    ckptr.wait_until_finished()


def load_training_checkpoint(
    ckpt_dir: str | os.PathLike,
    optimizer: optax.GradientTransformation,
) -> tuple[EditModelParams, optax.OptState, int, jax.Array, EditModelConfig]:
    """Load a resumable checkpoint into its components.

    Returns ``(params, opt_state, step, key, config)``. The ``optimizer`` is
    needed only to reconstruct the abstract optimizer-state structure for the
    restore (its arrays come from disk). Arrays are resharded onto the current
    topology, so a checkpoint saved across TPU chips resumes on a different
    device count.
    """
    params, config = load_checkpoint(ckpt_dir)
    state_path = _absolute(ckpt_dir) / TRAIN_STATE_SUBDIR
    if not state_path.exists():
        raise FileNotFoundError(f"No resumable train_state/ at {state_path} (was this saved with save_checkpoint?)")

    sharding = _replicated_current_sharding()
    abstract = {
        "opt_state": _with_sharding(jax.eval_shape(lambda: optimizer.init(params)), sharding),
        "step": jax.ShapeDtypeStruct((), jnp.int32, sharding=sharding),
        "key": _with_sharding(jax.eval_shape(lambda: jax.random.PRNGKey(0)), sharding),
    }
    ckptr = ocp.StandardCheckpointer()
    extras = ckptr.restore(state_path, target=abstract)

    logger.info(f"Loaded resumable checkpoint from {ckpt_dir} at step {int(extras['step'])}")
    return params, extras["opt_state"], int(extras["step"]), extras["key"], config


def _is_complete_checkpoint(ckpt_dir: epath.Path) -> bool:
    """True iff the checkpoint finished committing.

    A preemption can kill the job mid-write, leaving a partial ``step-XXXXXX``
    dir on GCS. Without this check, resume deterministically crashes on the
    incomplete latest dir and burns the job's failure-retry budget (exp12,
    2026-08-04: incomplete step-005500 turned 6 survivable preemptions into a
    terminal failure). Delegates to Orbax's own finalization check -- the same
    one whose restore-side counterpart raises 'Found incomplete checkpoint' --
    which understands both atomic-rename (local) and commit-marker (GCS)
    filesystems.
    """
    import orbax.checkpoint as ocp

    params_dir = ckpt_dir / PARAMS_SUBDIR
    return params_dir.exists() and ocp.utils.is_checkpoint_finalized(params_dir)


def find_best_checkpoint(checkpoint_dir: str | os.PathLike) -> epath.Path | None:
    """Find the COMPLETE checkpoint with the highest step number.

    Incomplete checkpoints (no Orbax commit marker; see
    :func:`_is_complete_checkpoint`) are skipped, so a mid-write preemption
    falls back to the previous durable checkpoint instead of crashing resume.

    Args:
        checkpoint_dir: Parent directory containing ``step-XXXXXX`` subdirectories.

    Returns:
        Path to the best complete checkpoint, or None if no checkpoints found.
    """
    root = epath.Path(checkpoint_dir)
    if not root.exists():
        return None
    ckpt_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir() and d.name.startswith("step-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    for d in reversed(ckpt_dirs):
        if _is_complete_checkpoint(d):
            return d
        logger.warning(f"Skipping incomplete checkpoint {d} (no commit marker; interrupted mid-write)")
    return None
