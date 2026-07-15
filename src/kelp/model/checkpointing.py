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

Note: saving is synchronous here (durable on return). Async checkpointing --
overlapping the GCS write with subsequent training steps via a persistent
``CheckpointManager`` -- is the tracked performance follow-up (see chainlink
#116); it needs the checkpointer to live across steps, so it is a training-loop
change rather than a change to these functions.
"""

import json
import logging
import os

import jax
import orbax.checkpoint as ocp
from etils import epath
from levanter.grug.attention import RotaryConfig

from kelp.model.config import EditModelConfig
from kelp.model.edit_model import EditModelParams, init_edit_params

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "config.json"
PARAMS_SUBDIR = "params"


def _absolute(ckpt_dir: str | os.PathLike) -> epath.Path:
    """Return an absolute path. Orbax rejects relative paths; cloud URIs
    (``gs://``) are already absolute and are left untouched."""
    raw = os.fspath(ckpt_dir)
    if "://" not in raw:
        raw = os.path.abspath(raw)
    return epath.Path(raw)


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
    ckptr = ocp.StandardCheckpointer()
    params = ckptr.restore(params_path, target=abstract_params)

    logger.info(f"Loaded checkpoint from {ckpt_dir} (vocab_size={config.vocab_size})")
    return params, config


def find_best_checkpoint(checkpoint_dir: str | os.PathLike) -> epath.Path | None:
    """Find the checkpoint with the highest step number.

    Args:
        checkpoint_dir: Parent directory containing ``step-XXXXXX`` subdirectories.

    Returns:
        Path to the best checkpoint, or None if no checkpoints found.
    """
    root = epath.Path(checkpoint_dir)
    if not root.exists():
        return None
    ckpt_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir() and d.name.startswith("step-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    if not ckpt_dirs:
        return None
    return ckpt_dirs[-1]
