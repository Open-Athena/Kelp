# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Data-parallel sharding helpers for training on multi-chip accelerators.

A bare ``jax.jit(train_step)`` runs on a single device, so on a TPU host with
N chips it uses one chip. These helpers build a 1-D device mesh and place the
batch (sharded along the batch axis) and the training state (replicated) so the
JITted step runs SPMD data-parallel across every chip -- with gradient
all-reduce inserted automatically by GSPMD.

Backward compatible: on a single device the mesh has size 1, so ``replicate``
and ``shard_batch`` are effectively no-ops and numerics are unchanged.

Only data parallelism is implemented (parameters are replicated on every chip).
Sharding parameters across chips (FSDP) for models that outgrow one chip's
memory is tracked in issue #6.
"""

import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

DATA_AXIS = "data"


def make_data_parallel_mesh(devices: list[jax.Device] | None = None) -> Mesh:
    """Build a 1-D device mesh with a single ``data`` axis over all devices.

    Args:
        devices: Devices to include. Defaults to ``jax.devices()`` (all local
            devices; on a multi-host job this is the global device set once
            ``jax.distributed.initialize()`` has run).

    Returns:
        A ``Mesh`` whose only axis, ``data``, spans every device.
    """
    devices = devices if devices is not None else jax.devices()
    mesh_devices = mesh_utils.create_device_mesh((len(devices),), devices=devices)
    return Mesh(mesh_devices, axis_names=(DATA_AXIS,))


def data_parallel_size(mesh: Mesh) -> int:
    """Number of data-parallel shards (the size of the ``data`` axis)."""
    return int(mesh.shape[DATA_AXIS])


def replicate[T](tree: T, mesh: Mesh) -> T:
    """Place every array in ``tree`` replicated on all devices of ``mesh``."""
    return jax.device_put(tree, NamedSharding(mesh, P()))


def shard_batch[T](batch: T, mesh: Mesh) -> T:
    """Shard each array in ``batch`` along axis 0 (the batch axis) over ``data``.

    Non-batch axes are replicated. The batch dimension must be divisible by
    ``data_parallel_size(mesh)``; callers are expected to validate this once
    up front for a clearer error than the device_put would give.
    """
    return jax.device_put(batch, NamedSharding(mesh, P(DATA_AXIS)))
