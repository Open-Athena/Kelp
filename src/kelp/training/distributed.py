# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Distributed bootstrap for multi-host TPU training on Iris.

A single entry point, :func:`bootstrap_distributed`, that brings up JAX's
distributed runtime from Iris job metadata and reports the resulting topology.
It must run before the first use of ``jax.devices()`` so that the training mesh
(see :mod:`kelp.training.sharding`) is built over the *global* device set.

Outside an Iris job -- laptop, CI, a single-host run -- this is a no-op:
``initialize_iris_jax`` detects the single-process case and skips
``jax.distributed.initialize()``, so the same call is safe everywhere.
"""

import logging

logger = logging.getLogger(__name__)


def bootstrap_distributed() -> None:
    """Initialize JAX distributed from Iris (no-op off-cluster) and log topology.

    Idempotent: safe to call once at process start. On a multi-host Iris slice
    it performs the coordinator handshake via Iris endpoint discovery; on a
    single process it returns immediately.
    """
    from levanter.distributed import initialize_iris_jax

    initialize_iris_jax()

    import jax

    logger.info(
        "JAX distributed: process %d/%d, %d local device(s), %d global device(s)",
        jax.process_index(),
        jax.process_count(),
        jax.local_device_count(),
        jax.device_count(),
    )
