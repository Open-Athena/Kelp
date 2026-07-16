# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Distributed bootstrap for multi-host TPU training on Iris.

A single entry point, :func:`bootstrap_distributed`, that brings up JAX's
distributed runtime from Iris job metadata (only when the run is genuinely
multi-process) and reports the resulting topology. It must run before the first
use of ``jax.devices()`` so that the training mesh (see
:mod:`kelp.training.sharding`) is built over the *global* device set.

``jax.distributed.initialize()`` is only needed to stitch together *multiple
host processes*. A single-host slice -- a laptop, CI, or a single-host TPU pod
such as v6e-4 or v5p-8 -- sees all its local chips without it, and calling it
there tries to reach a coordinator address that isn't routable in the job's
network and wedges. So we gate the call on a real multi-process job (more than
one Iris task, or supervised multi-GPU) rather than calling it unconditionally.
"""

import logging
import os

logger = logging.getLogger(__name__)


def _running_multiprocess() -> bool:
    """True when this run spans multiple host processes that JAX must stitch
    together (multi-task Iris job, or supervised multi-GPU). False for a single
    process -- including a single-host multi-chip TPU slice and off-cluster."""
    if any(k.startswith("IRIS_MULTIGPU") for k in os.environ):
        return True
    try:
        from iris.cluster.client.job_info import get_job_info

        job_info = get_job_info()
    except Exception:  # not in an Iris job / iris unavailable
        return False
    return job_info is not None and getattr(job_info, "num_tasks", 1) > 1


def bootstrap_distributed() -> None:
    """Bring up JAX distributed for multi-host runs, then log the topology.

    Multi-process (multi-task Iris slice / supervised multi-GPU): performs the
    coordinator handshake via Iris endpoint discovery. Single process
    (single-host TPU, laptop, CI): skips ``jax.distributed.initialize()`` --
    JAX uses the local devices directly. Safe to call once at process start,
    before any device use.
    """
    if _running_multiprocess():
        from levanter.distributed import initialize_iris_jax

        initialize_iris_jax()
    else:
        logger.info("Single-process run; using local devices (no jax.distributed init)")

    import jax

    logger.info(
        "JAX: process %d/%d, %d local device(s), %d global device(s)",
        jax.process_index(),
        jax.process_count(),
        jax.local_device_count(),
        jax.device_count(),
    )
