# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared CLI logging setup that stays observable under Iris.

JAX (via absl) registers a handler on the root logger when it is imported, which
makes a subsequent ``logging.basicConfig()`` a no-op -- so our INFO logs get
routed to absl's handler and never reach the stdout stream. Iris captures a
task's stdout/print but not absl's stream, so kelp-train's logs (bootstrap
topology, device mesh, per-step metrics) are then invisible in ``iris job
logs``.

:func:`configure_logging` uses ``force=True`` to remove any pre-existing root
handler and install a plain stdout handler, so those logs are captured. Call it
at the top of ``main()`` (after imports, so it wins over absl).
"""

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """Force a stdout root log handler that wins over absl/JAX's."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
