# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Kelp command-line entry points.

Each module exposes a ``main()`` runnable via ``python -m kelp.cli.<name>`` or
the console scripts declared in pyproject (``kelp-train``, ``kelp-evaluate``,
``kelp-prepare-corpus``, ...). These are thin argparse wrappers over the
library; no application logic lives here.
"""
