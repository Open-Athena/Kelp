# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for MBPP eval helpers."""

from etils import epath

from kelp.cli.evaluate_mbpp import _checkpoint_step


def test_checkpoint_step_parses_step_dir():
    """The W&B repair-vs-step x-axis is parsed from the step-XXXXXX dir name."""
    assert _checkpoint_step(epath.Path("gs://b/run/step-030000")) == 30000
    assert _checkpoint_step(epath.Path("checkpoints/run/step-000002")) == 2
    # Non-standard names yield None (logged without an x coordinate, not crash).
    assert _checkpoint_step(epath.Path("gs://b/run/latest")) is None
