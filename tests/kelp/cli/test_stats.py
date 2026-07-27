# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the eval bootstrap CI helper."""

from kelp.cli._stats import bootstrap_ci


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    """Same seed -> same interval (re-running an eval reproduces its report),
    and the interval brackets the sample mean for a mixed 0/1 sample."""
    values = [0.0] * 40 + [1.0] * 10  # solved_rate-like sample, mean 0.2
    ci = bootstrap_ci(values, seed=42)
    assert ci == bootstrap_ci(values, seed=42)
    assert ci[0] < 0.2 < ci[1]
    # At n=50 the interval is wide -- the whole point of reporting it: a few-pp
    # delta between two runs cannot clear error bars this size.
    assert ci[1] - ci[0] > 0.1


def test_bootstrap_ci_degenerate_inputs():
    """Empty and constant samples yield zero-width intervals, not crashes."""
    assert bootstrap_ci([]) == (0.0, 0.0)
    assert bootstrap_ci([0.7]) == (0.7, 0.7)
    lo, hi = bootstrap_ci([0.5] * 20)
    assert lo == hi == 0.5
