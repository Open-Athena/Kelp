# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Uncertainty estimates for eval aggregates (issue #142).

Every headline eval number is a mean over evaluated tasks, and at n~50 tasks
the task-level variance dominates: deltas of a few points are routinely inside
the noise. A percentile bootstrap over tasks makes that noise visible next to
every reported mean, so conclusions can be checked against their own error bars
instead of eyeballed.
"""

import random


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile-bootstrap confidence interval for the mean of ``values``.

    Resamples tasks (the unit of independence in our evals) with replacement.
    Deterministic for a given ``seed`` so re-running an eval reproduces its
    reported interval exactly. Returns ``(lo, hi)`` at the ``1 - alpha`` level;
    degenerate inputs (empty or single-element) return a zero-width interval.
    """
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])

    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_resamples))
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)
    return (means[lo_idx], means[hi_idx])
