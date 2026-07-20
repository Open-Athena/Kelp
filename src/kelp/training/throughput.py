# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Throughput and Model-FLOPs-Utilization (MFU) accounting for training.

MFU is the fraction of the accelerator's peak FLOP/s that the training step
actually delivers -- the single best scalar for "are we leaving compute on the
table?". W&B captures HBM use but not MXU utilization, so we compute MFU here
from wall-clock step time and a hardware peak table.

The estimate uses the standard dense-transformer approximation
``FLOPs/token ~= 6 * N`` (fwd + bwd, N = parameter count), i.e.
``FLOPs/step = 6 * N * tokens_per_step``. It ignores the attention score
matmuls (O(seq^2), a few percent at these dims) and the optimizer, so it is a
slight *under*-count of true work -- read the number as a floor, and compare
against itself run-over-run rather than as an absolute.

MFU is reported against the hardware's **bf16** peak (the conventional
denominator), regardless of the compute dtype: a float32 run simply shows a
lower MFU because fp32 matmuls run at a fraction of bf16 throughput on the MXU.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax

# Peak *bf16* FLOP/s per chip, by a substring of ``jax.Device.device_kind``.
# These are vendor dense (non-sparsity) peaks; they anchor MFU and are not exact
# for every SKU. Ordered most-specific-first so "v5 lite" wins over "v5".
_PEAK_BF16_FLOPS: tuple[tuple[str, float], ...] = (
    ("v6e", 918e12),
    ("v6 lite", 918e12),  # how v6e reports on some runtimes
    ("v5 lite", 197e12),  # v5e
    ("v5litepod", 197e12),
    ("v5p", 459e12),
    ("v5", 459e12),  # bare "TPU v5" == v5p
    ("v4", 275e12),
    ("a100", 312e12),
    ("h100", 989e12),
)


def count_params(params) -> int:
    """Total number of scalar parameters in a params pytree."""
    return int(sum(leaf.size for leaf in jax.tree_util.tree_leaves(params)))


def device_peak_flops(device_kind: str) -> float | None:
    """Peak bf16 FLOP/s for one chip of ``device_kind``, or None if unknown.

    ``device_kind`` is ``jax.Device.device_kind`` (e.g. ``"TPU v6e"``,
    ``"TPU v5 lite"``, ``"NVIDIA A100-SXM4-40GB"``). Matching is case-insensitive
    substring so we tolerate the vendor's naming drift; unknown hardware returns
    None and callers report throughput without an MFU denominator.
    """
    kind = device_kind.lower()
    for needle, peak in _PEAK_BF16_FLOPS:
        if needle in kind:
            return peak
    return None


@dataclass(frozen=True)
class ThroughputTracker:
    """Turns (elapsed wall-clock, steps completed) into throughput + MFU.

    Pure and stateless: the caller owns the clock and passes deltas to
    :meth:`metrics`, which keeps it trivially testable (no wall-clock in tests)
    and keeps compilation time out of the numbers -- baseline the clock *after*
    the first step so the first measured interval is compile-free.
    """

    param_count: int
    """Total model parameters (N in 6*N*tokens)."""

    tokens_per_step: int
    """Global tokens processed per optimizer step (batch_size * seq_len)."""

    num_devices: int
    """Accelerator chips doing the work (the MFU denominator scales with this)."""

    peak_flops_per_device: float | None
    """Per-chip peak bf16 FLOP/s, or None when the hardware is unrecognized."""

    def flops_per_step(self) -> int:
        """Approximate forward+backward FLOPs for one step (6*N*tokens)."""
        return 6 * self.param_count * self.tokens_per_step

    def metrics(self, elapsed_s: float, steps: int) -> dict[str, float]:
        """Throughput metrics for ``steps`` optimizer steps taken in ``elapsed_s``.

        Returns ``perf/steps_per_s``, ``perf/tokens_per_s``,
        ``perf/tflops_per_device`` and (when the hardware peak is known)
        ``perf/mfu`` in [0, 1]. Returns an empty dict for a non-positive
        interval (nothing meaningful to report yet).
        """
        if elapsed_s <= 0 or steps <= 0:
            return {}

        steps_per_s = steps / elapsed_s
        tokens_per_s = self.tokens_per_step * steps_per_s
        flops_per_s = self.flops_per_step() * steps_per_s
        tflops_per_device = flops_per_s / max(self.num_devices, 1) / 1e12

        out = {
            "perf/steps_per_s": steps_per_s,
            "perf/tokens_per_s": tokens_per_s,
            "perf/tflops_per_device": tflops_per_device,
        }
        if self.peak_flops_per_device:
            out["perf/mfu"] = flops_per_s / (self.num_devices * self.peak_flops_per_device)
        return out
