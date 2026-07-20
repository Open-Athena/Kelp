# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for MFU / throughput accounting."""

import jax.numpy as jnp

from kelp.training.throughput import (
    ThroughputTracker,
    count_params,
    device_peak_flops,
)


def test_count_params_sums_leaf_sizes():
    """count_params totals every scalar across an arbitrary pytree."""
    tree = {"a": jnp.zeros((2, 3)), "b": [jnp.zeros((4,)), jnp.zeros(())]}
    assert count_params(tree) == 2 * 3 + 4 + 1


def test_metrics_compute_mfu_from_known_quantities():
    """MFU = achieved FLOP/s / (chips * peak). Hand-computed on round numbers:
    2 steps in 1s of a 100-param model at 10 tokens/step over 2 chips whose peak
    is 6000 FLOP/s each -> 6*100*10*2 = 12000 FLOP/s -> MFU = 12000/(2*6000) = 1.0."""
    t = ThroughputTracker(param_count=100, tokens_per_step=10, num_devices=2, peak_flops_per_device=6000.0)
    m = t.metrics(elapsed_s=1.0, steps=2)
    assert m["perf/steps_per_s"] == 2.0
    assert m["perf/tokens_per_s"] == 20.0
    assert m["perf/mfu"] == 1.0
    # TFLOP/s/chip = 12000 / 2 chips / 1e12.
    assert m["perf/tflops_per_device"] == 12000 / 2 / 1e12


def test_metrics_omit_mfu_when_peak_unknown():
    """Unknown hardware (peak None) still reports throughput but no MFU denominator."""
    t = ThroughputTracker(param_count=100, tokens_per_step=10, num_devices=1, peak_flops_per_device=None)
    m = t.metrics(elapsed_s=2.0, steps=4)
    assert "perf/mfu" not in m
    assert m["perf/steps_per_s"] == 2.0


def test_metrics_empty_for_nonpositive_interval():
    """A zero/negative elapsed or step count yields nothing (no divide-by-zero, no
    compile-contaminated first reading)."""
    t = ThroughputTracker(param_count=1, tokens_per_step=1, num_devices=1, peak_flops_per_device=1.0)
    assert t.metrics(0.0, 5) == {}
    assert t.metrics(1.0, 0) == {}


def test_device_peak_flops_matches_specific_before_generic():
    """'TPU v5 lite' (v5e) must resolve to the v5e peak, not the bare-'v5' (v5p)
    peak -- the substring table is ordered most-specific-first."""
    assert device_peak_flops("TPU v5 lite") == 197e12
    assert device_peak_flops("TPU v5p") == 459e12
    assert device_peak_flops("TPU v6e") == 918e12
    assert device_peak_flops("cpu") is None
