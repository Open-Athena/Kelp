# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the distributed-bootstrap gating.

The gate decides whether to call jax.distributed.initialize(): it must fire for
genuinely multi-process runs (multi-task Iris job or supervised multi-GPU) and
stay off for a single process (single-host TPU, laptop, CI). Getting this wrong
is a silent multi-host correctness hazard, so pin the branches here.
"""

import types

import iris.cluster.client.job_info as job_info_mod

import kelp.training.distributed as dist


def test_off_cluster_is_single_process(monkeypatch):
    """No Iris job (get_job_info None) and no multi-GPU env -> single process."""
    monkeypatch.delenv("IRIS_MULTIGPU_PROCESS_COUNT", raising=False)
    monkeypatch.setattr(job_info_mod, "get_job_info", lambda: None)
    assert dist._running_multiprocess() is False


def test_multi_task_job_is_multiprocess(monkeypatch):
    """A job with more than one task requires distributed init."""
    monkeypatch.delenv("IRIS_MULTIGPU_PROCESS_COUNT", raising=False)
    monkeypatch.setattr(job_info_mod, "get_job_info", lambda: types.SimpleNamespace(num_tasks=2))
    assert dist._running_multiprocess() is True


def test_single_task_job_is_single_process(monkeypatch):
    """A single-task job (e.g. a single-host TPU slice) skips distributed init."""
    monkeypatch.delenv("IRIS_MULTIGPU_PROCESS_COUNT", raising=False)
    monkeypatch.setattr(job_info_mod, "get_job_info", lambda: types.SimpleNamespace(num_tasks=1))
    assert dist._running_multiprocess() is False


def test_supervised_multigpu_env_forces_multiprocess(monkeypatch):
    """Supervised multi-GPU mode is detected via its env var, even single-task."""
    monkeypatch.setenv("IRIS_MULTIGPU_PROCESS_COUNT", "4")
    monkeypatch.setattr(job_info_mod, "get_job_info", lambda: types.SimpleNamespace(num_tasks=1))
    assert dist._running_multiprocess() is True
