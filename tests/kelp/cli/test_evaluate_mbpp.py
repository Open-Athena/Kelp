# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for MBPP eval helpers."""

from etils import epath

from kelp.cli.evaluate_mbpp import _checkpoint_step, _load_completed, _tasks_dir, _write_task_result


def test_checkpoint_step_parses_step_dir():
    """The W&B repair-vs-step x-axis is parsed from the step-XXXXXX dir name."""
    assert _checkpoint_step(epath.Path("gs://b/run/step-030000")) == 30000
    assert _checkpoint_step(epath.Path("checkpoints/run/step-000002")) == 2
    # Non-standard names yield None (logged without an x coordinate, not crash).
    assert _checkpoint_step(epath.Path("gs://b/run/latest")) is None


def test_tasks_dir_derived_from_output():
    assert _tasks_dir("gs://b/run/step-030000-mbpp.json").name == "step-030000-mbpp-tasks"


def test_task_results_persist_and_reload_for_resume(tmp_path):
    """Per-task results are keyed by task_id and survive a reload -- the basis
    for resuming a preempted eval instead of restarting from task 0."""
    tasks_dir = _tasks_dir(str(tmp_path / "run" / "mbpp.json"))
    assert _load_completed(tasks_dir) == {}  # nothing written yet

    _write_task_result(tasks_dir, {"task_id": 7, "avg_test_pass_rate": 0.5})
    _write_task_result(tasks_dir, {"task_id": 42, "avg_test_pass_rate": 0.1})

    loaded = _load_completed(tasks_dir)
    assert set(loaded) == {7, 42}
    assert loaded[7]["avg_test_pass_rate"] == 0.5
