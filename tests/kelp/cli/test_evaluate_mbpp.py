# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for MBPP eval helpers and the shared resume store."""

from etils import epath

from kelp.cli._eval_resume import eval_fingerprint, load_completed, shard_dir, write_result
from kelp.cli.evaluate_mbpp import _checkpoint_step, run_mbpp_test


def test_run_mbpp_test_times_out_nonterminating_candidate():
    """A non-terminating candidate must fail the test within the wall-clock limit,
    not hang the eval (the vet-cond-v2 failure that lost 17/50 tasks). Uses a
    short timeout so the test itself stays fast."""
    infinite = "def f():\n    while True:\n        pass"
    assert run_mbpp_test(infinite, "assert f() == 1", timeout_s=0.3) is False


def test_run_mbpp_test_passes_correct_program():
    """A correct, fast program still passes (the timeout doesn't false-fail it)."""
    assert run_mbpp_test("def add(a, b):\n    return a + b", "assert add(2, 3) == 5", timeout_s=5.0) is True


def test_checkpoint_step_parses_step_dir():
    """The W&B repair-vs-step x-axis is parsed from the step-XXXXXX dir name."""
    assert _checkpoint_step(epath.Path("gs://b/run/step-030000")) == 30000
    assert _checkpoint_step(epath.Path("checkpoints/run/step-000002")) == 2
    # Non-standard names yield None (logged without an x coordinate, not crash).
    assert _checkpoint_step(epath.Path("gs://b/run/latest")) is None


def test_shard_dir_derived_from_output_and_fingerprint():
    d = shard_dir("gs://b/run/step-030000-mbpp.json", "abc123")
    assert d.name == "step-030000-mbpp-tasks-abc123"


def test_fingerprint_changes_with_config():
    """A different eval config yields a different shard dir, so a re-run with
    new parameters doesn't silently reuse stale per-task shards."""
    base = {"n_best_of": 16, "seed": 42}
    assert eval_fingerprint(base) == eval_fingerprint(dict(base))  # stable
    assert eval_fingerprint(base) != eval_fingerprint({**base, "n_best_of": 8})
    # Matched (0.85) and unmatched (0.0) evals must land in separate shards.
    assert eval_fingerprint({**base, "p_near_miss": 0.85}) != eval_fingerprint({**base, "p_near_miss": 0.0})


def test_results_persist_and_reload_for_resume(tmp_path):
    """Per-item results are keyed by id and survive a reload -- the basis for
    resuming a preempted eval instead of restarting from item 0."""
    shards = shard_dir(str(tmp_path / "run" / "mbpp.json"), "fp")
    assert load_completed(shards, "task_id") == {}  # nothing written yet

    shards.mkdir(parents=True, exist_ok=True)
    write_result(shards, {"task_id": 7, "avg_test_pass_rate": 0.5}, "task_id")
    write_result(shards, {"task_id": 42, "avg_test_pass_rate": 0.1}, "task_id")

    loaded = load_completed(shards, "task_id")
    assert set(loaded) == {7, 42}
    assert loaded[7]["avg_test_pass_rate"] == 0.5
