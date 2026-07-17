# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Iris launch entrypoint (build side; no live cluster)."""

from kelp.cli.launch import build_job_request, format_dry_run
from kelp.training.presets import get_preset


def test_job_request_carries_preset_resources_and_command():
    """The request runs kelp-train under the preset and inherits its TPU slice."""
    req = build_job_request(
        "tpu_v5p_8",
        ["--steps", "50000", "--wandb-project", "kelp"],
        name="kelp-test",
        environ={},
    )
    # Resources come straight from the preset (v5p-8 TPU).
    assert req.resources == get_preset("tpu_v5p_8").resource
    assert req.resources.device.variant == "v5p-8"

    # Entrypoint invokes the training module with the same preset + passthrough.
    be = req.entrypoint.binary_entrypoint
    assert be.command == "python"
    assert be.args == ["-m", "kelp.cli.train", "--preset", "tpu_v5p_8", "--steps", "50000", "--wandb-project", "kelp"]


def test_job_request_sets_preemption_retry_budget():
    """Preemptible slices get reclaimed mid-run; the request must allow Iris to
    restart the task (entrypoints resume from checkpoint/shards) rather than stop."""
    req = build_job_request("tpu_vet", [], name="kelp-test", environ={})
    assert req.max_retries_preemption > 0
    assert req.max_retries_failure > 0


def test_eval_request_runs_eval_module_without_preset_flag():
    """An eval launch runs the eval module on the preset's slice but does NOT
    inject --preset (the model config comes from the checkpoint)."""
    req = build_job_request(
        "tpu_vet",
        ["--checkpoint-dir", "gs://b/ckpts", "--n-best-of", "16"],
        name="kelp-eval-test",
        module="kelp.cli.evaluate_mbpp",
        inject_preset=False,
        environ={},
    )
    assert req.resources.device.variant == "v6e-4"
    be = req.entrypoint.binary_entrypoint
    assert be.args == ["-m", "kelp.cli.evaluate_mbpp", "--checkpoint-dir", "gs://b/ckpts", "--n-best-of", "16"]
    assert "--preset" not in be.args


def test_env_passthrough_reads_named_vars_only():
    """Named env vars present in the environment are forwarded; others are not."""
    req = build_job_request(
        "tpu_v5p_8",
        [],
        name="kelp-test",
        env_names=("WANDB_API_KEY", "HF_TOKEN"),
        environ={"WANDB_API_KEY": "secret", "IRRELEVANT": "x"},
    )
    assert req.environment.env_vars.get("WANDB_API_KEY") == "secret"
    assert "IRRELEVANT" not in req.environment.env_vars


def test_defaults_to_local_workspace_without_image():
    """Absent an --image, the launch syncs the current checkout (workspace set)."""
    req = build_job_request("tpu_v5p_8", [], name="kelp-test", env_names=(), environ={})
    assert req.environment.docker_image is None
    assert req.environment.workspace is not None


def test_tpu_preset_requests_tpu_extra():
    """TPU jobs must install JAX's TPU backend, requested via the `tpu` extra."""
    req = build_job_request("tpu_smoke", [], name="kelp-test", environ={})
    assert "tpu" in req.environment.extras


def test_cpu_preset_omits_tpu_extra():
    """Non-TPU presets do not drag in the linux-only TPU extra."""
    req = build_job_request("toy", [], name="kelp-test", environ={})
    assert "tpu" not in req.environment.extras


def test_image_when_provided():
    """An explicit image is used instead of a workspace sync."""
    req = build_job_request("tpu_v5p_8", [], name="kelp-test", image="gcr.io/x/kelp:latest", environ={})
    assert req.environment.docker_image == "gcr.io/x/kelp:latest"
    assert req.environment.workspace is None


def test_dry_run_summary_mentions_slice_and_command():
    req = build_job_request("tpu_v5p_8", ["--steps", "10"], name="kelp-test", environ={})
    summary = format_dry_run(req, "tpu_v5p_8")
    assert "not submitted" in summary
    assert "tpu:v5p-8" in summary
    assert "kelp.cli.train --preset tpu_v5p_8 --steps 10" in summary
