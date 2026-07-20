# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Iris launch entrypoint (build side; no live cluster)."""

import pytest

from kelp.cli.launch import build_job_request, format_dry_run, gcs_region_from_args
from kelp.training.presets import get_preset


def test_gcs_region_from_args_parses_bucket_region():
    """The region is read from the --output-dir / --checkpoint-dir bucket name;
    None when there is no gs:// output arg or no region token."""
    assert gcs_region_from_args(["--output-dir", "gs://marin-us-east5/kelp/ck"]) == "us-east5"
    assert gcs_region_from_args(["--checkpoint-dir", "gs://marin-europe-west4/x"]) == "europe-west4"
    assert gcs_region_from_args(["--output-dir=gs://marin-us-central2/y"]) == "us-central2"
    assert gcs_region_from_args(["--output-dir", "gs://acme-asia-southeast1-ckpts/z"]) == "asia-southeast1"
    assert gcs_region_from_args(["--steps", "50000"]) is None  # no gs:// output arg
    assert gcs_region_from_args(["--output-dir", "gs://plain-bucket/z"]) is None  # no region token


def test_gcs_region_rejects_non_region_tokens():
    """A hyphenated token that is not a real GCP region (wrong direction segment)
    must NOT be mistaken for one -- else the job pins to a nonexistent region and
    never schedules."""
    assert gcs_region_from_args(["--output-dir", "gs://marin-us-team1-data/x"]) is None
    assert gcs_region_from_args(["--output-dir", "gs://marin-me-data1/x"]) is None


def test_pins_tpu_region_from_output_bucket():
    """A TPU job checkpointing to a regioned bucket is pinned to that region, so
    checkpoint writes stay in-region (no cross-region egress)."""
    req = build_job_request(
        "tpu_vet",
        ["--output-dir", "gs://marin-us-east5/kelp/checkpoints/x", "--steps", "1"],
        name="kelp-test",
        environ={},
    )
    assert req.resources.regions == ["us-east5"]


def test_explicit_region_overrides_inferred():
    """An explicit --region wins over the bucket-inferred region."""
    req = build_job_request(
        "tpu_vet",
        ["--output-dir", "gs://marin-us-east5/kelp/x"],
        name="kelp-test",
        region="us-central2",
        environ={},
    )
    assert req.resources.regions == ["us-central2"]


def test_no_region_pin_without_region_token():
    """No region inferable and none given -> placement left unconstrained (the
    preset's resource is untouched)."""
    req = build_job_request("tpu_vet", ["--output-dir", "gs://plain/x"], name="kelp-test", environ={})
    assert req.resources.regions is None


def test_defaults_to_batch_priority_band():
    """Kelp jobs are long, preemptible and checkpointed, so they submit in the
    BATCH band (Iris priority_band 3) by default -- yielding capacity to
    interactive/production work rather than competing with it."""
    req = build_job_request("tpu_vet", ["--output-dir", "gs://marin-us-east5/kelp/x"], name="k", environ={})
    assert req.priority == 3


def test_priority_band_override_and_validation():
    """An explicit band overrides the batch default; an unknown band is rejected
    up front (rather than submitting a job with a bogus priority)."""
    req = build_job_request("tpu_vet", [], name="k", priority_band="interactive", environ={})
    assert req.priority == 2
    with pytest.raises(ValueError, match="Unknown priority_band"):
        build_job_request("tpu_vet", [], name="k", priority_band="urgent", environ={})


def test_dry_run_summary_shows_priority_band():
    req = build_job_request("tpu_vet", [], name="k", environ={})
    assert "priority    : batch (band 3)" in format_dry_run(req, "tpu_vet")


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
