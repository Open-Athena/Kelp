# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch a Kelp training run on Marin compute (Iris) from a preset.

Each model preset carries a ``fray.cluster.ResourceConfig`` describing the
accelerator it targets (e.g. ``tpu_v5p_8`` -> ``ResourceConfig.with_tpu('v5p-8')``).
This launcher turns a preset + training arguments into a fray ``JobRequest`` and
submits it via the current fray client, which converts it to an Iris job and
places it on a TPU slice. Kelp code never touches Iris types directly --
``fray.iris_backend`` performs the fray -> Iris translation.

Inside the job the entrypoint runs ``python -m kelp.cli.train --preset <preset>
<passthrough args>``; that process initializes JAX distributed from the Iris job
(a no-op on a single host) and shards the batch across the slice's chips.

Usage (dry-run is the default; nothing is submitted until ``--submit``)::

    # See exactly what would be launched:
    kelp-launch --preset tpu_v5p_8 -- --steps 50000 --wandb-project kelp

    # Actually submit to the cluster:
    kelp-launch --preset tpu_v5p_8 --submit -- --steps 50000 --wandb-project kelp

Prerequisites on the cluster:
- A worker image containing kelp + its deps, incl. gcsfs for gs:// checkpoints
  (build/push with ``iris build worker-image``); pass it via ``--image``.
- Auth: for headless/CI use ADC service-account impersonation
  (``gcloud auth application-default login --impersonate-service-account=...``);
  interactive users run ``iris login``. See lib/iris/OPS.md.
- Env vars are NOT copied from your shell into the job -- name the ones the run
  needs with ``--env`` (defaults cover W&B / HF tokens) and they are read from
  your current environment at launch time.
"""

import argparse
import dataclasses
import logging
import os
import re
from pathlib import Path

from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment

from kelp.cli._logging import configure_logging
from kelp.training.presets import PRESETS, get_preset

logger = logging.getLogger(__name__)

DEFAULT_ENV_PASSTHROUGH = ("WANDB_API_KEY", "WANDB_ENTITY", "HF_TOKEN")

# A GCP region token as it appears in a bucket name (us-east5, us-central2,
# europe-west4, asia-southeast1, ...). The middle segment must be a real GCP
# cardinal direction, so an arbitrary hyphenated token (e.g. a bucket named
# `...-us-team1-...`) is NOT mistaken for a region.
_GCP_AREAS = "us|europe|asia|northamerica|southamerica|australia|me|africa"
_GCP_DIRECTIONS = "central|east|west|north|south|northeast|northwest|southeast|southwest"
_GCP_REGION_RE = re.compile(rf"(?:{_GCP_AREAS})-(?:{_GCP_DIRECTIONS})\d+")

# GCS output/checkpoint flags whose bucket determines where the job should run.
_GCS_OUTPUT_FLAGS = ("--output-dir", "--checkpoint-dir")

# Iris scheduling priority bands, mirroring iris.rpc.job_pb2.PriorityBand wire
# values. Kept as plain ints so this module needs no iris import at load time
# (iris is imported lazily only at submit; see _submit_or_dry_run) -- fray
# forwards JobRequest.priority straight through as Iris's priority_band. Lower
# number = higher priority: PRODUCTION(1) beats INTERACTIVE(2) beats BATCH(3),
# and preemption can only evict strictly-lower-priority work.
#
# We default to BATCH: kelp training/eval jobs are long, non-interactive, already
# preemptible (max_retries_preemption below) and resume from checkpoints, so they
# should yield capacity to interactive/production work rather than compete with
# it. BATCH also avoids the budget-cliff churn where an over-budget INTERACTIVE
# job is downgraded to BATCH mid-run and can oscillate back (Iris
# compute_effective_band). Override with --priority-band interactive.
_PRIORITY_BANDS = {"production": 1, "interactive": 2, "batch": 3}
_DEFAULT_PRIORITY_BAND = "batch"


def _band_name(band: int) -> str:
    """Human-readable name for a priority-band int (for the dry-run summary)."""
    for name, value in _PRIORITY_BANDS.items():
        if value == band:
            return name
    return "unspecified" if band == 0 else f"band-{band}"


def _gcs_bucket_after(passthrough_args: list[str], flag: str) -> str | None:
    """Bucket name from a ``<flag> gs://bucket/...`` (or ``<flag>=gs://...``) arg."""
    for i, arg in enumerate(passthrough_args):
        value: str | None = None
        if arg == flag and i + 1 < len(passthrough_args):
            value = passthrough_args[i + 1]
        elif arg.startswith(flag + "="):
            value = arg.split("=", 1)[1]
        if value and value.startswith("gs://"):
            return value[len("gs://") :].split("/", 1)[0]
    return None


def gcs_output_bucket_from_args(passthrough_args: list[str]) -> str | None:
    """First GCS output/checkpoint bucket in the args, whatever its name."""
    for flag in _GCS_OUTPUT_FLAGS:
        bucket = _gcs_bucket_after(passthrough_args, flag)
        if bucket:
            return bucket
    return None


def gcs_region_from_args(passthrough_args: list[str]) -> str | None:
    """Best-effort region of the job's GCS output/checkpoint bucket, parsed from
    the bucket name (e.g. ``gs://marin-us-east5/...`` -> ``us-east5``).

    Used to pin compute to the bucket's region so checkpoint writes stay
    in-region -- cross-region GCS traffic bills as egress (see AGENTS.md). This is
    a name heuristic (it assumes the bucket is named for its region, as Marin's
    are); ``--region`` is the authoritative override. Returns None when there is
    no ``gs://`` output arg or the bucket name carries no GCP region token,
    leaving placement unconstrained (the caller warns).
    """
    for flag in _GCS_OUTPUT_FLAGS:
        bucket = _gcs_bucket_after(passthrough_args, flag)
        if bucket:
            match = _GCP_REGION_RE.search(bucket)
            if match:
                return match.group(0)
    return None


def _device_summary(resources: ResourceConfig) -> str:
    """Human-readable accelerator description for a ResourceConfig."""
    device = resources.device
    variant = getattr(device, "variant", None)
    kind = getattr(device, "kind", None)
    if variant:
        return f"{kind or 'device'}:{variant}"
    return f"cpu:{resources.cpu}"


def build_job_request(
    preset_name: str,
    passthrough_args: list[str],
    *,
    name: str,
    module: str = "kelp.cli.train",
    inject_preset: bool = True,
    image: str | None = None,
    env_names: tuple[str, ...] = DEFAULT_ENV_PASSTHROUGH,
    replicas: int | None = None,
    region: str | None = None,
    priority_band: str = _DEFAULT_PRIORITY_BAND,
    environ: dict[str, str] | None = None,
) -> JobRequest:
    """Build a fray JobRequest that runs a Kelp module under a preset's resources.

    The preset selects only the accelerator slice; ``module`` chooses the
    entrypoint (``kelp.cli.train`` for training, ``kelp.cli.evaluate_mbpp`` for
    eval, ...). ``inject_preset`` passes ``--preset <name>`` to the module (the
    trainer reads it; eval modules take the config from the checkpoint, so they
    set this False). ``priority_band`` selects the Iris scheduling band
    (``batch`` by default -- see ``_PRIORITY_BANDS``). No cluster or network I/O
    -- it reads only ``environ`` (env passthrough, defaulting to ``os.environ``)
    and the passthrough args, so it can be unit-tested without a cluster. (It does
    emit a log line about region pinning; the returned request is a pure function
    of the inputs.)
    """
    if priority_band not in _PRIORITY_BANDS:
        raise ValueError(f"Unknown priority_band {priority_band!r}; choose one of {sorted(_PRIORITY_BANDS)}.")
    environ = environ if environ is not None else dict(os.environ)
    preset = get_preset(preset_name)
    resources = preset.resource

    # Pin the compute to the region of the GCS output/checkpoint bucket so
    # checkpoint writes stay in-region -- cross-region GCS traffic bills as egress
    # (see AGENTS.md; vet-cond-v2 ran outside us-east5 while its bucket was
    # us-east5 and tripped a high-egress alert). Explicit ``region`` wins;
    # otherwise it is inferred from the bucket name. Only applies to a TPU slice
    # that isn't already region/zone-constrained (a preset that pins its own zone
    # is honored). When a gs:// output bucket has no inferable region, warn rather
    # than silently leave placement unconstrained.
    target_region = region or gcs_region_from_args(passthrough_args)
    is_tpu = getattr(resources.device, "kind", None) == "tpu"
    if is_tpu and resources.regions is None and resources.zone is None:
        if target_region:
            resources = dataclasses.replace(resources, regions=[target_region])
            logger.info("Pinned compute to region %s (matches GCS bucket; avoids cross-region egress).", target_region)
        elif (bucket := gcs_output_bucket_from_args(passthrough_args)) is not None:
            logger.warning(
                "Could not infer a region from GCS output bucket %r; compute placement is "
                "unconstrained and MAY run cross-region (egress). Pass --region <r> to pin. See AGENTS.md.",
                bucket,
            )

    preset_flag = ["--preset", preset_name] if inject_preset else []
    command_args = ["-m", module, *preset_flag, *passthrough_args]
    entrypoint = Entrypoint.from_binary("python", command_args)

    # TPU jobs need JAX's TPU backend (libtpu). It lives in the `tpu` optional
    # dependency (linux-only), so request that extra when the slice is a TPU;
    # otherwise the worker's synced venv has CPU-only JAX and cannot see chips.
    extras = ["tpu"] if getattr(resources.device, "kind", None) == "tpu" else []

    # Named vars are forwarded from the current environment (iris does not copy
    # the shell env). create_environment also injects HF_TOKEN / WANDB_API_KEY
    # by default, and -- absent an --image -- syncs the local workspace so a
    # laptop launch runs your current checkout.
    env_vars = {k: environ[k] for k in env_names if k in environ}
    environment = create_environment(docker_image=image, env_vars=env_vars, extras=extras)

    # Fall back to the preset's replica count (fray uses request.replicas or 1
    # at submit, so a None here would silently under-provision a multi-host
    # preset whose resource requests replicas>1).
    #
    # Preemptible slices get reclaimed mid-run; without a preemption-retry budget
    # (fray default is 0) Iris does not restart the task, so a long job just
    # stops. Both entrypoints resume on restart -- training from the latest GCS
    # checkpoint, eval from its per-task shards -- so allow generous preemption
    # retries and a couple of failure retries for transient errors.
    return JobRequest(
        name=name,
        entrypoint=entrypoint,
        resources=resources,
        environment=environment,
        replicas=replicas if replicas is not None else resources.replicas,
        max_retries_preemption=20,
        max_retries_failure=2,
        priority=_PRIORITY_BANDS[priority_band],
    )


def format_dry_run(request: JobRequest, preset_name: str) -> str:
    """Render a JobRequest as a readable launch plan (for --dry-run)."""
    resources = request.resources
    be = request.entrypoint.binary_entrypoint
    command = f"{be.command} {' '.join(be.args)}" if be is not None else "<callable>"
    env = request.environment
    env_keys = sorted(env.env_vars) if env is not None else []
    if env is not None and env.docker_image:
        source = f"image={env.docker_image}"
    elif env is not None and env.workspace:
        source = f"workspace={env.workspace}"
    else:
        source = "(cluster default)"
    lines = [
        "DRY RUN -- not submitted. Re-run with --submit to launch.",
        f"  job name    : {request.name}",
        f"  preset      : {preset_name}",
        f"  accelerator : {_device_summary(resources)}  (cpu={resources.cpu}, ram={resources.ram})",
        f"  priority    : {_band_name(request.priority)} (band {request.priority})",
        f"  regions     : {resources.regions or '(scheduler default -- may be cross-region from bucket!)'}",
        f"  replicas    : {request.replicas if request.replicas is not None else resources.replicas}",
        f"  code        : {source}",
        f"  env passthru: {', '.join(env_keys) or '(none)'}",
        f"  command     : {command}",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a Kelp training run on Marin/Iris compute from a preset.",
        epilog="Arguments after '--' are forwarded to kelp-train.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="tpu_v5p_8",
        choices=list(PRESETS.keys()),
        help="Model/resource preset (determines both the model and the TPU slice).",
    )
    parser.add_argument("--name", type=str, default=None, help="Job name (default: kelp-<preset>).")
    parser.add_argument("--image", type=str, default=None, help="Worker Docker image (kelp + deps + gcsfs).")
    parser.add_argument(
        "--env",
        action="append",
        default=None,
        metavar="KEY",
        help="Env var name to pass into the job (repeatable). Default: W&B / HF tokens.",
    )
    parser.add_argument("--replicas", type=int, default=None, help="Number of job replicas (default: preset).")
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Pin compute to this GCS region (default: inferred from the --output-dir bucket) so "
        "checkpoints stay in-region and avoid cross-region egress. See AGENTS.md.",
    )
    parser.add_argument(
        "--priority-band",
        type=str,
        default=_DEFAULT_PRIORITY_BAND,
        choices=list(_PRIORITY_BANDS),
        help="Iris scheduling band (default: batch). 'batch' yields capacity to interactive/"
        "production work and is right for long, preemptible, checkpointed runs; use 'interactive' "
        "for a short job you are actively waiting on.",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default="marin",
        help="Iris cluster to submit to (resolves controller + credentials; default: marin).",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit. Without this flag, prints the launch plan and exits (dry run).",
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Args after '--' forwarded to kelp-train (e.g. --steps 50000 --wandb-project kelp).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(argv)

    # argparse.REMAINDER keeps a leading '--'; drop it.
    train_args = args.train_args[1:] if args.train_args and args.train_args[0] == "--" else args.train_args
    env_names = tuple(args.env) if args.env else DEFAULT_ENV_PASSTHROUGH
    name = args.name or f"kelp-{args.preset}"

    request = build_job_request(
        args.preset,
        train_args,
        name=name,
        image=args.image,
        env_names=env_names,
        replicas=args.replicas,
        region=args.region,
        priority_band=args.priority_band,
    )
    _submit_or_dry_run(request, args.preset, submit=args.submit, cluster=args.cluster, name=name)


def _submit_or_dry_run(request: JobRequest, preset_name: str, *, submit: bool, cluster: str, name: str) -> None:
    """Print the launch plan, or submit to Iris when ``submit`` is set."""
    if not submit:
        print(format_dry_run(request, preset_name))
        return

    # Submit to the cluster explicitly. fray's current_client() returns a
    # LocalClient off-cluster (which would run the job on this machine), so we
    # open an authenticated Iris client via the iris CLI's own helper --
    # open_iris_client resolves the controller URL and IAP credentials for the
    # named cluster (a bare IrisClient.remote to the URL returns Unauthorized) --
    # then wrap it with fray so its ResourceConfig / environment conversion
    # applies. The workspace is bundled and synced on the worker (with the tpu
    # extra). Submission happens inside the context so the endpoint stays live.
    from fray.iris_backend import FrayIrisClient
    from iris.cli.connect import open_iris_client

    logger.info("Submitting job %r to Iris cluster %s", name, cluster)
    with open_iris_client(cluster_name=cluster, workspace=Path.cwd()) as iris_client:
        client = FrayIrisClient.from_iris_client(iris_client)
        handle = client.submit(request)
    logger.info("Submitted: job_id=%s", handle.job_id)
    print(handle.job_id)


def parse_eval_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a Kelp eval run on Marin/Iris compute (reads gs:// checkpoints).",
        epilog="Arguments after '--' are forwarded to the eval module.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="tpu_vet",
        choices=list(PRESETS.keys()),
        help="Preset whose resources (accelerator slice) the eval runs on. The model config comes "
        "from the checkpoint, so this only picks the slice (default: tpu_vet -> v6e-4).",
    )
    parser.add_argument(
        "--module",
        type=str,
        default="kelp.cli.evaluate_mbpp",
        help="Eval entrypoint module (default: kelp.cli.evaluate_mbpp).",
    )
    parser.add_argument("--name", type=str, default=None, help="Job name (default: kelp-eval-<preset>).")
    parser.add_argument("--image", type=str, default=None, help="Worker Docker image (kelp + deps + gcsfs).")
    parser.add_argument("--env", action="append", default=None, metavar="KEY", help="Env var name to pass in.")
    parser.add_argument("--replicas", type=int, default=None, help="Number of job replicas (default: preset).")
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Pin compute to this GCS region (default: inferred from the --checkpoint-dir bucket) so "
        "the job runs in-region and avoids cross-region egress. See AGENTS.md.",
    )
    parser.add_argument(
        "--priority-band",
        type=str,
        default=_DEFAULT_PRIORITY_BAND,
        choices=list(_PRIORITY_BANDS),
        help="Iris scheduling band (default: batch). Use 'interactive' for a short eval you are actively waiting on.",
    )
    parser.add_argument("--cluster", type=str, default="marin", help="Iris cluster (default: marin).")
    parser.add_argument("--submit", action="store_true", help="Actually submit (default: dry run).")
    parser.add_argument(
        "eval_args",
        nargs=argparse.REMAINDER,
        help="Args after '--' forwarded to the eval module (e.g. --checkpoint-dir gs://... --n-best-of 16).",
    )
    return parser.parse_args(argv)


def eval_main(argv: list[str] | None = None) -> None:
    """Entrypoint for ``kelp-eval``: run an eval module on a slice, reading gs:// checkpoints."""
    configure_logging()
    args = parse_eval_args(argv)

    eval_args = args.eval_args[1:] if args.eval_args and args.eval_args[0] == "--" else args.eval_args
    env_names = tuple(args.env) if args.env else DEFAULT_ENV_PASSTHROUGH
    name = args.name or f"kelp-eval-{args.preset}"

    request = build_job_request(
        args.preset,
        eval_args,
        name=name,
        module=args.module,
        inject_preset=False,
        image=args.image,
        env_names=env_names,
        replicas=args.replicas,
        region=args.region,
        priority_band=args.priority_band,
    )
    _submit_or_dry_run(request, args.preset, submit=args.submit, cluster=args.cluster, name=name)


if __name__ == "__main__":
    main()
