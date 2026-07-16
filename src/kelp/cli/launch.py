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
<passthrough args>``; that process calls ``bootstrap_distributed()`` (#115) and
shards across the slice's chips (#114).

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
import logging
import os
from pathlib import Path

from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment

from kelp.cli._logging import configure_logging
from kelp.training.presets import PRESETS, get_preset

logger = logging.getLogger(__name__)

DEFAULT_ENV_PASSTHROUGH = ("WANDB_API_KEY", "WANDB_ENTITY", "HF_TOKEN")


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
    train_args: list[str],
    *,
    name: str,
    image: str | None = None,
    env_names: tuple[str, ...] = DEFAULT_ENV_PASSTHROUGH,
    replicas: int | None = None,
    environ: dict[str, str] | None = None,
) -> JobRequest:
    """Build a fray JobRequest that runs kelp-train under the preset's resources.

    Pure and side-effect-free (reads ``environ`` for env passthrough, defaulting
    to ``os.environ``) so it can be unit-tested without a cluster.
    """
    environ = environ if environ is not None else dict(os.environ)
    preset = get_preset(preset_name)
    resources = preset.resource

    command_args = ["-m", "kelp.cli.train", "--preset", preset_name, *train_args]
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

    return JobRequest(
        name=name,
        entrypoint=entrypoint,
        resources=resources,
        environment=environment,
        replicas=replicas,
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
        args.preset, train_args, name=name, image=args.image, env_names=env_names, replicas=args.replicas
    )

    if not args.submit:
        print(format_dry_run(request, args.preset))
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

    logger.info("Submitting job %r to Iris cluster %s", name, args.cluster)
    with open_iris_client(cluster_name=args.cluster, workspace=Path.cwd()) as iris_client:
        client = FrayIrisClient.from_iris_client(iris_client)
        handle = client.submit(request)
    logger.info("Submitted: job_id=%s", handle.job_id)
    print(handle.job_id)


if __name__ == "__main__":
    main()
