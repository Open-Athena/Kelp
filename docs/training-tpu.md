# Training on TPU (Marin / Iris)

Kelp trains on Google TPUs through [Marin's](https://marin.community/) compute
stack. This guide covers launching a run, what happens inside the job, and how
checkpoints and monitoring work.

## The pieces

| Layer | Library | Role |
|-------|---------|------|
| Job orchestration | **Iris** | Places a container on a TPU slice, brings up multi-host JAX, streams logs |
| Backend abstraction | **fray** | Backend-agnostic `JobRequest` / `ResourceConfig`; converts to Iris (and gives a local backend for dev) |
| Distributed init | **Levanter** | `initialize_iris_jax()` — the Iris↔JAX handshake |
| Sharding | **kelp** | Data-parallel mesh over the slice's chips |
| Checkpoints | **Orbax** | TensorStore/Zarr checkpoints written directly to GCS |

You launch with `kelp-launch`; you do **not** hand-write Iris job specs — Kelp
speaks fray, and fray converts to Iris.

## Prerequisites

1. **A worker image** containing kelp + its dependencies (including `gcsfs` for
   `gs://` checkpoints). Build and push it with Iris:

   ```bash
   iris build worker-image
   ```

   Pass the resulting image reference to `kelp-launch --image ...`. Without
   `--image`, the launcher syncs your local checkout instead (handy for quick
   iteration, slower to start).

2. **Authentication.** For headless/CI use, impersonate the controller service
   account via Application Default Credentials:

   ```bash
   gcloud auth application-default login \
     --impersonate-service-account=iris-controller@<project>.iam.gserviceaccount.com
   ```

   Interactive users can instead run `iris login` (browser OAuth). See
   [lib/iris/OPS.md](https://github.com/marin-community/marin/blob/main/lib/iris/OPS.md).

3. **A GCS bucket** for checkpoints (the job container is ephemeral — local
   checkpoints vanish when the job ends).

## Launching a run

`kelp-launch` maps a preset's `ResourceConfig` (e.g. `tpu_v5p_8` →
`ResourceConfig.with_tpu('v5p-8')`) onto a TPU slice and runs `kelp-train`
inside the job. **Dry-run is the default** — nothing is submitted until you pass
`--submit`.

```bash
# 1. Inspect exactly what would be launched (no submission):
kelp-launch --preset tpu_v5p_8 -- --steps 50000 --wandb-project kelp

# 2. Submit for real, using a worker image and a GCS checkpoint dir:
kelp-launch --preset tpu_v5p_8 --submit \
  --image gcr.io/<project>/kelp:latest \
  -- --steps 50000 --wandb-project kelp \
     --output-dir gs://<bucket>/kelp/v8/
```

Everything after `--` is forwarded verbatim to `kelp-train`, so all the usual
training flags (`--augment`, `--prompt-conditioning`, `--corruption-curriculum`,
etc.) apply.

### Key flags

| Flag | Meaning |
|------|---------|
| `--preset` | Model **and** TPU slice (both come from the preset). Default `tpu_v5p_8`. |
| `--submit` | Actually submit. Omit for a dry-run plan. |
| `--image` | Worker Docker image. Omit to sync the local checkout. |
| `--env KEY` | Forward an env var into the job (repeatable). Defaults cover W&B / HF tokens. |
| `--replicas` | Number of job replicas (default: the preset's). |
| `--name` | Job name (default `kelp-<preset>`). |

> **Env vars are not copied from your shell.** Iris starts jobs with a clean
> environment, so secrets like `WANDB_API_KEY` must be named explicitly (they
> are then read from your current environment at launch time). The defaults
> already include the W&B and HF tokens.

## What happens inside the job

1. **Distributed init.** `kelp-train` calls `bootstrap_distributed()` first,
   which runs `initialize_iris_jax()` — a no-op on a single host, and the
   coordinator handshake on a multi-host slice. This must precede any device use
   so the training mesh sees the *global* device set.
2. **Data-parallel sharding.** Training builds a 1-D device mesh over all chips;
   the batch is sharded across chips and model state is replicated. The JITted
   step runs SPMD with gradient all-reduce inserted automatically. A single
   chip and 8 chips produce bit-identical loss — only placement changes.
   Requirement: the global `batch_size` must be divisible by the chip count
   (the launcher's presets already satisfy this).
3. **Checkpointing to GCS.** With `--output-dir gs://...`, checkpoints are
   written via Orbax (TensorStore/Zarr) directly to the bucket. Each checkpoint
   is a directory with a `config.json` sidecar and a `params/` array tree. On
   multi-host, the array save is a collective across all processes; only
   process 0 writes the JSON sidecar.

## Monitoring

Use the Iris CLI (see [lib/iris/OPS.md](https://github.com/marin-community/marin/blob/main/lib/iris/OPS.md)):

```bash
iris job list --state running
iris job logs /<user>/kelp-tpu_v5p_8 -f
iris task exec /<user>/kelp-tpu_v5p_8/0 -- python -c "import jax; print(jax.devices())"
```

## Loading a checkpoint

`load_checkpoint` accepts local or `gs://` paths transparently:

```python
from kelp.model.checkpointing import load_checkpoint, find_best_checkpoint

best = find_best_checkpoint("gs://<bucket>/kelp/v8/")
params, config = load_checkpoint(best)
```

## Current limitations / roadmap

- **Data generation is still inline.** Training examples are generated on the
  training host's CPU per step. For the 8B/v5p target a handful of cores keep the
  TPU fed, but smaller/faster-step models can starve. Distributed data-generation
  (streaming synthesis over the corpus) is tracked as phase 2 — see the Zephyr
  data-pipeline issues.
- **Checkpointing is synchronous.** The GCS write blocks the step it fires on.
  Async checkpointing (overlapping the write with subsequent steps via a
  persistent Orbax `CheckpointManager`), plus replica-parallel save and
  load+broadcast at scale, are tracked performance follow-ups.
- **FSDP.** Only data parallelism is implemented (params replicated). Sharding
  params for FSDP is a future extension for when one model replica outgrows a
  single chip.
