# AGENTS.md

Operational guidance for coding agents (Claude Code, etc.) working in the Kelp
repository. Keep this current — add a section whenever a new footgun bites.

## Launching TPU jobs: avoid cross-region GCS egress

**Incident (2026-07-18).** The `vet-cond-v2` training job (`kelp-tpu_vet`) tripped
a high-egress alert. Root cause: the v6e-4 slice was scheduled in **`us-east1-d`**,
but checkpoints were written to **`gs://marin-us-east5/...`** (region **us-east5**).
All **11.78 GiB** of checkpoints (10 × ~1.18 GiB, one every 5000 steps) crossed
regions `us-east1 → us-east5`, which bills as network egress. There was **no
external-download cause** — W&B logs metrics only, and the corpus streams from the
same-region bucket.

**Rule: the compute region MUST match the region of every GCS bucket the job
reads or writes (checkpoints and corpus).** Cross-region GCS traffic is egress.

Before launching any `kelp-launch` / `kelp-eval` job that touches GCS:

1. **Pick a zone in the bucket's region.** The default checkpoint bucket
   `gs://marin-us-east5` is in **us-east5**, and v6e is available in **`us-east5-b`**
   (marin cluster v6e zones: `europe-west4-a, us-east1-d, us-east5-b`). Prefer
   `us-east5-b` so checkpoint writes stay in-region.
   - `kelp-launch` does not yet expose a zone flag — the assigned zone is
     scheduler/capacity-driven. **Proper fix (recommended follow-up):** add a
     zone constraint to the `JobRequest` in `src/kelp/cli/launch.py` that prefers
     the `--output-dir` bucket's region. Until then, use step 2.
2. **Verify the assigned region right after submit.** `uv run iris --cluster=marin
   job summary /<user>/<job>` shows the worker name, e.g.
   `marin-tpu-v6e-preemptible-4-us-east1-d`. If that region ≠ the `--output-dir`
   bucket's region, you are paying egress on every checkpoint — kill and relaunch,
   or switch `--output-dir` to a same-region bucket.
3. **If capacity forces a different region,** point `--output-dir` at a bucket in
   the *compute's* region and copy only the FINAL checkpoint cross-region with a
   single `gsutil cp` — never one transfer per checkpoint interval.

## Minimize checkpoint volume

Egress and storage both scale with (checkpoint size × count). A ~115M-param
checkpoint here is **~1.18 GiB** (params + the optimizer `train_state`).

- **Set `--checkpoint-interval` deliberately.** The `kelp-train` default is
  **1000**; a 50k-step run at the default writes **50** checkpoints (~59 GiB).
  Use ≥ 5000 for long runs (the `scripts/*_vet_cond_v2.sh` runbooks do).
- No checkpoint rotation exists yet — all checkpoints are kept. A keep-last-N
  option and **bf16 checkpoints (issue #132)** would roughly halve the footprint.

## Keep data reads in-region too

Stream corpora from a **same-region GCS bucket** (e.g.
`gs://marin-us-east5/kelp/corpus/...`), never from the HuggingFace Hub on cluster
jobs. `kelp.cli.prepare_corpus` runs locally; upload the result to a same-region
bucket and point the runbooks at it. (Do **not** overwrite Marin's existing
corpora — write to a new object path.)

## W&B

W&B (`wandb.ai`) logging is metrics-only and negligible egress — fine to leave on.
Its API key is forwarded from the launching shell's `WANDB_API_KEY`; the runbooks
warn if it is unset before `--submit`.
