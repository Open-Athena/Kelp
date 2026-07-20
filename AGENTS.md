# AGENTS.md

Operational guidance for coding agents (Claude Code, etc.) working in the Kelp
repository. Keep this current — add a section whenever a new footgun bites.

## Launching TPU jobs: avoid cross-region GCS egress

**Incident (2026-07-18).** The `vet-cond-v2` training job (`kelp-tpu_vet`) tripped
a high-egress alert — the egress report attributed **169 of 224 cross-region GCS
operations (75%)** to it. Root cause: the v6e-4 slice ran **outside us-east5**
while checkpoints were written to **`gs://marin-us-east5/...`** (region
**us-east5**). All **11.78 GiB** of checkpoints (10 × ~1.18 GiB, one every 5000
steps) crossed regions, which bills as network egress. There was **no
external-download cause** — W&B logs metrics only, and the corpus streams from the
same-region bucket.

Two subtleties the alert revealed:
- The dominant route was **`europe-west4 → us-east5`** (i.e. *inter-continental*,
  the most expensive tier), not intra-US — the `us-east1-d` worker seen in `job
  summary` was only the final slice.
- The run was **preempted once**, and the retry **migrated it to a different
  region** mid-run. So pinning must constrain *every* placement, including
  preemption retries — which `regions=[...]` does.

**Rule: the compute region MUST match the region of every GCS bucket the job
reads or writes (checkpoints and corpus).** Cross-region GCS traffic is egress.

Before launching any `kelp-launch` / `kelp-eval` job that touches GCS:

1. **Region pinning is automatic (as of the launcher fix).** `build_job_request`
   in `src/kelp/cli/launch.py` infers the region from the `--output-dir` /
   `--checkpoint-dir` bucket (`gs://marin-us-east5/...` → `us-east5`) and pins the
   TPU slice to it (`ResourceConfig.regions`), so checkpoint writes stay in-region
   on every placement *and* on preemption retries. The dry-run prints the pinned
   `regions:` line; a `Pinned compute to region …` log confirms it on submit.
   Override with `kelp-launch --region <r>` (authoritative) if you must. The
   inference is a **bucket-name heuristic** — it assumes the bucket is named for
   its region, as Marin's are (`marin-us-east5` → us-east5), and validates the
   region token's shape. If it can't infer a region from a `gs://` output bucket
   it **warns and leaves placement unconstrained** (so watch for that warning, or
   just pass `--region`). The default bucket `gs://marin-us-east5` is us-east5,
   where v6e is available in `us-east5-b` (marin v6e zones: `europe-west4-a,
   us-east1-d, us-east5-b`).
   - Caveat: pinning to one region can leave the job *pending* if that region is
     out of (preemptible) capacity. That's the right trade — wait for in-region
     capacity rather than silently pay inter-continental egress. Widen with
     `--region` only if you also move the bucket.
2. **Still sanity-check the region after submit.** `uv run iris --cluster=marin
   job summary /<user>/<job>` shows the worker (e.g.
   `marin-tpu-v6e-preemptible-4-us-east5-b`). Confirm it matches the bucket's
   region — belt and suspenders.
3. **If you deliberately run in another region,** point `--output-dir` at a bucket
   in the *compute's* region and copy only the FINAL checkpoint cross-region with
   a single `gsutil cp` — never one transfer per checkpoint interval.

## Scheduling: jobs default to the BATCH priority band

`kelp-launch` / `kelp-eval` submit in Iris's **BATCH** band (`priority_band=3`)
by default. Bands are `PRODUCTION(1) > INTERACTIVE(2) > BATCH(3)` (lower number =
higher priority; the scheduler orders by band ascending, and preemption can only
evict strictly-lower-priority work). BATCH is right for kelp jobs because they
are long, non-interactive, already preemptible (`max_retries_preemption=20`) and
resume from checkpoints -- they should yield capacity to interactive/production
work rather than compete with it. It also avoids a budget-cliff churn: an
over-budget INTERACTIVE job is downgraded to BATCH mid-run and can oscillate back
(Iris `compute_effective_band`); starting at BATCH sidesteps that.

- The dry-run prints `priority : batch (band 3)`.
- Override with `kelp-launch --priority-band interactive` for a **short** job you
  are actively waiting on. Don't use `interactive`/`production` for long training
  runs -- that competes with real interactive work for scarce TPU capacity.
- Trade-off: a BATCH job can wait longer for capacity and is preempted first.
  That is fine here (checkpointed + resumable + region-pinned); if a run keeps
  getting preempted, check capacity before reaching for a higher band.

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
