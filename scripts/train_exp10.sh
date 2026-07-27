#!/usr/bin/env bash
# Kelp exp10 training: single-edit repair, scaled model, bf16 (Iris).
#
# The successor to vet-cond-v2 (exp9). exp9 lifted MBPP repair ~2% -> ~15% with
# 100% validity and concluded the wall is now localization/correctness, not
# validity or corpus. exp10 changes two things on top of that clean task:
#
#   1. Single-edit training: --max-corruption-steps 1 (constant curriculum). Tree
#      diffusion repairs iteratively (one edit per step at inference), so a
#      single-edit target IS the per-step objective -- it removes a train/infer
#      mismatch and sharpens the localization signal that is the current wall.
#   2. Capacity: the tpu_vet_300m preset (~305M, bf16) on the same v6e-4 slice.
#      The bf16 + fused-attention landing (#132) makes ~2.6x the params fit at
#      ~similar wall-clock -- this run tests whether capacity moves the
#      correctness wall now that the task/corpus are good.
#
# Everything else is held at the exp9 values (realistic-or-drop corruption,
# prompt conditioning, curated_v2 corpus, batch 64, 50K steps, LR 3e-4) so exp10
# varies only {task=single-edit, capacity}. Run the 115M single-edit CONTROL by
# overriding the preset/paths (see the env block) to separate the two effects:
#
#   PRESET=tpu_vet \
#   OUTPUT_DIR=gs://marin-us-east5/kelp/checkpoints/exp10-115m-control \
#   WANDB_RUN_NAME=exp10-115m-control \
#   bash scripts/train_exp10.sh --submit
#
# Dry-run by default; pass --submit to launch. Jobs submit in the BATCH band by
# default (kelp-launch) and pin to the bucket's region (us-east5) automatically.
#
# Usage:
#   bash scripts/train_exp10.sh            # dry-run: inspect the 300M job
#   bash scripts/train_exp10.sh --submit   # launch the 300M run on Iris

set -euo pipefail

PRESET="${PRESET:-tpu_vet_300m}"
CORPUS_FILE="${CORPUS_FILE:-gs://marin-us-east5/kelp/corpus/curated_v2.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-gs://marin-us-east5/kelp/checkpoints/exp10-300m}"
STEPS="${STEPS:-50000}"
# Single edit per example: one realistic in-context bug, one repair. Constant
# (not curriculum) because there is nothing to ramp at max=1.
MAX_CORRUPTION_STEPS="${MAX_CORRUPTION_STEPS:-1}"
CORRUPTION_CURRICULUM="${CORRUPTION_CURRICULUM:-constant}"
# Realistic-or-drop: always attempt an in-context corruption; drop if none
# applies rather than grafting an alien bank subtree (exp9's key fix).
P_NEAR_MISS="${P_NEAR_MISS:-1.0}"
P_PROMPT="${P_PROMPT:-0.5}"
# 2000, not 5000: what the recorded exp10 runs actually used after preemption
# thrash (README v10) -- the step-48000 eval checkpoint only exists at 2000.
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-2000}"
SEED="${SEED:-42}"
WANDB_PROJECT="${WANDB_PROJECT:-kelp}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-exp10-300m}"
# Optional worker image (needs gcsfs for gs:// checkpoints). Empty => sync the
# local workspace instead of using a prebuilt image.
IMAGE="${IMAGE:-}"

SUBMIT=""
for arg in "$@"; do
    case "$arg" in
        --submit) SUBMIT="--submit" ;;
    esac
done

# Iris scheduling band. Defaults to the launcher's default (batch) -- the good
# citizen for long runs. Override with PRIORITY_BAND=interactive for a run you
# are actively waiting on against a deadline (batch can be preempted, and a
# preemption before the first checkpoint restarts from step 0).
PRIORITY_BAND="${PRIORITY_BAND:-}"

LAUNCH_FLAGS=(--preset "$PRESET")
[ -n "$SUBMIT" ] && LAUNCH_FLAGS+=("$SUBMIT")
[ -n "$IMAGE" ] && LAUNCH_FLAGS+=(--image "$IMAGE")
[ -n "$PRIORITY_BAND" ] && LAUNCH_FLAGS+=(--priority-band "$PRIORITY_BAND")

# W&B logging depends on WANDB_API_KEY being present at launch (kelp-launch
# forwards it from the shell env). Missing it silently disables charts -- but the
# MFU/throughput lines still print to the job logs.
if [ -n "$SUBMIT" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY is unset -- the run will NOT log to W&B." >&2
    echo "         Re-run as: WANDB_API_KEY=... bash scripts/train_exp10.sh --submit" >&2
fi

echo "=== exp10: ${SUBMIT:-dry-run} (preset=$PRESET, max_corruption_steps=$MAX_CORRUPTION_STEPS, run=$WANDB_RUN_NAME) ==="
uv run kelp-launch "${LAUNCH_FLAGS[@]}" \
    -- \
    --corpus-file "$CORPUS_FILE" \
    --steps "$STEPS" \
    --augment \
    --prompt-conditioning \
    --p-prompt "$P_PROMPT" \
    --p-near-miss "$P_NEAR_MISS" \
    --no-bank-swap-fallback \
    --max-corruption-steps "$MAX_CORRUPTION_STEPS" \
    --corruption-curriculum "$CORRUPTION_CURRICULUM" \
    --data-loader streaming \
    --output-dir "$OUTPUT_DIR" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --log-interval 10 \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$WANDB_RUN_NAME" \
    --seed "$SEED"
