#!/usr/bin/env bash
# Kelp vet-cond-v2 training: realistic in-context corruption on TPU (Iris).
#
# The successor to vet-cond-v1. Same ~115M tpu_vet model and prompt conditioning,
# but the training task is fixed per docs/vet-cond-v1-failure-analysis.md:
#
#   (a) --p-near-miss 0.85  -> most examples are realistic in-context bugs
#       (operator-flip / variable-swap), so alien bank-swap grafts are rare.
#   (b) docstring/string-literal statements are excluded from bank-swap
#       candidates in code (mutation._is_string_expr) -- no runbook flag needed.
#
# The corpus (stack_edu_python_vet.txt) is already prepared in GCS, so there is
# no corpus-prep phase here. Re-run scripts/train_v7.sh-style prep only if you
# need to regenerate it.
#
# Dry-run by default (kelp-launch prints the JobRequest and submits nothing).
# Pass --submit to actually launch on the cluster.
#
# Usage:
#   bash scripts/train_vet_cond_v2.sh            # dry-run: inspect the job
#   bash scripts/train_vet_cond_v2.sh --submit   # launch on Iris
#
# Override any knob via env, e.g.:
#   STEPS=80000 P_NEAR_MISS=0.9 bash scripts/train_vet_cond_v2.sh --submit

set -euo pipefail

PRESET="${PRESET:-tpu_vet}"
CORPUS_FILE="${CORPUS_FILE:-gs://marin-us-east5/kelp/corpus/stack_edu_python_vet.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-gs://marin-us-east5/kelp/checkpoints/vet-cond-v2}"
STEPS="${STEPS:-50000}"
P_NEAR_MISS="${P_NEAR_MISS:-0.85}"
# Cap corruption at 2 steps: realistic bugs are usually 1-2 edits, and this
# matches the eval's --corruption-steps (see scripts/eval_vet_cond_v2.sh). The
# linear curriculum still ramps 1 -> MAX_CORRUPTION_STEPS over warmup.
MAX_CORRUPTION_STEPS="${MAX_CORRUPTION_STEPS:-2}"
P_PROMPT="${P_PROMPT:-0.5}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"
SEED="${SEED:-42}"
WANDB_PROJECT="${WANDB_PROJECT:-kelp}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-vet-cond-v2}"
# Optional worker image (needs gcsfs for gs:// checkpoints). Empty => sync the
# local workspace instead of using a prebuilt image.
IMAGE="${IMAGE:-}"

SUBMIT=""
for arg in "$@"; do
    case "$arg" in
        --submit) SUBMIT="--submit" ;;
    esac
done

LAUNCH_FLAGS=(--preset "$PRESET")
[ -n "$SUBMIT" ] && LAUNCH_FLAGS+=("$SUBMIT")
[ -n "$IMAGE" ] && LAUNCH_FLAGS+=(--image "$IMAGE")

# W&B logging depends on WANDB_API_KEY being present at launch (kelp-launch
# forwards it from the shell env). Missing it silently disables charts.
if [ -n "$SUBMIT" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY is unset -- the run will NOT log to W&B." >&2
    echo "         Re-run as: WANDB_API_KEY=... bash scripts/train_vet_cond_v2.sh --submit" >&2
fi

echo "=== vet-cond-v2: ${SUBMIT:-dry-run} (preset=$PRESET, p_near_miss=$P_NEAR_MISS) ==="
uv run kelp-launch "${LAUNCH_FLAGS[@]}" \
    -- \
    --corpus-file "$CORPUS_FILE" \
    --steps "$STEPS" \
    --augment \
    --prompt-conditioning \
    --p-prompt "$P_PROMPT" \
    --p-near-miss "$P_NEAR_MISS" \
    --max-corruption-steps "$MAX_CORRUPTION_STEPS" \
    --corruption-curriculum linear \
    --data-loader streaming \
    --output-dir "$OUTPUT_DIR" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --log-interval 10 \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$WANDB_RUN_NAME" \
    --seed "$SEED"
