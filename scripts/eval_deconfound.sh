#!/usr/bin/env bash
# The exp10 deconfounding grid (issues #142, #143): checkpoints x corruption
# steps, on the SAME task set with the SAME (subprocess, non-escapable) timeout.
#
# The README v10 comparison changed training AND eval difficulty at once (v9
# evaluated at 2-step corruption, exp10 at 1-step) with a hang-biased v9
# denominator (33/50 survivors). This grid separates the effects:
#
#   * v9 (vet-cond-v2, multi-step-trained) at steps 1,2,3  -> was the ~15->18%
#     "floor lift" the training change, or just the easier 1-step eval?
#   * exp10 (single-edit-trained) at steps 2,3              -> what did single-
#     edit training cost in multi-error repair (the capability that makes this
#     iterative diffusion rather than one-shot cloze repair)?
#
# All cells run the matched (realistic-or-drop) arm with best-of-16, n=500
# tasks (the "final number" from eval_exp10.sh that was never run) and now
# report bootstrap CIs + tasks-fully-repaired. Fingerprinted shards mean cells
# resume after preemption and re-runs reuse completed tasks.
#
# Usage:
#   bash scripts/eval_deconfound.sh              # dry-run the full grid
#   bash scripts/eval_deconfound.sh --submit     # launch all cells on Iris
#   CHECKPOINT_DIRS="gs://.../exp10-300m" STEPS_LIST="2 3" \
#     bash scripts/eval_deconfound.sh --submit   # just the missing cells

set -euo pipefail

PRESET="${PRESET:-tpu_vet}"   # slice only; model config comes from the checkpoint
GS_BASE="${GS_BASE:-gs://marin-us-east5/kelp/checkpoints}"
# v9 multi-step-trained, exp10 single-edit 115M control, exp10 300M.
CHECKPOINT_DIRS="${CHECKPOINT_DIRS:-$GS_BASE/vet-cond-v2 $GS_BASE/exp10-115m-control $GS_BASE/exp10-300m}"
STEPS_LIST="${STEPS_LIST:-1 2 3}"
CORPUS_FILE="${CORPUS_FILE:-gs://marin-us-east5/kelp/corpus/curated_v2.txt}"
NUM_CORRUPTIONS="${NUM_CORRUPTIONS:-5}"
N_BEST_OF="${N_BEST_OF:-16}"
MAX_TASKS="${MAX_TASKS:-500}"
TEST_TIMEOUT="${TEST_TIMEOUT:-5}"
SEED="${SEED:-42}"
WANDB_PROJECT="${WANDB_PROJECT:-kelp}"
RUN_PREFIX="${RUN_PREFIX:-deconfound}"
IMAGE="${IMAGE:-}"
PRIORITY_BAND="${PRIORITY_BAND:-}"

SUBMIT=""
for arg in "$@"; do
    case "$arg" in
        --submit) SUBMIT="--submit" ;;
    esac
done

if [ -n "$SUBMIT" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY is unset -- eval metrics will NOT log to W&B." >&2
fi

for CKPT_DIR in $CHECKPOINT_DIRS; do
    CKPT_TAG=$(basename "$CKPT_DIR")
    for STEPS in $STEPS_LIST; do
        TAG="${CKPT_TAG}-s${STEPS}"
        launch_flags=(--preset "$PRESET" --name "kelp-eval-${RUN_PREFIX}-$TAG")
        [ -n "$SUBMIT" ] && launch_flags+=("$SUBMIT")
        [ -n "$IMAGE" ] && launch_flags+=(--image "$IMAGE")
        [ -n "$PRIORITY_BAND" ] && launch_flags+=(--priority-band "$PRIORITY_BAND")

        echo "=== cell [$TAG] ckpt=$CKPT_DIR steps=$STEPS tasks=$MAX_TASKS (${SUBMIT:-dry-run}) ==="
        uv run kelp-eval "${launch_flags[@]}" -- \
            --checkpoint-dir "$CKPT_DIR" \
            --corpus-file "$CORPUS_FILE" \
            --num-corruptions "$NUM_CORRUPTIONS" \
            --corruption-steps "$STEPS" \
            --no-bank-swap-fallback \
            --n-best-of "$N_BEST_OF" \
            --max-tasks "$MAX_TASKS" \
            --test-timeout "$TEST_TIMEOUT" \
            --seed "$SEED" \
            --wandb-project "$WANDB_PROJECT" \
            --wandb-run-name "${RUN_PREFIX}-$TAG"
    done
done
