#!/usr/bin/env bash
# Kelp exp10 evaluation: matched + unmatched MBPP repair on TPU (Iris).
#
# Same two-distribution eval as vet-cond-v2, with two exp10 changes:
#
#   * --corruption-steps 1 (default) to MATCH exp10's single-edit training. The
#     matched arm measures the trained (single realistic in-context edit)
#     distribution; the unmatched arm (bank-swap) measures transfer.
#   * The per-test execution timeout (--test-timeout, default 5s in the eval CLI)
#     now bounds each candidate, so a non-terminating repair fails its test
#     instead of hanging the run -- the vet-cond-v2 failure that lost 17/50 tasks.
#
# Point CHECKPOINT_DIR at the run you want to score (defaults to the 300M run);
# for the control use CHECKPOINT_DIR=.../exp10-115m-control. The eval fingerprint
# includes p_near_miss + corruption_steps, so arms/steps write separate shards.
#
# Usage:
#   bash scripts/eval_exp10.sh                    # dry-run both (300M ckpt)
#   bash scripts/eval_exp10.sh --submit           # launch both (latest ckpt)
#   CHECKPOINT_DIR=gs://marin-us-east5/kelp/checkpoints/exp10-115m-control \
#     bash scripts/eval_exp10.sh --submit         # score the control
#   MAX_TASKS=500 bash scripts/eval_exp10.sh --submit    # full eval set

set -euo pipefail

PRESET="${PRESET:-tpu_vet}"   # slice only; model config comes from the checkpoint
CHECKPOINT_DIR="${CHECKPOINT_DIR:-gs://marin-us-east5/kelp/checkpoints/exp10-300m}"
CHECKPOINT="${CHECKPOINT:-}"  # specific step-XXXXXX; empty => latest in CHECKPOINT_DIR
CORPUS_FILE="${CORPUS_FILE:-gs://marin-us-east5/kelp/corpus/curated_v2.txt}"
CORRUPTION_STEPS="${CORRUPTION_STEPS:-1}"   # match exp10 single-edit training
NUM_CORRUPTIONS="${NUM_CORRUPTIONS:-5}"
N_BEST_OF="${N_BEST_OF:-16}"
MAX_TASKS="${MAX_TASKS:-50}"   # 50 = fast loop; 500 = final number
TEST_TIMEOUT="${TEST_TIMEOUT:-5}"
SEED="${SEED:-42}"
WANDB_PROJECT="${WANDB_PROJECT:-kelp}"
RUN_PREFIX="${RUN_PREFIX:-exp10}"
IMAGE="${IMAGE:-}"
# Iris band; default launcher default (batch). Set interactive when you're
# waiting on the result and the cluster is contended.
PRIORITY_BAND="${PRIORITY_BAND:-}"

SUBMIT=""
for arg in "$@"; do
    case "$arg" in
        --submit) SUBMIT="--submit" ;;
    esac
done

if [ -n "$SUBMIT" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY is unset -- eval metrics will NOT log to W&B." >&2
    echo "         Re-run as: WANDB_API_KEY=... bash scripts/eval_exp10.sh --submit" >&2
fi

run_eval() {
    local p_near_miss="$1" tag="$2" no_bank_swap="$3"
    local launch_flags=(--preset "$PRESET" --name "kelp-eval-${RUN_PREFIX}-$tag")
    [ -n "$SUBMIT" ] && launch_flags+=("$SUBMIT")
    [ -n "$IMAGE" ] && launch_flags+=(--image "$IMAGE")
    [ -n "$PRIORITY_BAND" ] && launch_flags+=(--priority-band "$PRIORITY_BAND")

    local eval_args=(
        --checkpoint-dir "$CHECKPOINT_DIR"
        --corpus-file "$CORPUS_FILE"
        --num-corruptions "$NUM_CORRUPTIONS"
        --corruption-steps "$CORRUPTION_STEPS"
        --p-near-miss "$p_near_miss"
        --n-best-of "$N_BEST_OF"
        --max-tasks "$MAX_TASKS"
        --test-timeout "$TEST_TIMEOUT"
        --seed "$SEED"
        --wandb-project "$WANDB_PROJECT"
        --wandb-run-name "${RUN_PREFIX}-eval-$tag"
    )
    [ -n "$no_bank_swap" ] && eval_args+=(--no-bank-swap-fallback)
    [ -n "$CHECKPOINT" ] && eval_args+=(--checkpoint "$CHECKPOINT")

    echo "=== eval [$tag] p_near_miss=$p_near_miss steps=$CORRUPTION_STEPS tasks=$MAX_TASKS (${SUBMIT:-dry-run}) ==="
    uv run kelp-eval "${launch_flags[@]}" -- "${eval_args[@]}"
}

# matched: realistic-or-drop, mirrors training. unmatched: pure bank-swap.
run_eval 1.0 matched --no-bank-swap-fallback
run_eval 0.0 unmatched ""
