#!/usr/bin/env bash
# Kelp vet-cond-v2 evaluation: matched + unmatched MBPP repair on TPU (Iris).
#
# Runs the MBPP repair eval twice against the same checkpoint, differing only in
# the corruption distribution:
#
#   matched   (--p-near-miss 1.0 --no-bank-swap-fallback) -- same realistic
#             in-context bugs the model trained on (alien grafts dropped, not
#             injected). Answers "did it learn the trained task?"
#   unmatched (--p-near-miss 0.0, bank-swap) -- the original out-of-context
#             bank-swap. Answers "does the skill transfer to a different bug
#             distribution?"
#
# Both use --corruption-steps 2 to match training's capped corruption. The eval
# fingerprint includes p_near_miss, so the two runs write separate result shards
# and can run/cache independently. Each logs to its own W&B run.
#
# Dry-run by default; pass --submit to launch both on the cluster.
#
# Usage:
#   bash scripts/eval_vet_cond_v2.sh                       # dry-run both
#   bash scripts/eval_vet_cond_v2.sh --submit              # launch both (latest ckpt)
#   CHECKPOINT=step-020000 bash scripts/eval_vet_cond_v2.sh --submit   # a specific step
#   MAX_TASKS=500 bash scripts/eval_vet_cond_v2.sh --submit            # full eval set

set -euo pipefail

PRESET="${PRESET:-tpu_vet}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-gs://marin-us-east5/kelp/checkpoints/vet-cond-v2}"
CHECKPOINT="${CHECKPOINT:-}"  # specific step-XXXXXX; empty => latest in CHECKPOINT_DIR
CORPUS_FILE="${CORPUS_FILE:-gs://marin-us-east5/kelp/corpus/stack_edu_python_vet.txt}"
CORRUPTION_STEPS="${CORRUPTION_STEPS:-2}"
NUM_CORRUPTIONS="${NUM_CORRUPTIONS:-5}"
N_BEST_OF="${N_BEST_OF:-16}"
MAX_TASKS="${MAX_TASKS:-50}"   # 50 = fast loop; 500 = final number
SEED="${SEED:-42}"
WANDB_PROJECT="${WANDB_PROJECT:-kelp}"
IMAGE="${IMAGE:-}"

SUBMIT=""
for arg in "$@"; do
    case "$arg" in
        --submit) SUBMIT="--submit" ;;
    esac
done

if [ -n "$SUBMIT" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY is unset -- eval metrics will NOT log to W&B." >&2
    echo "         Re-run as: WANDB_API_KEY=... bash scripts/eval_vet_cond_v2.sh --submit" >&2
fi

run_eval() {
    local p_near_miss="$1" tag="$2" no_bank_swap="$3"
    local launch_flags=(--preset "$PRESET" --name "kelp-eval-v2-$tag")
    [ -n "$SUBMIT" ] && launch_flags+=("$SUBMIT")
    [ -n "$IMAGE" ] && launch_flags+=(--image "$IMAGE")

    local eval_args=(
        --checkpoint-dir "$CHECKPOINT_DIR"
        --corpus-file "$CORPUS_FILE"
        --num-corruptions "$NUM_CORRUPTIONS"
        --corruption-steps "$CORRUPTION_STEPS"
        --p-near-miss "$p_near_miss"
        --n-best-of "$N_BEST_OF"
        --max-tasks "$MAX_TASKS"
        --seed "$SEED"
        --wandb-project "$WANDB_PROJECT"
        --wandb-run-name "vet-cond-v2-eval-$tag"
    )
    [ -n "$no_bank_swap" ] && eval_args+=(--no-bank-swap-fallback)
    [ -n "$CHECKPOINT" ] && eval_args+=(--checkpoint "$CHECKPOINT")

    echo "=== eval [$tag] p_near_miss=$p_near_miss steps=$CORRUPTION_STEPS tasks=$MAX_TASKS (${SUBMIT:-dry-run}) ==="
    uv run kelp-eval "${launch_flags[@]}" -- "${eval_args[@]}"
}

# matched: realistic-or-drop, mirrors training. unmatched: pure bank-swap.
run_eval 1.0 matched --no-bank-swap-fallback
run_eval 0.0 unmatched ""
