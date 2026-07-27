#!/usr/bin/env bash
# Kelp exp12 evaluation: the M2 conditioning ablation grid (kelp_v2.md).
#
# Six arms over ONE joint-dropout checkpoint (train_exp12.sh):
#
#   none          -- no conditioning (floor)
#   nl            -- task text only (the exp11-equivalent arm)
#   spec-holdout  -- asserts only, K-1 shown / K scored  } the headline pair:
#   both-holdout  -- NL + K-1 asserts, K scored          } anti-copying honest
#   both          -- NL + all asserts (system number; rerank partially circular)
#   oracle        -- clean program AS the spec (information ceiling: if repair
#                    doesn't move even here, the bottleneck is mechanical, not
#                    informational, and conditioning work should stop)
#
# Decision rules (docs/kelp_v2.md M2): spec/both-holdout beat nl with
# non-overlapping CIs -> M2 confirmed, proceed to M3. Flat but oracle lifts ->
# richer specs. Oracle flat -> mechanical bottleneck; pivot.
#
# Usage:
#   bash scripts/eval_exp12.sh              # dry-run all arms
#   bash scripts/eval_exp12.sh --submit     # launch on Iris

set -euo pipefail

PRESET="${PRESET:-tpu_vet}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-gs://marin-us-east5/kelp/checkpoints/exp12-spec}"
CORPUS_FILE="${CORPUS_FILE:-gs://marin-us-east5/kelp/corpus/curated_v3.txt}"
CORRUPTION_STEPS="${CORRUPTION_STEPS:-1}"
NUM_CORRUPTIONS="${NUM_CORRUPTIONS:-5}"
N_BEST_OF="${N_BEST_OF:-16}"
MAX_TASKS="${MAX_TASKS:-500}"
TEST_TIMEOUT="${TEST_TIMEOUT:-5}"
SEED="${SEED:-42}"
WANDB_PROJECT="${WANDB_PROJECT:-kelp}"
RUN_PREFIX="${RUN_PREFIX:-exp12}"
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

run_arm() {
    local tag="$1"; shift
    local launch_flags=(--preset "$PRESET" --name "kelp-eval-${RUN_PREFIX}-$tag")
    [ -n "$SUBMIT" ] && launch_flags+=("$SUBMIT")
    [ -n "$IMAGE" ] && launch_flags+=(--image "$IMAGE")
    [ -n "$PRIORITY_BAND" ] && launch_flags+=(--priority-band "$PRIORITY_BAND")

    echo "=== arm [$tag] (${SUBMIT:-dry-run}) ==="
    uv run kelp-eval "${launch_flags[@]}" -- \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --corpus-file "$CORPUS_FILE" \
        --num-corruptions "$NUM_CORRUPTIONS" \
        --corruption-steps "$CORRUPTION_STEPS" \
        --no-bank-swap-fallback \
        --n-best-of "$N_BEST_OF" \
        --max-tasks "$MAX_TASKS" \
        --test-timeout "$TEST_TIMEOUT" \
        --seed "$SEED" \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-run-name "${RUN_PREFIX}-$tag" \
        "$@"
}

run_arm none --conditioning none
run_arm nl --conditioning nl
run_arm spec-holdout --conditioning spec --spec-holdout
run_arm both-holdout --conditioning both --spec-holdout
run_arm both --conditioning both
run_arm oracle --conditioning both --spec-oracle
