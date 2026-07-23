#!/usr/bin/env bash
# Kelp exp11 training: restore multi-step diffusion (issue #143, kelp_v2.md M1).
#
# exp10 trained on inputs exactly ONE edit from clean (--max-corruption-steps 1),
# which collapsed the input state distribution while inference remained a
# 10-step iterative loop -- after the model's first imperfect edit, every state
# is off-distribution. The tree-diffusion paper's recipe (Sec 3.3) is the
# opposite: corrupt with s ~ Uniform[1, S] mutations, supervise a random step of
# the reverse path. That machinery is already in training/generation.py; this
# runbook just turns it back on:
#
#   1. --max-corruption-steps 3 (uniform 1..3 per example): input states span
#      1-3 edits from clean, targets remain single path-steps.
#   2. --p-random now EXPLICIT (0.2, the paper's rho-mixture): 20% of examples
#      are random-program -> clean long-range paths. Every prior run silently
#      used 0.2; exp11 makes it a recorded, ablatable choice.
#
# Everything else is held at exp10 values (realistic-or-drop, prompt
# conditioning, curated_v2, batch 64, LR 3e-4, 50K steps, 115M tpu_vet) so the
# exp10-115m-control -> exp11 delta isolates {input distribution}. Evaluate with
# scripts/eval_deconfound.sh (steps 1,2,3 cells) -- exit criterion: >= exp10 at
# steps=1, strictly better at steps>=2. Kill criterion: no difference anywhere
# => iterative denoising adds nothing over one-shot repair at this scale.
#
# Usage:
#   bash scripts/train_exp11.sh            # dry-run
#   bash scripts/train_exp11.sh --submit   # launch on Iris

set -euo pipefail

PRESET="${PRESET:-tpu_vet}"
CORPUS_FILE="${CORPUS_FILE:-gs://marin-us-east5/kelp/corpus/curated_v2.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-gs://marin-us-east5/kelp/checkpoints/exp11-multistep}"
STEPS="${STEPS:-50000}"
MAX_CORRUPTION_STEPS="${MAX_CORRUPTION_STEPS:-3}"
CORRUPTION_CURRICULUM="${CORRUPTION_CURRICULUM:-constant}"
P_NEAR_MISS="${P_NEAR_MISS:-1.0}"
P_PROMPT="${P_PROMPT:-0.5}"
P_RANDOM="${P_RANDOM:-0.2}"
# 2000 (not 5000): what the exp10 runs actually used after preemption thrash.
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-2000}"
SEED="${SEED:-42}"
WANDB_PROJECT="${WANDB_PROJECT:-kelp}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-exp11-multistep}"
IMAGE="${IMAGE:-}"
PRIORITY_BAND="${PRIORITY_BAND:-}"

SUBMIT=""
for arg in "$@"; do
    case "$arg" in
        --submit) SUBMIT="--submit" ;;
    esac
done

LAUNCH_FLAGS=(--preset "$PRESET")
[ -n "$SUBMIT" ] && LAUNCH_FLAGS+=("$SUBMIT")
[ -n "$IMAGE" ] && LAUNCH_FLAGS+=(--image "$IMAGE")
[ -n "$PRIORITY_BAND" ] && LAUNCH_FLAGS+=(--priority-band "$PRIORITY_BAND")

if [ -n "$SUBMIT" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY is unset -- the run will NOT log to W&B." >&2
fi

echo "=== exp11: ${SUBMIT:-dry-run} (preset=$PRESET, max_corruption_steps=$MAX_CORRUPTION_STEPS, p_random=$P_RANDOM, run=$WANDB_RUN_NAME) ==="
uv run kelp-launch "${LAUNCH_FLAGS[@]}" \
    -- \
    --corpus-file "$CORPUS_FILE" \
    --steps "$STEPS" \
    --augment \
    --prompt-conditioning \
    --p-prompt "$P_PROMPT" \
    --p-near-miss "$P_NEAR_MISS" \
    --p-random "$P_RANDOM" \
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
