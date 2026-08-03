#!/usr/bin/env bash
# Kelp exp12 training: spec conditioning via joint dropout (kelp_v2.md M2).
#
# The v11 grid measured every process-side lever flat (capacity, corruption
# recipe, corpus realism) while training loss saturates -- conditioning is the
# only untested lever. exp12 trains ONE model with independent prompt/spec
# dropout (p_prompt=0.5, p_spec=0.5) so the {none, NL, spec, NL+spec} ablation
# is an EVAL-TIME grid over a single checkpoint (see eval_exp12.sh). exp11 is
# the free NL-only-trained control.
#
# Corpus: curated_v3 -- 5,687 standalone Stack Edu functions, 100% docstring'd,
# corruptible, and spec'd (sandbox-validated assert sidecar). curated_v2 FAILED
# the spec-coverage gate (5.5%: 58% methods); do not use it here.
# Recipe otherwise matches exp11 (multi-step S=3, reverse-path targets,
# p_random=0.2, realistic-or-drop).
#
# Usage:
#   bash scripts/train_exp12.sh            # dry-run
#   bash scripts/train_exp12.sh --submit   # launch on Iris

set -euo pipefail

PRESET="${PRESET:-tpu_vet}"
CORPUS_FILE="${CORPUS_FILE:-gs://marin-us-east5/kelp/corpus/curated_v3.txt}"
SPEC_FILE="${SPEC_FILE:-gs://marin-us-east5/kelp/corpus/curated_v3_specs.jsonl}"
# Precomputed augmented bank (kelp.cli.build_bank): skips the minutes-long
# e-graph startup that cost the first two exp12 submissions 29 attempts.
BANK_FILE="${BANK_FILE:-gs://marin-us-east5/kelp/corpus/curated_v3_bank.json.gz}"
OUTPUT_DIR="${OUTPUT_DIR:-gs://marin-us-east5/kelp/checkpoints/exp12-spec}"
STEPS="${STEPS:-50000}"
MAX_CORRUPTION_STEPS="${MAX_CORRUPTION_STEPS:-3}"
P_NEAR_MISS="${P_NEAR_MISS:-1.0}"
P_PROMPT="${P_PROMPT:-0.5}"
P_SPEC="${P_SPEC:-0.5}"
P_RANDOM="${P_RANDOM:-0.2}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
SEED="${SEED:-42}"
WANDB_PROJECT="${WANDB_PROJECT:-kelp}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-exp12-spec}"
IMAGE="${IMAGE:-}"
PRIORITY_BAND="${PRIORITY_BAND:-}"

SUBMIT=""
for arg in "$@"; do
    case "$arg" in
        --submit) SUBMIT="--submit" ;;
    esac
done

LAUNCH_FLAGS=(--preset "$PRESET" --name "kelp-exp12-spec")
[ -n "$SUBMIT" ] && LAUNCH_FLAGS+=("$SUBMIT")
[ -n "$IMAGE" ] && LAUNCH_FLAGS+=(--image "$IMAGE")
[ -n "$PRIORITY_BAND" ] && LAUNCH_FLAGS+=(--priority-band "$PRIORITY_BAND")

if [ -n "$SUBMIT" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY is unset -- the run will NOT log to W&B." >&2
fi

echo "=== exp12: ${SUBMIT:-dry-run} (corpus=curated_v3, p_prompt=$P_PROMPT, p_spec=$P_SPEC, run=$WANDB_RUN_NAME) ==="
uv run kelp-launch "${LAUNCH_FLAGS[@]}" \
    -- \
    --corpus-file "$CORPUS_FILE" \
    --spec-file "$SPEC_FILE" \
    --bank-file "$BANK_FILE" \
    --steps "$STEPS" \
    --augment \
    --prompt-conditioning \
    --spec-conditioning \
    --p-prompt "$P_PROMPT" \
    --p-spec "$P_SPEC" \
    --p-near-miss "$P_NEAR_MISS" \
    --p-random "$P_RANDOM" \
    --no-bank-swap-fallback \
    --max-corruption-steps "$MAX_CORRUPTION_STEPS" \
    --corruption-curriculum constant \
    --data-loader streaming \
    --output-dir "$OUTPUT_DIR" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --log-interval 10 \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$WANDB_RUN_NAME" \
    --seed "$SEED"
