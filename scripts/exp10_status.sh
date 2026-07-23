#!/usr/bin/env bash
# One-shot exp10 status. Run on demand (cheap) instead of a polling loop:
#   bash scripts/exp10_status.sh
#
# Prints each run's job state, preemption count, and latest logged metrics
# (loss/acc + throughput/MFU). The training jobs are self-sufficient -- interactive
# band, checkpoint every 2000 steps, auto-resume from checkpoint on preemption --
# so they complete without any babysitting; this just reports where they are.

set -euo pipefail

CLUSTER="${CLUSTER:-marin}"
JOBS=(
    "/alxmrs/kelp-tpu_vet_300m"   # ~305M
    "/alxmrs/kelp-tpu_vet"        # 115M control
)

for J in "${JOBS[@]}"; do
    echo "=== $J ==="
    uv run iris --cluster="$CLUSTER" job summary "$J" 2>/dev/null \
        | grep -E "^State:|^     0 " | head -2 || echo "  (no summary)"
    uv run iris --cluster="$CLUSTER" job logs "$J" --max-lines 4000 2>/dev/null \
        | grep -E "step=[0-9]+ (loss=|steps/s=)" | tail -2 || echo "  (no metric lines yet)"
    echo
done

echo "Checkpoints written so far (latest is what an eval would load):"
for D in exp10-300m exp10-115m-control; do
    echo "  gs://marin-us-east5/kelp/checkpoints/$D :"
    gcloud storage ls "gs://marin-us-east5/kelp/checkpoints/$D/" 2>/dev/null \
        | grep -oE "step-[0-9]+" | sort -u | tail -3 | sed 's/^/    /' || echo "    (none / gcloud unavailable)"
done
