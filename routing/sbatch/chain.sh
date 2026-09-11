#!/bin/bash
# Chain driver for the routed ImageNet arm: resubmit on TIMEOUT so --auto_resume continues the
# same output dir. Run under nohup from the repo root:
#   nohup bash routing/sbatch/chain.sh > $OUT_ROOT/logs/chain.log 2>&1 &
SLURM_ID=${SLURM_ID:-0}
SEED=${SEED:-0}
ROUTING_IMPL=${ROUTING_IMPL:-buf}   # chosen by the gate: +22% vs loop +124%
OUT_ROOT=${OUT_ROOT:-/work/dlc2workfs3/schrodi-procedural/kempfe/routing}
SCRIPT="$(dirname "$(readlink -f "$0")")/train_route.sh"
umask 002

while true; do
    EXPORT="SEED=$SEED,ROUTING_IMPL=$ROUTING_IMPL,OUT_ROOT=$OUT_ROOT"
    if [[ "$SLURM_ID" -ne 0 ]]; then EXPORT="SLURM_ID=$SLURM_ID,$EXPORT"; fi
    echo "$(date) submitting (SLURM_ID=$SLURM_ID seed=$SEED impl=$ROUTING_IMPL)"
    JOB_ID=$(sbatch --parsable --export="$EXPORT" "$SCRIPT" | awk '{print $1}' | tr -d ':')
    [[ -z "$JOB_ID" ]] && { echo "sbatch failed - stopping chain"; exit 1; }
    [[ "$SLURM_ID" -eq 0 ]] && SLURM_ID=$JOB_ID

    mins=0
    while true; do
        STATE=$(sacct -j "$JOB_ID" --format=State --noheader --parsable2 2>/dev/null | head -1)
        if [[ "$STATE" =~ ^(COMPLETED|TIMEOUT) ]]; then
            if (( mins <= 10 )); then echo "job $JOB_ID ended after ${mins}min - stopping"; exit 1; fi
            break
        elif [[ "$STATE" =~ ^(FAILED|CANCELLED|NODE_FAIL|OUT_OF_ME) ]]; then
            echo "job $JOB_ID $STATE - stopping chain"; exit 1
        fi
        sleep 60; mins=$((mins + 1))
    done
    echo "$(date) job $JOB_ID finished ($STATE) after ${mins}min"

    # done when main.py wrote the final epoch
    LOG="$OUT_ROOT/runs/route_${SLURM_ID}/s$SEED/log.txt"
    if [[ -f "$LOG" ]] && tail -1 "$LOG" | grep -q '"epoch": 299'; then
        echo "reached epoch 299 - chain complete"; exit 0
    fi
done
