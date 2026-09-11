#!/bin/bash
# Chain driver for the routed ImageNet arm: resubmit on TIMEOUT so --auto_resume continues the
# same output dir, and survive TRANSIENT node failures.
#
# A 300-epoch run on this cluster does not get 30 uninterrupted hours: job 29590809 died at epoch
# 21 with a /dev/shm allocation failure inside a DataLoader worker (nothing to do with the model).
# So FAILED is only fatal when the segment made NO progress -- that is the signature of a real
# bug, e.g. a config the run rejects at startup. A segment that logged epochs and then died is
# retried, up to MAX_RETRIES consecutive times.
#
# Run under nohup from the repo root:
#   nohup env SEED=0 ROUTING_IMPL=buf OUT_ROOT=... bash routing/sbatch/chain.sh > chain.log 2>&1 &
SLURM_ID=${SLURM_ID:-0}
SEED=${SEED:-0}
ROUTING_IMPL=${ROUTING_IMPL:-buf}   # chosen by the gate: +22% vs loop +124%
OUT_ROOT=${OUT_ROOT:-/work/dlc2workfs3/schrodi-procedural/kempfe/routing}
MAX_RETRIES=${MAX_RETRIES:-5}
SCRIPT="$(dirname "$(readlink -f "$0")")/train_route.sh"
umask 002

epochs_done() {   # lines in log.txt, 0 if it does not exist yet
    local log="$OUT_ROOT/runs/route_${SLURM_ID}/s$SEED/log.txt"
    [[ -f "$log" ]] && wc -l < "$log" || echo 0
}

finished() {      # main.py wrote the final epoch
    local log="$OUT_ROOT/runs/route_${SLURM_ID}/s$SEED/log.txt"
    [[ -f "$log" ]] && tail -1 "$log" | grep -q '"epoch": 299'
}

retries=0
while true; do
    before=$(epochs_done)
    EXPORT="ALL,SEED=$SEED,ROUTING_IMPL=$ROUTING_IMPL,OUT_ROOT=$OUT_ROOT"
    if [[ "$SLURM_ID" -ne 0 ]]; then EXPORT="$EXPORT,SLURM_ID=$SLURM_ID"; fi
    echo "$(date) submitting (SLURM_ID=$SLURM_ID seed=$SEED impl=$ROUTING_IMPL epochs_done=$before)"
    JOB_ID=$(sbatch --parsable --export="$EXPORT" "$SCRIPT" | awk '{print $1}' | tr -d ':')
    [[ -z "$JOB_ID" ]] && { echo "sbatch failed - stopping chain"; exit 1; }
    [[ "$SLURM_ID" -eq 0 ]] && SLURM_ID=$JOB_ID

    mins=0
    while true; do
        STATE=$(sacct -j "$JOB_ID" --format=State --noheader --parsable2 2>/dev/null | head -1)
        [[ "$STATE" =~ ^(COMPLETED|TIMEOUT|FAILED|CANCELLED|NODE_FAIL|OUT_OF_ME|PREEMPTED|DEADLINE) ]] && break
        sleep 60; mins=$((mins + 1))
    done
    after=$(epochs_done)
    echo "$(date) job $JOB_ID ended $STATE after ${mins}min (epochs $before -> $after)"

    if finished; then echo "reached epoch 299 - chain complete"; exit 0; fi

    case "$STATE" in
        COMPLETED|TIMEOUT)
            retries=0 ;;
        *)
            if [[ "$after" -gt "$before" ]]; then
                retries=$((retries + 1))
                if (( retries > MAX_RETRIES )); then
                    echo "$STATE $retries times in a row despite progress - stopping, investigate"; exit 1
                fi
                echo "$STATE after making progress -> treating as transient, retry $retries/$MAX_RETRIES"
            else
                echo "$STATE with no progress at all - a real failure, stopping chain"; exit 1
            fi ;;
    esac
done
