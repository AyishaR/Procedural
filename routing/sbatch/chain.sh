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
#   setsid nohup env SEED=0 ROUTING_IMPL=buf OUT_ROOT=... bash routing/sbatch/chain.sh > chain.log 2>&1 < /dev/null &
# Readout-only arms (one chain each, they run side by side):
#   setsid nohup env SEED=0 READOUT=fixed_last4 RUN_PREFIX=readout_A OUT_ROOT=... bash routing/sbatch/chain.sh > ... &
SLURM_ID=${SLURM_ID:-0}
SEED=${SEED:-0}
ROUTING_IMPL=${ROUTING_IMPL:-buf}   # chosen by the gate: +22% vs loop +124%
READOUT=${READOUT:-off}             # off = the full router; else a --readout_routing mode
if [[ "$READOUT" != "off" ]]; then RUN_PREFIX=${RUN_PREFIX:-readout_$READOUT}; else RUN_PREFIX=${RUN_PREFIX:-route}; fi
OUT_ROOT=${OUT_ROOT:-/work/dlc2workfs3/schrodi-procedural/kempfe/routing}
MAX_RETRIES=${MAX_RETRIES:-5}
SCRIPT="$(dirname "$(readlink -f "$0")")/train_route.sh"
# Partitions. Segments are submitted to the script's own #SBATCH partition (alldlc2), whose jobs
# get PREEMPTED by lmb jobs. After a preemption every later segment is WIDENED right after
# submission to PARTITION=$PREEMPT_PARTITION via scontrol (swrap rejects a comma list at submit
# time, scontrol does not -- see layer-scale/slurm/clip-lmb-mover.sh), so Slurm starts it in
# whichever partition has a slot first: lmb (PreemptMode=OFF) when its 13-GPU account cap has
# room, alldlc2 otherwise instead of waiting on MaxGRESPerAccount. PARTITION=<list> widens from
# the start.
PARTITION=${PARTITION:-}
PREEMPT_PARTITION=${PREEMPT_PARTITION:-alldlc2_gpu-h200,lmbdlc2_gpu-h200}
# WAIT_JOB: attach to a segment that is already running (a driver restarted with new settings)
# instead of submitting one; the loop continues normally when it ends. Needs SLURM_ID too.
WAIT_JOB=${WAIT_JOB:-}
umask 002
# pidfile so liveness is checked by pid, not by grepping process lists (which matches the
# grep itself). Launch with setsid so the driver is in its own session and survives the
# death of whatever shell started it -- the first driver died with its parent at epoch ~116
# and the preempted segment 2 was never resubmitted.
PIDFILE="$OUT_ROOT/logs/chain_${RUN_PREFIX}_s${SEED}.pid"
mkdir -p "$OUT_ROOT/logs"; echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

epochs_done() {   # lines in log.txt, 0 if it does not exist yet
    local log="$OUT_ROOT/runs/${RUN_PREFIX}_${SLURM_ID}/s$SEED/log.txt"
    [[ -f "$log" ]] && wc -l < "$log" || echo 0
}

finished() {      # main.py wrote the final epoch
    local log="$OUT_ROOT/runs/${RUN_PREFIX}_${SLURM_ID}/s$SEED/log.txt"
    [[ -f "$log" ]] && tail -1 "$log" | grep -q '"epoch": 299'
}

retries=0
while true; do
    before=$(epochs_done)
    if [[ -n "$WAIT_JOB" ]]; then
        JOB_ID=$WAIT_JOB; WAIT_JOB=""
        [[ "$SLURM_ID" -eq 0 ]] && { echo "WAIT_JOB needs SLURM_ID (the first job of the chain)"; exit 1; }
        echo "$(date) attaching to running job $JOB_ID (SLURM_ID=$SLURM_ID prefix=$RUN_PREFIX epochs_done=$before partition after preemption: $PREEMPT_PARTITION)"
    else
        EXPORT="ALL,SEED=$SEED,ROUTING_IMPL=$ROUTING_IMPL,READOUT=$READOUT,RUN_PREFIX=$RUN_PREFIX,OUT_ROOT=$OUT_ROOT"
        if [[ "$SLURM_ID" -ne 0 ]]; then EXPORT="$EXPORT,SLURM_ID=$SLURM_ID"; fi
        echo "$(date) submitting (SLURM_ID=$SLURM_ID seed=$SEED readout=$READOUT impl=$ROUTING_IMPL prefix=$RUN_PREFIX epochs_done=$before)"
        JOB_ID=$(sbatch --parsable -J "$RUN_PREFIX" --export="$EXPORT" "$SCRIPT" | awk '{print $1}' | tr -d ':')
        [[ -z "$JOB_ID" ]] && { echo "sbatch failed - stopping chain"; exit 1; }
        [[ "$SLURM_ID" -eq 0 ]] && SLURM_ID=$JOB_ID
        if [[ -n "$PARTITION" ]]; then
            if out=$(scontrol update JobId="$JOB_ID" Partition="$PARTITION" 2>&1); then
                echo "job $JOB_ID widened to Partition=$PARTITION"
            else
                echo "could not widen job $JOB_ID to $PARTITION ($out) - it stays on the script partition"
            fi
        fi
    fi

    mins=0
    while true; do
        STATE=$(sacct -j "$JOB_ID" --format=State --noheader --parsable2 2>/dev/null | head -1)
        [[ "$STATE" =~ ^(COMPLETED|TIMEOUT|FAILED|CANCELLED|NODE_FAIL|OUT_OF_ME|PREEMPTED|DEADLINE) ]] && break
        sleep 60; mins=$((mins + 1))
    done
    after=$(epochs_done)
    echo "$(date) job $JOB_ID ended $STATE after ${mins}min (epochs $before -> $after)"

    if finished; then echo "reached epoch 299 - chain complete"; exit 0; fi

    if [[ "$STATE" == PREEMPTED* && "$PARTITION" != "$PREEMPT_PARTITION" ]]; then
        PARTITION=$PREEMPT_PARTITION
        echo "preempted -> every later segment is widened to Partition=$PARTITION"
    fi
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
