#!/bin/bash
# Move a running chain segment to another partition with minimal loss: this job is submitted to the
# target partition while the old segment keeps training; once it starts it waits for the old
# segment to finish an epoch (log.txt gains a line, which main.py writes AFTER checkpoint-N.pth,
# the analysis dump and the eval), cancels it within that ~4 min save-free window, waits until it
# has left the queue (the two must never write the same output dir), then runs train_route.sh,
# whose --auto_resume continues from checkpoint-N.pth. Loss: the restart only.
#
#   sbatch -p lmbdlc2_gpu-h200 -J readout_G --export=ALL,OLD_JOB=<running segment>,SLURM_ID=<chain id>,\
#       SEED=0,READOUT=<mode>,RUN_PREFIX=readout_G,OUT_ROOT=...,PROC_ROOT=<run clone> routing/sbatch/handover.sh
# Then attach a chain driver: WAIT_JOB=<this job> SLURM_ID=<chain id> PARTITION=... chain.sh
#SBATCH --job-name handover
#SBATCH --partition lmbdlc2_gpu-h200
#SBATCH --nodes 1
#SBATCH --gres=gpu:4
#SBATCH --time 23:29:59
#SBATCH -o /work/dlc2workfs3/schrodi-procedural/kempfe/routing/logs/%x_%j.out
#SBATCH -e /work/dlc2workfs3/schrodi-procedural/kempfe/routing/logs/%x_%j.err
for v in OLD_JOB SLURM_ID RUN_PREFIX PROC_ROOT; do
    [[ -z "${!v}" ]] && { echo "$v must be exported"; exit 1; }
done
OUT_ROOT=${OUT_ROOT:-/work/dlc2workfs3/schrodi-procedural/kempfe/routing}
SEED=${SEED:-0}
LOG="$OUT_ROOT/runs/${RUN_PREFIX}_${SLURM_ID}/s$SEED/log.txt"
[[ -f "$LOG" ]] || { echo "no $LOG - refusing"; exit 1; }
old_state() { squeue -h -j "$OLD_JOB" -o %T 2>/dev/null; }

echo "$(date) handover job $SLURM_JOB_ID on $(hostname): old segment $OLD_JOB is '$(old_state)', log $(wc -l < "$LOG") lines"
if [[ "$(old_state)" == "RUNNING" ]]; then
    n0=$(wc -l < "$LOG")
    for _ in $(seq 1 90); do                 # an epoch is ~4.5 min; give it up to 7.5 min
        [[ $(wc -l < "$LOG") -gt $n0 ]] && break
        [[ "$(old_state)" != "RUNNING" ]] && break
        sleep 5
    done
    echo "$(date) epoch boundary: log $(wc -l < "$LOG") lines (was $n0); cancelling $OLD_JOB"
fi
scancel "$OLD_JOB"
for _ in $(seq 1 120); do
    [[ -z "$(old_state)" ]] && break
    sleep 5
done
if [[ -n "$(old_state)" ]]; then echo "old segment still '$(old_state)' after 10 min - refusing to start"; exit 1; fi
echo "$(date) old segment gone ($(sacct -j "$OLD_JOB" -X -n -o State | tr -d ' ')); resuming, log $(wc -l < "$LOG") lines"
export SLURM_ID SEED OUT_ROOT RUN_PREFIX
exec bash "$PROC_ROOT/routing/sbatch/train_route.sh"
