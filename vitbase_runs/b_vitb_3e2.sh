#!/bin/bash
# Seeds 1 and 2 for the a2 arm (random 0-8 + pr 9-11, blocks 9-11 downscaled to the random
# model's ratios). Reuses run_train_ftb3e.sh unchanged - that script already IS the a2
# config, and produced seed 0 as job 29407014.
#
# Why: a2 is the one IN-1k arm that does not fit. It scores 76.99, i.e. 1.17 BELOW the
# random baseline (78.16 +/- 0.22), while the same arm on IN-100 sat +0.21 ABOVE random.
# Single seed, no siblings, so these two seeds decide whether that reversal is real.
#
# SLURM_ID is pinned to 29407014 so the new seeds land in
# results/imnet_base/results_IMNET_BASE_29407014/s{1,2} beside the existing s0.
#
# Partition and --requeue are passed as sbatch flags rather than edited into
# run_train_ftb3e.sh, so that script stays untouched for the runs that used it.
# CLI flags override the #SBATCH lines inside the script.

SLURM_ID=29407014

cd "$(dirname "$0")" || exit 1

SCRIPT="run_train_ftb3e.sh"
PARTITION="alldlc2_gpu-h200"

if [[ ! -f "$SCRIPT" ]]; then
    echo "missing $SCRIPT"; exit 1
fi

for i in 2; do    # seeds; seed 0 already exists as job 29407014
    timeout_flag=0
    echo "Starting chain for seed $i with SLURM_ID=$SLURM_ID"
    while true; do
        echo "Submitting job with SLURM_ID=$SLURM_ID..."
        JOB_ID=$(sbatch --parsable \
            --partition "$PARTITION" \
            --requeue \
            --job-name "ftb3es$i" \
            --export=SLURM_ID=$SLURM_ID,SEED=$i \
            $SCRIPT | awk '{print $1}' | tr -d ':')

        if [[ -z "$JOB_ID" ]]; then
            echo "sbatch returned no job id - stopping chain"
            exit 1
        fi

        count_mins=0
        while true; do
            JOB_STATE=$(sacct -j $JOB_ID --format=State --noheader --parsable2 2>/dev/null | tail -1)
            EXIT_CODE=$(sacct -j $JOB_ID --format=ExitCode --noheader --parsable2 --brief 2>/dev/null | tail -1)

            echo "Checked sacct for job $JOB_ID, state: $JOB_STATE, exit code: $EXIT_CODE, mins total: $count_mins"

            if [[ "$JOB_STATE" =~ ^(COMPLETED|TIMEOUT)$ ]]; then
                if (( $count_mins <= 10 )); then
                    echo "Job $JOB_ID COMPLETED ran for less than $count_mins minutes - stopping chain"
                    timeout_flag=1
                fi
                break
            elif [[ "$JOB_STATE" =~ ^(RUNNING)$ ]]; then
                count_mins=$((count_mins + 1))
                echo "Job $JOB_ID is still running. Waiting..."
            elif [[ "$JOB_STATE" =~ ^(PREEMPTED|REQUEUED|RESIZING)$ ]]; then
                # --requeue puts the job straight back in the queue; auto_resume continues
                # from the last epoch checkpoint, so keep waiting rather than stopping.
                echo "Job $JOB_ID $JOB_STATE - requeued, waiting..."
            elif [[ "$JOB_STATE" =~ ^(FAILED|CANCELLED)$ ]]; then
                echo "Job $JOB_ID $JOB_STATE - stopping chain"
                exit 1
            else
                echo "Job $JOB_ID state: $JOB_STATE (unknown, waiting...)"
            fi

            sleep 10
        done
        if [[ "$timeout_flag" -eq 1 ]]; then
            echo "Job $JOB_ID timed out too quickly, moving to next seed..."
            break
        fi
    done
done
