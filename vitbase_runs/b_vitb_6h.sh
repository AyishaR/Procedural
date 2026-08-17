#!/bin/bash
SLURM_ID=0
ASSIGNED=false

cd Procedural

SCRIPT="run_train_ftb6h.sh"

for i in 0; do
    timeout_flag=0
    echo "Starting chain for seed $i with SLURM_ID=$SLURM_ID"
    while true; do
        # Submit with all remaining args passed through
        if [[ "$SLURM_ID" -eq 0 ]]; then
            echo "Submitting initial job..."
            JOB_ID=$(sbatch --parsable  \
                --export=SEED=$i \
                $SCRIPT | awk '{print $1}' | tr -d ':')
            ASSIGNED=true
            SLURM_ID=$JOB_ID
        else
            echo "Submitting job with SLURM_ID=$SLURM_ID..."
            JOB_ID=$(sbatch --parsable  \
                --export=SLURM_ID=$SLURM_ID,SEED=$i \
                $SCRIPT | awk '{print $1}' | tr -d ':')
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
            elif [[ "$JOB_STATE" =~ ^(FAILED|CANCELLED)$ ]]; then
                echo "Job $JOB_ID $JOB_STATE - stopping chain"
                exit 1
            else
                echo "Job $JOB_ID state: $JOB_STATE (unknown, waiting...)"
            fi
                
            # if ! squeue -j $JOB_ID -h >/dev/null 2>&1; then
            #     # Job not in queue, check sacct
                
            #     if [[ "$JOB_STATE" == "COMPLETED" ]]; then
            #         EXIT_CODE=$(sacct -j $JOB_ID --format=ExitCode --noheader --parsable2 2>/dev/null | tail -1 | cut -d: -f1)
            #         echo "Job $JOB_ID COMPLETED with exit code $EXIT_CODE"

            #         if [[ "$EXIT_CODE" != "0" ]]; then
            #             echo "Job $JOB_ID COMPLETED but exit code $EXIT_CODE - stopping chain"
            #             exit 1
            #         fi
                    
            #         echo "Job $JOB_ID COMPLETED (exit 0)"
            #         break
            #     elif [[ "$JOB_STATE" =~ ^(FAILED|TIMEOUT|CANCELLED)$ ]]; then
            #         echo "Job $JOB_ID $JOB_STATE - stopping chain"
            #         exit 1
            #     else
            #         echo "Job $JOB_ID state: $JOB_STATE (unknown, waiting...)"
            #     fi
            # else
            #     echo "Job $JOB_ID still in queue (running/pending). Waiting..."
            # fi
            sleep 60
        done
        if [[ "$timeout_flag" -eq 1 ]]; then
            echo "Job $JOB_ID timed out too quickly, moving to next seed..."
            break
        fi
    done
done
