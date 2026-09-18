#!/bin/bash
#SBATCH --job-name dyn_rawlens
#SBATCH --partition alldlc2_gpu-h200
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 16
#SBATCH --mem 120G
#SBATCH --time 03:00:00
#SBATCH --array 0-16
#SBATCH -o /work/dlc2workfs3/schrodi-procedural/kempfe/dynamics/logs/%x_%A_%a.out
# one array task per dynamics run (dynamics.registry.RUNS order); every stage skips finished (run, epoch) results,
# so a preempted array is simply resubmitted.  Submit from the worktree root.
set -eo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/dynamics/sbatch/common.sh"
RUN=$(run_name "$SLURM_ARRAY_TASK_ID")
echo "run $RUN"
python -m dynamics.rawlens --runs $RUN --files ${FILES:-dense} --init --workers 16
