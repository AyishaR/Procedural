#!/bin/bash
#SBATCH --job-name dyn_canonical
#SBATCH --partition alldlc2_gpu-h200
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 16
#SBATCH --mem 120G
#SBATCH --time 03:00:00
#SBATCH --array 0-11
#SBATCH -o /work/dlc2workfs3/schrodi-procedural/kempfe/dynamics/logs/%x_%A_%a.out
# Endpoint pass on the canonical r / p / ftb3b / ftbrho seeds (only dumps 298-299 survive): raw lens, bypass,
# gradients on both files and the single 298->299 function-change pair (fp32 checkpoints + fp16 variant).
# Bridges the substitute runs to the canonical arms beyond the historical lens.  Submit from the worktree root.
set -eo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/dynamics/sbatch/common.sh"
RUN=$(python -c "from dynamics import registry as r; print(r.HISTORY_RUNS[int('$SLURM_ARRAY_TASK_ID')])")
echo "run $RUN"
python -m dynamics.rawlens --runs $RUN --files avail --workers 16
python -m dynamics.bypass --runs $RUN --files avail --workers 16
python -m dynamics.grads --runs $RUN --files avail
python -m dynamics.funcchange --runs $RUN --pairs all --floor
