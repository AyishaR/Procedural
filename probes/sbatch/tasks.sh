#!/bin/bash
#SBATCH --job-name probe_tasks
#SBATCH --partition alldlc2_gpu-h200
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 16
#SBATCH --mem 120G
#SBATCH --time 06:00:00
#SBATCH --array 0-1
#SBATCH -o /work/dlc2workfs3/schrodi-procedural/kempfe/probes/logs/%x_%A_%a.out
# task 0 = LOST object discovery (VOC12 trainval), task 1 = SPair-71k semantic correspondence.
# Both are training-free readouts of the frozen patch tokens at all 13 depths; every model stays resident on
# the GPU, so one pass covers MODELS=<name,...> (default: every model that passes the final-checkpoint gate).
set -eo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/probes/sbatch/common.sh"   # submit from the repo root
case $SLURM_ARRAY_TASK_ID in
  0) python -m probes.discovery --models "${MODELS:-all}" --batch_size 64 --workers 12 ;;
  1) python -m probes.correspondence --models "${MODELS:-all}" --batch_size 32 --workers 12 ;;
esac
