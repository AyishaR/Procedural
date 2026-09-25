#!/bin/bash
#SBATCH --job-name probe_dense
#SBATCH --partition alldlc2_gpu-h200
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 16
#SBATCH --mem 120G
#SBATCH --time 06:00:00
#SBATCH --array 0-2
#SBATCH -o /work/dlc2workfs3/schrodi-procedural/kempfe/probes/logs/%x_%A_%a.out
set -eo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/probes/sbatch/common.sh"   # submit from the repo root
DATASETS=(voc ade20k nyu)
python -m probes.dense_probes --datasets "${DATASETS[$SLURM_ARRAY_TASK_ID]}" --models "${MODELS:-all}" --workers 12
