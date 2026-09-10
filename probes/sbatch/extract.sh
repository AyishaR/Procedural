#!/bin/bash
#SBATCH --job-name probe_extract
#SBATCH --partition lmbdlc2_gpu-h200
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 16
#SBATCH --mem 120G
#SBATCH --time 04:00:00
#SBATCH --array 0-8
#SBATCH -o /work/dlc2workfs3/schrodi-procedural/kempfe/probes/logs/%x_%A_%a.out
# one array task per cached dataset; all models resident on the GPU.  MODELS=<name,...> overrides 'all'.
set -eo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/probes/sbatch/common.sh"   # submit from the repo root
DATASETS=(imnet places365 cifar100 dtd flowers cars aircraft eurosat imagenet_r)
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}
python -m probes.extract_features --datasets "$DS" --models "${MODELS:-all}" --batch_size 256 --workers 16
