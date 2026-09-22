#!/bin/bash
#SBATCH --job-name probe_ood
#SBATCH --partition alldlc2_gpu-h200
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 16
#SBATCH --mem 60G
#SBATCH --time 01:30:00
#SBATCH --array 0-20
#SBATCH -o /work/dlc2workfs3/schrodi-procedural/kempfe/probes/logs/%x_%A_%a.out
# one array task per final checkpoint (registry.MODELS order); zero-shot V2 / R / C with the model's own head.
# MODELS=<name,...> overrides the array mapping (submit with --array=0, the directive above is not removable
# from the command line).  Skips models whose JSON exists.
set -eo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/probes/sbatch/common.sh"   # submit from the repo root
if [[ -z "${MODELS:-}" ]]; then
  MODELS=$(python -c "from probes import registry; print(registry.MODELS[${SLURM_ARRAY_TASK_ID:-0}]['name'])")
elif [[ "${SLURM_ARRAY_TASK_COUNT:-1}" -gt 1 ]]; then
  echo "MODELS override needs a single-task array (sbatch --array=0); got ${SLURM_ARRAY_TASK_COUNT} tasks" >&2; exit 2
fi
python -m probes.ood_eval --models "$MODELS" --batch_size 256 --workers 16
