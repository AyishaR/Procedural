#!/bin/bash
#SBATCH --job-name probe_train
#SBATCH --partition lmbdlc2_gpu-h200
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 8
#SBATCH --mem 120G
#SBATCH --time 06:00:00
#SBATCH --array 0-4
#SBATCH -o /work/dlc2workfs3/schrodi-procedural/kempfe/probes/logs/%x_%A_%a.out
# Array task i owns registry positions i, i+5, i+10, ... of the FIXED model list (registry order, or the
# MODELS=<name,...> override), then drops the ones that fail the final-checkpoint gate.  Partitioning before
# the gate keeps task ownership stable even if a model finishes training while the array is running.
set -eo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/probes/sbatch/common.sh"   # submit from the repo root
SEL=$(MODELS="${MODELS:-}" python -c "
import os
from probes import registry
names = [m['name'] for m in registry.MODELS] if not os.environ['MODELS'] else os.environ['MODELS'].split(',')
mine = names[int(os.environ['SLURM_ARRAY_TASK_ID'])::int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 5))]
print(','.join(n for n in mine if registry.finished(registry.BY_NAME[n])[0]))")
echo "models: $SEL"
[[ -z "$SEL" ]] && { echo "nothing to do"; exit 0; }
python -m probes.train_probes --models "$SEL" --datasets all --ftypes cls,mean,concat
python -m probes.knn_lowlevel --models "$SEL" --datasets all
