#!/bin/bash
#SBATCH --job-name probe_lens
#SBATCH --partition lmbdlc2_gpu-h200
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 8
#SBATCH --mem 80G
#SBATCH --time 03:00:00
#SBATCH -o /work/dlc2workfs3/schrodi-procedural/kempfe/probes/logs/%x_%j.out
set -eo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/probes/sbatch/common.sh"   # submit from the repo root
python -m probes.lens --models "${MODELS:-all}"
