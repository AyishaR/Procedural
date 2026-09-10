#!/bin/bash
# Pre-training gate for --resid_routing (routing/gate.py). Submit from the repo root:
#   sbatch -J route-gate routing/sbatch/gate.sh
#SBATCH --partition testdlc2_gpu-h200
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 16
#SBATCH --mem 120G
#SBATCH --time 0-00:55
#SBATCH -o /work/dlc2workfs3/schrodi-procedural/kempfe/routing/logs/%x_%j.out
#SBATCH -e /work/dlc2workfs3/schrodi-procedural/kempfe/routing/logs/%x_%j.err
set -eo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/probes/sbatch/common.sh" 2>/dev/null || {
  umask 002
  ROOT=${PROC_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
  VENV=${PROC_VENV:-/work/dlc2workfs3/schrodi-procedural/kempfe/.venv}
  source "$VENV/bin/activate"
  cd "$ROOT"
  echo "host $(hostname) job ${SLURM_JOB_ID:-none} root $ROOT venv $VENV commit $(git rev-parse --short HEAD)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
}
python routing/gate.py "$@"
