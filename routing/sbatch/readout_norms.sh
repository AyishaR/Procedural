#!/bin/bash
# CLS-row norms and eval-only c / LN ablations of the finished readout arms (routing/readout_norms.py).
# Analysis job: alldlc2_gpu-h200 only. Submit from the repo root:
#   sbatch -J readout-norms routing/sbatch/readout_norms.sh
#SBATCH --partition alldlc2_gpu-h200
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 16
#SBATCH --mem 160G
#SBATCH --time 0-02:00
#SBATCH -o /work/dlc2workfs3/schrodi-procedural/kempfe/routing/logs/%x_%j.out
#SBATCH -e /work/dlc2workfs3/schrodi-procedural/kempfe/routing/logs/%x_%j.err
set -eo pipefail
umask 002
ROOT=${PROC_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
source /work/dlc2workfs3/schrodi-procedural/kempfe/.venv/bin/activate
cd "$ROOT"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} root $ROOT commit $(git rev-parse --short HEAD)"
python routing/readout_norms.py
