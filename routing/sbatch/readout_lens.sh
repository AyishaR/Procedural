#!/bin/bash
# Logit lens over depth (stream + window lens) for readout and reference arms (routing/readout_lens.py).
# Analysis job: alldlc2_gpu-h200 only. Submit from the repo root:
#   sbatch -J readout-lens routing/sbatch/readout_lens.sh
#SBATCH --partition alldlc2_gpu-h200
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 16
#SBATCH --mem 160G
#SBATCH --time 0-02:30
#SBATCH -o /work/dlc2workfs3/schrodi-procedural/kempfe/routing/logs/%x_%j.out
#SBATCH -e /work/dlc2workfs3/schrodi-procedural/kempfe/routing/logs/%x_%j.err
set -eo pipefail
umask 002
ROOT=${PROC_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
source /work/dlc2workfs3/schrodi-procedural/kempfe/.venv/bin/activate
cd "$ROOT"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} root $ROOT commit $(git rev-parse --short HEAD)"
python routing/readout_lens.py
