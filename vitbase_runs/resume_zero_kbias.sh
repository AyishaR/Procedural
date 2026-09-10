#!/bin/bash
#SBATCH --partition alldlc2_gpu-h200
#SBATCH --requeue
#SBATCH --nodes 1
#SBATCH --gres=gpu:4
#SBATCH --time 23:29:59
#SBATCH -o /home/schrodi/Procedural/logs/ft_%j_%x.out
#SBATCH -e /home/schrodi/Procedural/logs/ft_%j_%x.err # STDERR
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user schrodi@cs.uni-freiburg.de

# Resume wrapper: zero the attention k-bias in the latest checkpoint of the run, then hand over
# to the arm's normal run script (same SBATCH settings as the run scripts; the job name must be
# the arm, e.g. `--job-name ftbqmlnvot`, because the run scripts key wandb/log names on it).
#
#   sbatch --job-name ftbqmlnvot --partition=lmbdlc2_gpu-l40s --gres=gpu:8 \
#          --dependency=afterany:<running job> --export=SLURM_ID=<results id>,SEED=<s> \
#          vitbase_runs/resume_zero_kbias.sh
#
# Why: the k-bias is a softmax null direction (q.(k+b) = q.k + const per query), so nothing pulls
# it back and it random-walks; at |b_k| ~ 50-65 in blocks 9-10 the fp16 q.k product overflows and
# the run dies with a non-finite loss at the same iteration on every resume (ftbrhos s0 at epoch
# 272 and ftblrm s0 at epoch 228, 2026-09-09). Zeroing is an exact no-op for the forward pass;
# plots/verify/zero_kbias.py keeps the original as checkpoint-N.pth.orig_kbias.

ROOT='/home/schrodi/Procedural'
cd $ROOT
ARM=${ARM:-$SLURM_JOB_NAME}
SEED=${SEED:-0}
if [[ -z "$SLURM_ID" ]]; then echo "SLURM_ID (results id) must be exported"; exit 1; fi

d=results/imnet_base/results_IMNET_BASE_${SLURM_ID}/s${SEED}
latest=$(ls $d 2>/dev/null | grep -E '^checkpoint-[0-9]+\.pth$' | sed -E 's/checkpoint-([0-9]+)\.pth/\1/' | sort -n | tail -1)
if [[ -n "$latest" ]]; then
    echo "resume_zero_kbias: zeroing k-bias in $d/checkpoint-$latest.pth"
    .venv/bin/python plots/verify/zero_kbias.py "$d/checkpoint-$latest.pth" --inplace || { echo "zero_kbias failed"; exit 1; }
else
    echo "resume_zero_kbias: no checkpoint in $d, starting the run script unchanged"
fi

export SLURM_ID SEED
exec bash vitbase_runs/run_train_${ARM}.sh
