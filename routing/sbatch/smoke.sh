#!/bin/bash
# Smoke + resume rehearsal for the routed arm: 5 epochs of the real recipe so that epoch 4 hits
# the analysis path ((epoch+1)%5==0), then re-submit this same script with EPOCHS=7 to exercise
# --auto_resume on a routed checkpoint.
#SBATCH --job-name route-smoke
#SBATCH --partition alldlc2_gpu-h200
#SBATCH --nodes 1
#SBATCH --gres=gpu:4
#SBATCH --time 2:00:00
#SBATCH -o /work/dlc2workfs3/schrodi-procedural/kempfe/routing/logs/%x_%j.out
#SBATCH -e /work/dlc2workfs3/schrodi-procedural/kempfe/routing/logs/%x_%j.err
set -o pipefail
ROOT=${PROC_ROOT:-/work/dlc2workfs3/schrodi-procedural/kempfe/Procedural}
OUT=${OUT:-/work/dlc2workfs3/schrodi-procedural/kempfe/routing/smoke}
EPOCHS=${EPOCHS:-5}
WARMUP=${WARMUP:-50}      # keep the real recipe: epochs 0-4 sit inside the LR ramp
ROUTED=${ROUTED:-true}   # false = the vanilla control at the identical config
ROUTING_IMPL=${ROUTING_IMPL:-buf}
cd "$ROOT" || exit 1
umask 002
source /work/dlc2workfs3/schrodi-procedural/kempfe/.venv/bin/activate
echo "host $(hostname) job $SLURM_JOB_ID root $ROOT commit $(git rev-parse --short HEAD) epochs $EPOCHS warmup $WARMUP routed $ROUTED impl $ROUTING_IMPL"
nvidia-smi --query-gpu=name --format=csv,noheader

TOTAL_BATCH_SIZE=4096
BATCH_SIZE=128
UPDATE_FREQ=$((($TOTAL_BATCH_SIZE / $SLURM_GPUS_ON_NODE) / $BATCH_SIZE))
MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 20000) ))

torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:$MASTER_PORT \
    --nproc_per_node=$SLURM_GPUS_ON_NODE main.py \
    --model vit_base --warmup_epochs $WARMUP --epochs $EPOCHS \
    --total_batch_size $TOTAL_BATCH_SIZE \
    --batch_size $BATCH_SIZE --lr 2e-3 --update_freq $UPDATE_FREQ --use_amp true \
    --data_path "/data/datasets/ILSVRC2012" \
    --data_set "IMNET" \
    --output_dir "$OUT/s0" \
    --accuracy_json "$OUT/accuracy_smoke.json" \
    --enable_wandb false \
    --slurm_id 1 \
    --seed 0 \
    --resid_routing "$ROUTED" \
    --routing_impl "$ROUTING_IMPL"
echo "exit $?"
