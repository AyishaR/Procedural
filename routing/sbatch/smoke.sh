#!/bin/bash
# Smoke + resume rehearsal for the routed arm.
#
# EPOCHS/WARMUP default to the REAL recipe (300/50) and the run is bounded by --time instead, so
# the LR trajectory is the one the actual run will see. Do not shorten it by lowering --epochs:
# utils.cosine_scheduler asserts len(schedule) == epochs * niter_per_ep, so warmup_epochs must be
# <= epochs, and a compressed warmup at lr 2e-3 diverges -- verified, and it diverges for the
# VANILLA arm too (job 29579323), so it is the schedule, not routing.
#
# The run is expected to end in TIMEOUT; re-submitting against the same OUT rehearses
# --auto_resume on a routed checkpoint, exactly as the chain driver does.
#SBATCH --job-name route-smoke
#SBATCH --partition alldlc2_gpu-h200
#SBATCH --nodes 1
#SBATCH --gres=gpu:4
#SBATCH --time 1:20:00
#SBATCH -o /work/dlc2workfs3/schrodi-procedural/kempfe/routing/logs/%x_%j.out
#SBATCH -e /work/dlc2workfs3/schrodi-procedural/kempfe/routing/logs/%x_%j.err
set -o pipefail
ROOT=${PROC_ROOT:-/work/dlc2workfs3/schrodi-procedural/kempfe/Procedural}
OUT=${OUT:-/work/dlc2workfs3/schrodi-procedural/kempfe/routing/smoke}
EPOCHS=${EPOCHS:-300}
WARMUP=${WARMUP:-50}     # must be <= EPOCHS (cosine_scheduler asserts on the length)
ROUTED=${ROUTED:-true}   # false = the vanilla control at the identical config
ROUTING_IMPL=${ROUTING_IMPL:-buf}
READOUT=${READOUT:-off}  # a --readout_routing mode replaces the routing flags (readout-only arms)
if [[ "$READOUT" != "off" ]]; then ARM_FLAGS=(--readout_routing "$READOUT"); else ARM_FLAGS=(--resid_routing "$ROUTED" --routing_impl "$ROUTING_IMPL"); fi
cd "$ROOT" || exit 1
umask 002
source /work/dlc2workfs3/schrodi-procedural/kempfe/.venv/bin/activate
echo "host $(hostname) job $SLURM_JOB_ID root $ROOT commit $(git rev-parse --short HEAD) epochs $EPOCHS warmup $WARMUP readout $READOUT routed $ROUTED impl $ROUTING_IMPL arm flags ${ARM_FLAGS[*]}"
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
    "${ARM_FLAGS[@]}"
echo "exit $?"
