#!/bin/bash
# Residual-routing ImageNet arm: the r-arm recipe (slurm 29384839, random init, vit_base, 300
# epochs) with --resid_routing true and nothing else changed. Submitted by routing/sbatch/chain.sh.
#SBATCH --job-name route
#SBATCH --partition alldlc2_gpu-h200
#SBATCH --nodes 1
#SBATCH --gres=gpu:4
#SBATCH --time 23:29:59
#SBATCH -o /work/dlc2workfs3/schrodi-procedural/kempfe/routing/logs/%x_%j.out
#SBATCH -e /work/dlc2workfs3/schrodi-procedural/kempfe/routing/logs/%x_%j.err
#SBATCH --mail-type END,FAIL

SECONDS=0
ROOT=${PROC_ROOT:-/work/dlc2workfs3/schrodi-procedural/kempfe/Procedural}
OUT_ROOT=${OUT_ROOT:-/work/dlc2workfs3/schrodi-procedural/kempfe/routing}

cd "$ROOT" || exit 1
umask 002
echo "Started at $(date)"
echo "Running $SLURM_JOB_NAME on $SLURM_GPUS_ON_NODE gpus, JID $SLURM_JOB_ID, queue $SLURM_JOB_PARTITION"

# the chain pins SLURM_ID to the FIRST job's id so every segment resumes into the same output dir
if [[ -z "$SLURM_ID" ]] || [[ "$SLURM_ID" -eq 0 ]]; then SLURM_ID=$SLURM_JOB_ID; fi
if [[ -z "$SEED" ]]; then SEED=0; fi
if [[ -z "$ROUTING_IMPL" ]]; then ROUTING_IMPL="buf"; fi   # chosen by the gate: +22% vs loop +124%
if [[ -z "$EPOCHS" ]]; then EPOCHS=300; fi
if [[ -z "$WARMUP" ]]; then WARMUP=50; fi
echo "SLURM_ID $SLURM_ID  seed $SEED  routing_impl $ROUTING_IMPL  epochs $EPOCHS"

source /work/dlc2workfs3/schrodi-procedural/kempfe/.venv/bin/activate
echo "cwd $(pwd)  commit $(git rev-parse --short HEAD)"
nvidia-smi
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

TOTAL_BATCH_SIZE=4096
BATCH_SIZE=128
UPDATE_FREQ=$((($TOTAL_BATCH_SIZE / $SLURM_GPUS_ON_NODE) / $BATCH_SIZE))

# --standalone pins rendezvous to localhost:29400 and deadlocks against any other torchrun on the
# node (docs 5.12); derive a unique port from the job id instead.
MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 20000) ))
echo "rendezvous port: $MASTER_PORT"

torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:$MASTER_PORT \
    --nproc_per_node=$SLURM_GPUS_ON_NODE main.py \
    --model vit_base --warmup_epochs $WARMUP --epochs $EPOCHS \
    --total_batch_size $TOTAL_BATCH_SIZE \
    --batch_size $BATCH_SIZE --lr 2e-3 --update_freq $UPDATE_FREQ --use_amp true \
    --data_path "/data/datasets/ILSVRC2012" \
    --data_set "IMNET" \
    --output_dir "$OUT_ROOT/runs/route_${SLURM_ID}/s$SEED" \
    --accuracy_json "$OUT_ROOT/runs/accuracy_route_${SLURM_ID}.json" \
    --enable_wandb true \
    --project "vit base routing" \
    --wandb_entity_name "procedural_pretraining" \
    --slurm_id $SLURM_ID \
    --seed $SEED \
    --resid_routing true \
    --routing_impl "$ROUTING_IMPL"
TORCH_EXIT=$?
echo "Torchrun exited with code $TORCH_EXIT"

duration=$SECONDS
if (( duration < 300 )); then
    echo "Runtime ${duration}s too short. Stop chain."
    exit 2
fi
exit $TORCH_EXIT
