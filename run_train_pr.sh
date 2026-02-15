#!/bin/bash
#SBATCH --job-name pr
#SBATCH --partition lmbhiwidlc_gpu-rtx2080
#SBATCH --nodes 1
#SBATCH --gres=gpu:4
#SBATCH --time 23:59:59
#SBATCH -o /home/dawooda/code/procedural/Procedural/logs/pr_%j_%x.out
#SBATCH -e /home/dawooda/code/procedural/Procedural/logs/pr_%j_%x.err # STDERR
#SBATCH --mail-type END,FAIL 

ROOT='/home/dawooda/code/procedural'

cd $ROOT
echo "Started at $(date)";

echo "Running job $SLURM_JOB_NAME using $SLURM_GPUS_ON_NODE gpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

source ~/.bashrc
conda activate prvenv

cd Procedural
# print present working directory
echo "Current working directory: $(pwd)";

nvidia-smi
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

TOTAL_BATCH_SIZE=256
BATCH_SIZE=$(($TOTAL_BATCH_SIZE / $SLURM_GPUS_ON_NODE))

torchrun --standalone --nproc_per_node=$SLURM_GPUS_ON_NODE procedural.py \
    --model vit_tiny_patch16_224  --warmup_steps 1000 --training_steps 15000 \
    --k 64 --procedural_data "kdyck" --p_open 0.6 \
    --total_batch_size $TOTAL_BATCH_SIZE \
    --batch_size $BATCH_SIZE --lr 2e-3 \
    --output_dir "/home/dawooda/code/procedural/Procedural/results_pr64_freeze_$SLURM_JOB_ID" \
    --wandb_project_name "procedural_models" \
    --slurm_id $SLURM_JOB_ID \
    --freeze_patch_embeddings true \
    --freeze_pos_embeddings true

echo "DONE";
echo "Finished at $(date)";

# torchrun --standalone --nproc_per_node=1 procedural.py \
#     --model vit_tiny_patch16_224  --warmup_steps 1000 --training_steps 15000 \
#     --k 64 \
#     --total_batch_size 256 \
#     --batch_size 256 --lr 2e-3 \
#     --output_dir "/home/dawooda/code/procedural/Procedural/results_pr64" \
#     --wandb_project_name "procedural_models" \
#     --slurm_id 0 