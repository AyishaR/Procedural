#!/bin/bash
#SBATCH --job-name pr
#SBATCH --partition lmbhiwidlc_gpu-rtx2080
#SBATCH --nodes 1
#SBATCH --gres=gpu:4
#SBATCH --time 11:59:59
#SBATCH -o /home/dawooda/code/procedural/Procedural/logs/pr_%j_%x.out
#SBATCH -e /home/dawooda/code/procedural/Procedural/logs/pr_%j_%x.err # STDERR
#SBATCH --mail-type END,FAIL 
#SBATCH --exclude=dlcgpu02

ROOT='/home/dawooda/code/procedural'

cd $ROOT
echo "Started at $(date)";

echo "Running job $SLURM_JOB_NAME using $SLURM_GPUS_ON_NODE gpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

source ~/.bashrc
conda activate prvenv

cd Procedural
echo "Current working directory: $(pwd)";

nvidia-smi
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

TOTAL_BATCH_SIZE=256
BATCH_SIZE=8
UPDATE_FREQ=$((($TOTAL_BATCH_SIZE / $SLURM_GPUS_ON_NODE) / $BATCH_SIZE))

torchrun --standalone --nproc_per_node=$SLURM_GPUS_ON_NODE procedural.py \
    --model vit_base_patch16_224  --warmup_steps 1000 --training_steps 15000 \
    --save_every 2000 \
    --k 64 --procedural_data "kdyck_truncated" --p_open 0.6 --max_depth 4 \
    --procedural_order "standard" \
    --embeddings_path "kdyck/kdyck_orthogonal_embeddings_vitb.pt" \
    --total_batch_size $TOTAL_BATCH_SIZE \
    --batch_size $BATCH_SIZE --lr 2e-3 \
    --update_freq $UPDATE_FREQ \
    --output_dir "/home/dawooda/code/procedural/Procedural/results_pr_vitb" \
    --wandb_entity_name "procedural_pretraining" \
    --wandb_project_name "procedural_models" \
    --slurm_id $SLURM_JOB_ID \
    --freeze_patch_embeddings true \
    --freeze_pos_embeddings true

echo "DONE";
echo "Finished at $(date)";
