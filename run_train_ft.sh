#!/bin/bash
#SBATCH --job-name ft
#SBATCH --partition lmbhiwi_gpu-rtx2080
#SBATCH --nodes 1
#SBATCH --gres=gpu:4
#SBATCH --time 02:59:59
#SBATCH -o /home/dawooda/code/procedural/Procedural/logs/ft_%j_%x.out
#SBATCH -e /home/dawooda/code/procedural/Procedural/logs/ft_%j_%x.err # STDERR
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

TOTAL_BATCH_SIZE=512
UPDATE_FREQ=1
BATCH_SIZE=$((($TOTAL_BATCH_SIZE / $SLURM_GPUS_ON_NODE) / $UPDATE_FREQ))

# $OUTPUT_DIR="/home/dawooda/code/procedural/Procedural/results_ft64_freeze_512_$SLURM_JOB_ID"
# mkdir -p $OUTPUT_DIR

# SKIP_KEYS="head.weight;head.bias;cls_token;pos_embed;patch_embed.proj.bias;patch_embed.proj.weight"

torchrun --standalone --nproc_per_node=$SLURM_GPUS_ON_NODE ft.py \
    --model vit_tiny  --warmup_epochs 50 --epochs 300 \
    --total_batch_size $TOTAL_BATCH_SIZE \
    --batch_size $BATCH_SIZE --lr 2e-3 --update_freq $UPDATE_FREQ --use_amp true \
    --data_path "/home/dawooda/code/procedural/data" \
    --data_set "CIFAR100" \
    --output_dir "/home/dawooda/code/procedural/Procedural/results_ft64t_512_$SLURM_JOB_ID" \
    --enable_wandb true \
    --project "cifar100" \
    --notes "kdyck_truncated" \
    --slurm_id $SLURM_JOB_ID \
    --initialize "/home/dawooda/code/procedural/Procedural/results_pr64t_freeze_27167395/pr_27167395_14999_fixed.pth" 
    # --skip_keys $SKIP_KEYS 


echo "DONE";
echo "Finished at $(date)";