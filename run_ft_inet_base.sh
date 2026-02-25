#!/bin/bash
#SBATCH --job-name ft
#SBATCH --partition lmbhiwidlc_gpu-rtx2080
#SBATCH --nodes 1
#SBATCH --gres=gpu:2
#SBATCH --time 09:59:59
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
echo "Current working directory: $(pwd)";

nvidia-smi
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

TOTAL_BATCH_SIZE=4096
BATCH_SIZE=64
UPDATE_FREQ=$((($TOTAL_BATCH_SIZE / $SLURM_GPUS_ON_NODE) / $BATCH_SIZE))

for i in 0; do
    torchrun --standalone --nproc_per_node=$SLURM_GPUS_ON_NODE ft.py \
        --model vit_base  --warmup_epochs 50 --epochs 300 \
        --total_batch_size $TOTAL_BATCH_SIZE \
        --batch_size $BATCH_SIZE --lr 2e-3 --update_freq $UPDATE_FREQ --use_amp true \
        --data_path "/data/datasets/ILSVRC2012" \
        --data_set "IMNET" \
        --output_dir "/home/dawooda/code/procedural/Procedural/results_ft_inet_kt/s$i" \
        --enable_wandb true \
        --project "imagenet" \
        --notes "kdyck_truncated p0.6" \
        --slurm_id $SLURM_JOB_ID \
        --initialize "/home/dawooda/code/procedural/Procedural/results_pr_vitb/pr_27246796_final.pth" \
        --seed $i
        # check "initialize" path

    sleep 10
done

echo "DONE";
echo "Finished at $(date)";