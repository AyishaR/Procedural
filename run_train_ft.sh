#!/bin/bash
#SBATCH --job-name ft
#SBATCH --partition lmbhiwidlc_gpu-rtx2080
#SBATCH --nodes 1
#SBATCH --gres=gpu:4
#SBATCH --time 15:59:59
#SBATCH -o /home/dawooda/code/procedural/Procedural/logs/ft_%j_%x.out
#SBATCH -e /home/dawooda/code/procedural/Procedural/logs/ft_%j_%x.err # STDERR
#SBATCH --mail-type END,FAIL 
#SBATCH --exclude=dlcgpu04,dlcgpu16

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

for i in 0 1 2; do
    torchrun --standalone --nproc_per_node=$SLURM_GPUS_ON_NODE main.py \
        --model vit_tiny  --warmup_epochs 50 --epochs 300 \
        --total_batch_size $TOTAL_BATCH_SIZE --image_flip True \
        --batch_size $BATCH_SIZE --lr 2e-3 --update_freq $UPDATE_FREQ --use_amp true \
        --data_path "/home/dawooda/code/procedural/data" \
        --data_set "CIFAR100" \
        --output_dir "/work/dlclarge1/dawooda-pr_pretraining/results/results_ft64_512_$SLURM_JOB_ID/s$i" \
        --enable_wandb true \
        --project "cifar100" \
        --wandb_entity_name "ayisharyhanadawood-universit-t-freiburg" \
        --notes "kdyck_truncated p0.6 vertical emr0.1 flipped" \
        --slurm_id $SLURM_JOB_ID \
        --initialize "/home/dawooda/code/procedural/Procedural/results_pr64_27409982/pr_27409982_final.pth" \
        --seed $i
        # --skip_keys $SKIP_KEYS 

    sleep 10
done

echo "DONE";
echo "Finished at $(date)";

