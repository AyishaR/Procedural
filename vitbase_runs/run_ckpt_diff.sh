#!/bin/bash
#SBATCH --job-name ckptdiff
#SBATCH --partition testdlc2_gpu-l40s
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --time 1:00:00
#SBATCH --mem 80G
#SBATCH --cpus-per-task 8
#SBATCH -o /home/schrodi/Procedural/logs/ft_%j_%x.out
#SBATCH -e /home/schrodi/Procedural/logs/ft_%j_%x.err

ROOT='/home/schrodi/Procedural'
cd $ROOT
source .venv/bin/activate
echo "Started at $(date) on $(hostname)"
nvidia-smi -L || echo "WARNING: no GPU"

python plots/analyse_ckpt_differences.py \
    --model "vit_base" \
    --data_set "IMNET" \
    --data_path "/data/datasets/ILSVRC2012" \
    --initialize "results/pr_vitb_n/pr_6066174_final.pth" \
    --procedural_data "kdyck" --procedural_order "standard" \
    --skip_norm true \
    --input_size 224 \
    --n_images 128 \
    --out plots/cache/ckpt_diff.json

echo "CKPTDIFF_DONE at $(date)"
