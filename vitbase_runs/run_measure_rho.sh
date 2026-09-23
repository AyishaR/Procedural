#!/bin/bash
#SBATCH --job-name rhoinit
#SBATCH --partition lmbdlc2_gpu-l40s
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --time 0:40:00
#SBATCH -o /home/schrodi/Procedural/logs/ft_%j_%x.out
#SBATCH -e /home/schrodi/Procedural/logs/ft_%j_%x.err

ROOT='/home/schrodi/Procedural'
cd $ROOT
source .venv/bin/activate
echo "Started at $(date) on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L || echo "WARNING: nvidia-smi found no GPU on this node"

python plots/measure_init_rho_arms.py \
    --model "vit_base" \
    --data_set "IMNET" \
    --data_path "/data/datasets/ILSVRC2012" \
    --initialize "results/pr_vitb_n/pr_6066174_final.pth" \
    --procedural_data "kdyck" --procedural_order "standard" \
    --skip_norm true \
    --input_size 224 \
    --n_images 256 \
    --out plots/cache/init_rho.json

echo "Done at $(date)"
