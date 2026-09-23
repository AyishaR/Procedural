#!/bin/bash
#SBATCH --job-name structinit
#SBATCH --partition lmbdlc2_gpu-l40s
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --time 0:50:00
#SBATCH -o /home/schrodi/Procedural/logs/ft_%j_%x.out
#SBATCH -e /home/schrodi/Procedural/logs/ft_%j_%x.err
cd /home/schrodi/Procedural
source .venv/bin/activate
python plots/analyse_init_structure.py \
    --model "vit_base" --data_set "IMNET" --data_path "/data/datasets/ILSVRC2012" \
    --initialize "results/pr_vitb_n/pr_6066174_final.pth" \
    --procedural_data "kdyck" --procedural_order "standard" --skip_norm true \
    --out plots/cache/init_structure.json
echo "STRUCT_DONE"
