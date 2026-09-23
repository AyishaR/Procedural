#!/bin/bash
#SBATCH --job-name rngprobe
#SBATCH --partition lmbdlc2_gpu-l40s
#SBATCH --nodes 1
#SBATCH --gres=gpu:2
#SBATCH --time 0:15:00
#SBATCH --mem 32G
#SBATCH --cpus-per-task 4
#SBATCH -o /home/schrodi/Procedural/logs/ft_%j_%x.out
#SBATCH -e /home/schrodi/Procedural/logs/ft_%j_%x.err
cd /home/schrodi/Procedural; source .venv/bin/activate
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:$((20000+RANDOM%20000)) --nproc_per_node=2 \
  plots/rank_rng_probe.py
echo "RNGPROBE_DONE"
