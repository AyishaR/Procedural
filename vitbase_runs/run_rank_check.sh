#!/bin/bash
#SBATCH --job-name rankchk
#SBATCH --partition lmbdlc2_gpu-l40s
#SBATCH --nodes 1
#SBATCH --gres=gpu:2
#SBATCH --time 1:00:00
#SBATCH --mem 120G
#SBATCH --cpus-per-task 16
#SBATCH -o /home/schrodi/Procedural/logs/ft_%j_%x.out
#SBATCH -e /home/schrodi/Procedural/logs/ft_%j_%x.err
cd /home/schrodi/Procedural; source .venv/bin/activate
COMMON="--model vit_base --warmup_epochs 50 --epochs 300 --total_batch_size 4096 \
 --batch_size 128 --lr 2e-3 --update_freq 16 --use_amp true \
 --data_path /data/datasets/ILSVRC2012 --data_set IMNET \
 --initialize results/pr_vitb_n/pr_6066174_final.pth \
 --output_dir plots/cache/rank_tmp --enable_wandb false \
 --procedural_data kdyck --procedural_order standard --skip_norm true --num_workers 8 --seed 0"
SH=""; for b in 0 1 2 3 4 5 6 7 8; do
  SH="${SH}${b}[norm1.weight,norm1.bias,attn.qk.weight,attn.v.weight,attn.qkv.bias,attn.proj.weight,attn.proj.bias,norm2.weight,norm2.bias,mlp.fc1.weight,mlp.fc1.bias,mlp.fc2.weight,mlp.fc2.bias];"
done; SH="${SH%;}"
echo "########## ftb4e3 (weight_shuffle -> torch.randperm)"
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:$((20000+RANDOM%20000)) --nproc_per_node=2 \
  plots/check_rank_divergence.py $COMMON --random_blocks "9,10,11" --weight_shuffle "$SH"
echo "########## ftbqm1dv (quantile match -> torch.argsort)"
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:$((20000+RANDOM%20000)) --nproc_per_node=2 \
  plots/check_rank_divergence.py $COMMON --init_method "quantile_match_target_blocks" \
  --quantile_1d_mode "shuffle" --quantile_qkv_mode "qk_v" --init_method_scaled_blocks "0,1,2,3,4,5,6,7,8"
echo "RANKCHK_DONE"
