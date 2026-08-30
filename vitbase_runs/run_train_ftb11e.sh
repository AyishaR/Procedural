#!/bin/bash
#SBATCH --job-name ftb11e
#SBATCH --partition lmbdlc2_gpu-h200
#SBATCH --nodes 1
#SBATCH --gres=gpu:4
#SBATCH --time 23:29:59
#SBATCH -o /home/schrodi/Procedural/logs/ft_%j_%x.out
#SBATCH -e /home/schrodi/Procedural/logs/ft_%j_%x.err # STDERR
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user schrodi@cs.uni-freiburg.de 

SECONDS=0

ROOT='/home/schrodi/Procedural'

cd $ROOT
echo "Started at $(date)";

echo "Running job $SLURM_JOB_NAME using $SLURM_GPUS_ON_NODE gpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";
echo $([[ -z "$SLURM_ID" ]]);
echo $([[ "$SLURM_ID" -eq "" ]]);
if [[ -z "$SLURM_ID" ]] | [[ "$SLURM_ID" -eq "" ]]; then
    SLURM_ID=$SLURM_JOB_ID
fi
if [[ -z "$SEED" ]] | [[ "$SEED" -eq "" ]]; then
    SEED=0
fi
echo "Running with ID $SLURM_ID";

export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate

echo "Current working directory: $(pwd)";

nvidia-smi
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

TOTAL_BATCH_SIZE=4096
BATCH_SIZE=128
UPDATE_FREQ=$((($TOTAL_BATCH_SIZE / $SLURM_GPUS_ON_NODE) / $BATCH_SIZE))

# for i in 0 1 2; do
torchrun --standalone --nproc_per_node=$SLURM_GPUS_ON_NODE main.py \
    --model vit_base  --warmup_epochs 50 --epochs 300 \
    --total_batch_size $TOTAL_BATCH_SIZE \
    --batch_size $BATCH_SIZE --lr 2e-3 --update_freq $UPDATE_FREQ --use_amp true \
    --data_path "/data/datasets/ILSVRC2012" \
    --data_set "IMNET" \
    --initialize "results/pr_vitb/pr_6066174_final.pth" \
    --output_dir "results/imnet_base/results_IMNET_BASE_$SLURM_ID/s$SEED" \
    --enable_wandb true \
    --project "vit base kdyck" \
    --wandb_entity_name "procedural_pretraining" \
    --notes "" \
    --accuracy_json "results/imnet_base/accuracy_IMNET_BASE_$SLURM_ID.json" \
    --procedural_data "kdyck" \
    --procedural_order "standard" \
    --pr_notes "" \
    --skip_norm true \
    --random_blocks "0" \
    --init_method "downscale_pr_match_delta_norms" \
    --init_method_scaled_blocks "1,2,3,4,5,6,7,8,9,10,11" \
    --stage_wise_metrics true \
    --detailed_metrics true \
    --slurm_id $SLURM_ID \
    --seed $SEED
    # --skip_keys $SKIP_KEYS 

#     sleep 10
# done
TORCH_EXIT=$?
echo "Torchrun exited with code $TORCH_EXIT"

duration=$SECONDS
if (( duration < 300 )); then  # 5 min = 300s
    echo "Runtime ${duration}s too short. Stop chain."
    exit 2
else
    exit $TORCH_EXIT
fi