#!/bin/bash
#SBATCH --job-name ftb3
#SBATCH --partition gpu_h100_short
#SBATCH --nodes 1
#SBATCH --gres=gpu:4
#SBATCH --time 00:29:59
#SBATCH -o /home/fr/fr_fr/fr_ad457/procedural/Procedural/logs/ft_%j_%x.out
#SBATCH -e /home/fr/fr_fr/fr_ad457/procedural/Procedural/logs/ft_%j_%x.err # STDERR
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user ayishardawood@gmail.com 

SECONDS=0

ROOT='/home/fr/fr_fr/fr_ad457'

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

module load devel/cuda/12.8
module load devel/miniforge/25.3.1-python-3.12
source ../../.venv/bin/activate

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
    --initialize "/work/dlclarge1/dawooda-pr_pretraining/results/pr_vitb/pr_27267764_final.pth" \
    --output_dir "/work/dlclarge1/dawooda-pr_pretraining/imnet_base/results_IMNET_BASE_$SLURM_ID/s$SEED" \
    --enable_wandb true \
    --project "vit base" \
    --wandb_entity_name "ayisharyhanadawood-universit-t-freiburg" \
    --notes "" \
    --accuracy_json "/work/dlclarge1/dawooda-pr_pretraining/imnet_base/s$SLURM_ID.json" \
    --procedural_data "kdyck_truncated" \
    --procedural_order "standard" \
    --pr_notes "" \
    --random_blocks "0,1,2,3,4,5,6,7,8,9,10" \
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