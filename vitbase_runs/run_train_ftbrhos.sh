#!/bin/bash
#SBATCH --job-name ftbrhos
#SBATCH --partition alldlc2_gpu-h200
#SBATCH --requeue
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

# DataLoader workers follow the CPU allocation rather than a constant.
CPUS_PER_RANK=$(( ${SLURM_CPUS_ON_NODE:-8} / $SLURM_GPUS_ON_NODE ))
if [[ $CPUS_PER_RANK -lt 1 ]]; then CPUS_PER_RANK=1; fi
NUM_WORKERS=${NUM_WORKERS:-$CPUS_PER_RANK}
echo "CPUs on node: ${SLURM_CPUS_ON_NODE:-?}, gpus: $SLURM_GPUS_ON_NODE -> num_workers=$NUM_WORKERS"

# for i in 0 1 2; do
# Transient /dev/shm exhaustion kills DataLoader workers; the chain reads the short runtime
# as "completed too quickly" and stops (docs 5.7). Retry in-job instead.
MAX_RETRIES=${MAX_RETRIES:-6}
attempt=0
while true; do
attempt=$((attempt+1))
echo "=== attempt $attempt/$MAX_RETRIES at $(date) ==="

# --standalone pins rendezvous to localhost:29400, which collides with any other
# torchrun on the same node (ours or another user's) and deadlocks at the first
# collective. Derive a unique port from the job id instead (docs 5.12).
MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 20000) ))
echo "rendezvous port: $MASTER_PORT"
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:$MASTER_PORT --nproc_per_node=$SLURM_GPUS_ON_NODE main.py \
    --model vit_base  --warmup_epochs 50 --epochs 300 \
    --total_batch_size $TOTAL_BATCH_SIZE \
    --batch_size $BATCH_SIZE --lr 2e-3 --update_freq $UPDATE_FREQ --use_amp true \
    --data_path "/work/dlcsmall2/schrodi-imagenet" \
    --data_set "IMNET" \
    --initialize "results/pr_vitb_n/pr_6066174_final.pth" \
    --output_dir "results/imnet_base/results_IMNET_BASE_$SLURM_ID/s$SEED" \
    --enable_wandb true \
    --project "vit base kdyck" \
    --wandb_entity_name "procedural_pretraining" \
    --notes "rank-sync-fix rho-match-random-to-shuffled-proc" \
    --accuracy_json "results/imnet_base/accuracy_IMNET_BASE_${SLURM_ID}_s${SEED}.json" \
    --grad_norms_json "results/imnet_base/grad_norms_IMNET_BASE_${SLURM_ID}_s${SEED}.json" \
    --procedural_data "kdyck" \
    --procedural_order "standard" \
    --pr_notes "" \
    --skip_norm true \
    --random_blocks "" \
    --init_method "upscale_random_match_delta_norms" \
    --init_method_scaled_blocks "0,1,2,3,4,5,6,7,8" \
    --target_model_weight_shuffle "0[norm1.weight,norm1.bias,attn.qk.weight,attn.v.weight,attn.qkv.bias,attn.proj.weight,attn.proj.bias,norm2.weight,norm2.bias,mlp.fc1.weight,mlp.fc1.bias,mlp.fc2.weight,mlp.fc2.bias];1[norm1.weight,norm1.bias,attn.qk.weight,attn.v.weight,attn.qkv.bias,attn.proj.weight,attn.proj.bias,norm2.weight,norm2.bias,mlp.fc1.weight,mlp.fc1.bias,mlp.fc2.weight,mlp.fc2.bias];2[norm1.weight,norm1.bias,attn.qk.weight,attn.v.weight,attn.qkv.bias,attn.proj.weight,attn.proj.bias,norm2.weight,norm2.bias,mlp.fc1.weight,mlp.fc1.bias,mlp.fc2.weight,mlp.fc2.bias];3[norm1.weight,norm1.bias,attn.qk.weight,attn.v.weight,attn.qkv.bias,attn.proj.weight,attn.proj.bias,norm2.weight,norm2.bias,mlp.fc1.weight,mlp.fc1.bias,mlp.fc2.weight,mlp.fc2.bias];4[norm1.weight,norm1.bias,attn.qk.weight,attn.v.weight,attn.qkv.bias,attn.proj.weight,attn.proj.bias,norm2.weight,norm2.bias,mlp.fc1.weight,mlp.fc1.bias,mlp.fc2.weight,mlp.fc2.bias];5[norm1.weight,norm1.bias,attn.qk.weight,attn.v.weight,attn.qkv.bias,attn.proj.weight,attn.proj.bias,norm2.weight,norm2.bias,mlp.fc1.weight,mlp.fc1.bias,mlp.fc2.weight,mlp.fc2.bias];6[norm1.weight,norm1.bias,attn.qk.weight,attn.v.weight,attn.qkv.bias,attn.proj.weight,attn.proj.bias,norm2.weight,norm2.bias,mlp.fc1.weight,mlp.fc1.bias,mlp.fc2.weight,mlp.fc2.bias];7[norm1.weight,norm1.bias,attn.qk.weight,attn.v.weight,attn.qkv.bias,attn.proj.weight,attn.proj.bias,norm2.weight,norm2.bias,mlp.fc1.weight,mlp.fc1.bias,mlp.fc2.weight,mlp.fc2.bias];8[norm1.weight,norm1.bias,attn.qk.weight,attn.v.weight,attn.qkv.bias,attn.proj.weight,attn.proj.bias,norm2.weight,norm2.bias,mlp.fc1.weight,mlp.fc1.bias,mlp.fc2.weight,mlp.fc2.bias]" \
    --num_workers $NUM_WORKERS \
    --stage_wise_metrics true \
    --detailed_metrics true \
    --slurm_id $SLURM_ID \
    --seed $SEED
    # --skip_keys $SKIP_KEYS 

#     sleep 10
# done
TORCH_EXIT=$?
echo "Torchrun exited with code $TORCH_EXIT"
if [ $TORCH_EXIT -eq 0 ]; then break; fi
if [ $attempt -ge $MAX_RETRIES ]; then echo "giving up after $attempt attempts"; break; fi
echo "retrying in 30s; auto_resume continues from the last epoch checkpoint"
sleep 30
done

duration=$SECONDS
if (( duration < 300 )); then  # 5 min = 300s
    echo "Runtime ${duration}s too short. Stop chain."
    exit 2
else
    exit $TORCH_EXIT
fi