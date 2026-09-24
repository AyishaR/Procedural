#!/bin/bash
# Shared by train_unfreeze.sh and smoke.sh: schrodi's i100_playground arm-r recipe
# (/home/schrodi/Procedural/i100_playground/run_train_i100.sh, the r_random base this experiment
# pairs against), with --unfreeze_arm added. Sourced, not run.
#
#   run_main <output_dir> <accuracy_json> <arm tag or ''> [extra main.py flags...]
#
# NUM_WORKERS is FIXED (not derived from the CPU allocation as in run_train_i100.sh): the
# DataLoader's per-worker augmentation RNG depends on the worker count, so a fixed count keeps
# every arm on the same data + augmentation stream (same seed -> paired arms).
ROOT=${PROC_ROOT:-/work/dlc2workfs3/schrodi-procedural/kempfe/Procedural}
OUT_ROOT=${OUT_ROOT:-/work/dlc2workfs3/schrodi-procedural/kempfe/unfreeze}
NUM_WORKERS=${NUM_WORKERS:-12}
TOTAL_BATCH_SIZE=512
BATCH_SIZE=128

run_main() {
    local out=$1 acc=$2 arm=$3; shift 3
    local update_freq=$(( (TOTAL_BATCH_SIZE / SLURM_GPUS_ON_NODE) / BATCH_SIZE ))
    # --standalone pins rendezvous to localhost:29400 and deadlocks against any other torchrun
    # on the node (docs 5.12); derive a unique port from the job id (+ PORT_OFFSET for
    # sequential runs inside one job)
    local port=$(( 20000 + (SLURM_JOB_ID % 20000) + ${PORT_OFFSET:-0} ))
    echo "run_main: out=$out arm='${arm}' update_freq=$update_freq num_workers=$NUM_WORKERS port=$port extra: $*"
    torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:$port \
        --nproc_per_node=$SLURM_GPUS_ON_NODE main.py \
        --model vit_small --warmup_epochs 50 --epochs 300 \
        --total_batch_size $TOTAL_BATCH_SIZE \
        --batch_size $BATCH_SIZE --lr 2e-3 --update_freq $update_freq --use_amp true \
        --data_path "/home/schrodi/Procedural/imagenet100" \
        --data_set "IMNET100" \
        --initialize "" \
        --output_dir "$out" \
        --accuracy_json "$acc" \
        --grad_norms_json "$out/grad_norms.json" \
        --wandb_entity_name "procedural_pretraining" \
        --procedural_data "kdyck" \
        --procedural_order "standard" \
        --pr_notes "" \
        --skip_norm true \
        --init_method default \
        --num_workers $NUM_WORKERS \
        --stage_wise_metrics true \
        --detailed_metrics true \
        --slurm_id ${SLURM_ID:-$SLURM_JOB_ID} \
        --seed ${SEED:-0} \
        ${arm:+--unfreeze_arm "$arm"} \
        "$@"
}
