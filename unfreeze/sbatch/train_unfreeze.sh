#!/bin/bash
# One progressive-unfreezing arm on ViT-S / ImageNet-100 (docs/unfreeze_vit_handoff.md): the
# r_random recipe (recipe.sh) + --unfreeze_arm $ARM, seed $SEED. ~4 h on 2x H200, so one job;
# --requeue covers preemption and the in-job retry loop the transient /dev/shm DataLoader
# deaths (auto_resume continues from the last epoch checkpoint either way).
#   sbatch -J uf-u4f-lin0.2 --export=ALL,ARM=u4f-lin0.2,SEED=0 unfreeze/sbatch/train_unfreeze.sh
#SBATCH --job-name unfreeze
#SBATCH --partition alldlc2_gpu-h200
#SBATCH --nodes 1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --time 23:29:59
#SBATCH --requeue
#SBATCH -o /work/dlc2workfs3/schrodi-procedural/kempfe/unfreeze/logs/%x_%j.out
#SBATCH -e /work/dlc2workfs3/schrodi-procedural/kempfe/unfreeze/logs/%x_%j.err
#SBATCH --mail-type END,FAIL

SECONDS=0
[[ -z "$ARM" ]] && { echo "ARM (an unfreeze tag, e.g. u4f-lin0.2) is required"; exit 2; }
SEED=${SEED:-0}
source "${PROC_ROOT:-/work/dlc2workfs3/schrodi-procedural/kempfe/Procedural}/unfreeze/sbatch/recipe.sh"
cd "$ROOT" || exit 1
umask 002
echo "Started at $(date) on $(hostname): arm $ARM seed $SEED, JID $SLURM_JOB_ID, queue $SLURM_JOB_PARTITION, restart ${SLURM_RESTART_COUNT:-0}"
source /work/dlc2workfs3/schrodi-procedural/kempfe/.venv/bin/activate
echo "cwd $(pwd)  commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.used --format=csv

OUT="$OUT_ROOT/runs/$ARM/s$SEED"
mkdir -p "$OUT"
MAX_RETRIES=${MAX_RETRIES:-6}
attempt=0
while true; do
    attempt=$((attempt + 1))
    echo "=== attempt $attempt/$MAX_RETRIES at $(date) ==="
    PORT_OFFSET=$attempt run_main "$OUT" "$OUT_ROOT/runs/accuracy_${ARM}_s$SEED.json" "$ARM" \
        --enable_wandb ${ENABLE_WANDB:-true} --project "i100-unfreeze" --notes "unfreeze $ARM"
    TORCH_EXIT=$?
    echo "Torchrun exited with code $TORCH_EXIT"
    [[ $TORCH_EXIT -eq 0 ]] && break
    [[ $attempt -ge $MAX_RETRIES ]] && { echo "giving up after $attempt attempts"; break; }
    echo "retrying in 30s; auto_resume continues from the last epoch checkpoint"
    sleep 30
done
echo "Finished at $(date), runtime ${SECONDS}s"
exit $TORCH_EXIT
