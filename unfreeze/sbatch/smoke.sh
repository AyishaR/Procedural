#!/bin/bash
# GPU smoke of the unfreezing mechanism on the REAL recipe and DDP configuration (2x H200,
# update_freq 2, fp16 GradScaler, T = 74,100), four runs in sequence, each stopped after
# STOP optimizer steps with snapshots (engine.save_smoke_snapshot) for unfreeze/check.py:
#   uf    u4b-lin0.008 -> u = 0 / 198 / 395 / 593: every chunk unfreezes inside the smoke
#   baseA / baseB      two identical vanilla runs (the GPU run-to-run noise floor)
#   ctrl  u4f-lin0     E = 0: must deviate from baseA no more than baseB does
# Output dirs are fresh per job ($OUT_ROOT/smoke/<jobid>/<name>) so auto_resume never picks up
# an older smoke.
#SBATCH --job-name uf-smoke
#SBATCH --partition alldlc2_gpu-h200
#SBATCH --nodes 1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --time 2:00:00
#SBATCH -o /work/dlc2workfs3/schrodi-procedural/kempfe/unfreeze/logs/%x_%j.out
#SBATCH -e /work/dlc2workfs3/schrodi-procedural/kempfe/unfreeze/logs/%x_%j.err

source "${PROC_ROOT:-/work/dlc2workfs3/schrodi-procedural/kempfe/Procedural}/unfreeze/sbatch/recipe.sh"
cd "$ROOT" || exit 1
umask 002
source /work/dlc2workfs3/schrodi-procedural/kempfe/.venv/bin/activate
echo "Started at $(date) on $(hostname), JID $SLURM_JOB_ID; cwd $(pwd) commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.used --format=csv

STOP=${STOP:-650}
SNAP="197,198,199,394,395,396,592,593,594"
SMOKE="$OUT_ROOT/smoke/$SLURM_JOB_ID"
mkdir -p "$SMOKE"
i=0
for spec in uf:u4b-lin0.008 baseA: baseB: ctrl:u4f-lin0; do
    name=${spec%%:*}; arm=${spec#*:}; i=$((i + 1))
    echo "=== $name (arm '${arm}') at $(date) ==="
    T0=$SECONDS
    PORT_OFFSET=$i run_main "$SMOKE/$name" "$SMOKE/accuracy_$name.json" "$arm" \
        --enable_wandb false --smoke_stop_step $STOP \
        --smoke_snapshot_every 100 --smoke_snapshot_steps "$SNAP" \
        > "$SMOKE.$name.log" 2>&1
    echo "$name exited $? after $((SECONDS - T0)) s (log $SMOKE.$name.log)"
done
echo "Finished at $(date)"
