#!/bin/bash
#SBATCH --partition alldlc2_gpu-h200
#SBATCH --requeue
#SBATCH --nodes 1
#SBATCH --gres=gpu:4
#SBATCH --time 23:29:59
#SBATCH -o /home/schrodi/Procedural/logs/ft_%j_%x.out
#SBATCH -e /home/schrodi/Procedural/logs/ft_%j_%x.err # STDERR
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user schrodi@cs.uni-freiburg.de

# Hand-over of a run that is training on L40S to this H200 job, whenever this job gets its slot (2026-09-21):
#   1. cancel every L40S job of the same arm (job name = arm; only partitions with "l40s" in the name, never anything else) and
#      wait until they have left the queue, so two jobs never write into one results directory;
#   2. make the directory safe to resume (plots/verify/verify_resume_checkpoint.py: a checkpoint cut off by the cancel is moved
#      aside, the per-epoch model file of the resume epoch is re-created if it was hit);
#   3. continue through the normal resume wrapper (k-bias zeroing + the arm's run script) on the unchanged schedule.
# 8 L40S ranks -> 4 H200 ranks: same global batch (update_freq follows the GPU count), different per-rank data split.
#   sbatch --job-name <arm> --export=SLURM_ID=<results id>,SEED=0 vitbase_runs/handover_to_h200.sh
ROOT='/home/schrodi/Procedural'
cd $ROOT
ARM=${ARM:-$SLURM_JOB_NAME}
SEED=${SEED:-0}
if [[ -z "$SLURM_ID" ]]; then echo "SLURM_ID (results id) must be exported"; exit 1; fi
d=results/imnet_base/results_IMNET_BASE_${SLURM_ID}/s${SEED}
if [[ ! -d results/imnet_base || ! -f vitbase_runs/run_train_${ARM}.sh ]]; then echo "results tree or run script not found (filesystem unmounted?), refusing to start"; exit 1; fi

# Routes have tiers: group H200 (3) > shared H200 (2) > group L40S (1) > shared L40S (0). This hand-over (tier MY) takes the run over
# from every RUNNING job of the run and from every PENDING job of the run on a tier <= MY (their continuations included), but it
# leaves PENDING jobs on HIGHER tiers alone, so a queued upgrade (e.g. the group-H200 hand-over) survives a start on a lower tier.
# If a job of the run is already RUNNING on a higher tier, this hand-over would be a downgrade: it exits without touching anything.
# Its own continuations (afterany on this job) are kept. Membership: the primary (job id = results id) or SLURM_ID=<results id>.
# (2026-09-22: earlier versions cancelled everything else, which wiped the queued group hand-overs and the L40S fallbacks.)
tier() { case "$1" in lmbdlc2_gpu-h200) echo 3;; alldlc2_gpu-h200) echo 2;; lmbdlc2_gpu-l40s) echo 1;; *) echo 0;; esac; }
MY=$(tier "$SLURM_JOB_PARTITION")
mine() { [[ "$1" == "$SLURM_ID" ]] || scontrol show job "$1" 2>/dev/null | grep -q "SLURM_ID=${SLURM_ID}\b"; }
higher_running=""
while read -r j p; do [[ "$j" == "$SLURM_JOB_ID" ]] && continue; mine "$j" && [[ $(tier "$p") -gt $MY ]] && higher_running="$higher_running $j"; done < <(squeue -u $USER -h -n "$ARM" -t R -o "%i %P")
if [[ -n "$higher_running" ]]; then echo "handover: $ARM already runs on a higher tier (job(s)$higher_running); this tier-$MY job exits untouched"; exit 0; fi
twins() { squeue -u $USER -h -n "$ARM" -o "%i|%P|%T|%E" | awk -F'|' -v me="$SLURM_JOB_ID" '$1 != me && $4 !~ ("afterany:" me "\\(|afterany:" me "$") {print $1"|"$2"|"$3}' | while IFS='|' read -r j p st; do mine "$j" || continue; if [[ "$st" == "PENDING" && $(tier "$p") -gt $MY ]]; then continue; fi; echo "$j"; done; }
for j in $(twins); do echo "handover: cancelling job $j of $ARM (results id $SLURM_ID)"; scancel $j; done
for i in $(seq 1 72); do left=$(twins); [[ -z "$left" ]] && break; sleep 5; done
if [[ -n "$left" ]]; then echo "handover: job(s) $left of $ARM still in the queue after 6 min, refusing to start"; exit 1; fi
sleep 30
export SLURM_ID SEED ARM
export STOP_AFTER_EPOCH=${STOP_AFTER_EPOCH:--1}
if ! ls "$d" 2>/dev/null | grep -qE '^checkpoint-[0-9]+\.pth(\.corrupt)?$'; then
    # the L40S twin never wrote a checkpoint (still pending, or stopped inside epoch 0): this job runs the arm from the start,
    # in the same results id
    echo "handover: no checkpoint in $d yet, starting $ARM from scratch under results id $SLURM_ID"
    exec bash vitbase_runs/run_train_${ARM}.sh
fi
.venv/bin/python plots/verify/verify_resume_checkpoint.py "$d" || { echo "handover: no resumable checkpoint"; exit 1; }

exec bash vitbase_runs/resume_zero_kbias.sh
