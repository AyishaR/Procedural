# sourced by every dynamics sbatch script (submit from the worktree root: SLURM_SUBMIT_DIR is used as ROOT)
umask 002
ROOT=${PROC_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
export DYN_ROOT=${DYN_ROOT:-/work/dlc2workfs3/schrodi-procedural/kempfe/dynamics}
export PROBE_ROOT=$DYN_ROOT                 # the probes package (features/lens/probes/subsets) writes under the dynamics root
export PROC_RESULTS=${PROC_RESULTS:-$ROOT/results}
VENV=${PROC_VENV:-/work/dlc2workfs3/schrodi-procedural/kempfe/.venv}
source "$VENV/bin/activate"
cd "$ROOT"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} task ${SLURM_ARRAY_TASK_ID:-none} root $ROOT dyn_root $DYN_ROOT venv $VENV commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
run_name() { python -c "from dynamics import registry as r; print(r.RUNS[int('$1')]['name'])"; }
