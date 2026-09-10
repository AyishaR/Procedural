# sourced by every probes sbatch script (submit from the repo root: SLURM_SUBMIT_DIR is used as ROOT)
umask 002
ROOT=${PROC_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
export PROBE_ROOT=${PROBE_ROOT:-/work/dlc2workfs3/schrodi-procedural/kempfe/probes}
export PROC_RESULTS=${PROC_RESULTS:-$ROOT/results}
VENV=${PROC_VENV:-}
if [[ -z "$VENV" ]]; then
  for c in "$ROOT/../.venv" /work/dlc2workfs3/schrodi-procedural/kempfe/.venv; do
    [[ -x "$c/bin/python" ]] && VENV=$c && break
  done
fi
source "$VENV/bin/activate"
cd "$ROOT"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} task ${SLURM_ARRAY_TASK_ID:-none} root $ROOT venv $VENV commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
