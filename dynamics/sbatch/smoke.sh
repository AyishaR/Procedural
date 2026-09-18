#!/bin/bash
#SBATCH --job-name dyn_smoke
#SBATCH --partition alldlc2_gpu-h200
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 16
#SBATCH --mem 120G
#SBATCH --time 03:00:00
#SBATCH -o /work/dlc2workfs3/schrodi-procedural/kempfe/dynamics/smoke/logs/%x_%j.out
# Every stage on a few checkpoints with --limit into the separate smoke root, plus the numerical floors and the
# historical-lens validation at full size.  Submit from the worktree root.
set -eo pipefail
export DYN_ROOT=/work/dlc2workfs3/schrodi-procedural/kempfe/dynamics/smoke
export PROBE_ALLOW_LIMITED=1
source "${SLURM_SUBMIT_DIR:-$(pwd)}/dynamics/sbatch/common.sh"
# STAGES selects stages, '+'-separated (Slurm's --export cannot carry commas): sbatch --export=ALL,STAGES=grads+lens_probe,FORCE=1 ...
STAGES=${STAGES:-registry+rawlens+floors+hist+bypass+funcchange+grads+lens_probe}
F=${FORCE:+--force}
want() { [[ "+$STAGES+" == *"+$1+"* ]]; }
want registry && python -m dynamics.registry
if want rawlens; then
echo "=== rawlens (limit 512, files 0/149/299 + init) ==="
python -m dynamics.rawlens --runs ftbqmlnvo_s0 --files 0,149,299 --init --limit 512 --resident 4 $F
fi
if want floors; then
echo "=== precision floors: full-model loss under fp32 / tf32 / bf16 on 2048 images (file 299, and ftblrm file 200 with k-bias 676, ftbrhopl 299) ==="
for prec in fp32 tf32 bf16; do
  python -m dynamics.rawlens --runs ftbqmlnvo_s0,ftblrm_s0,ftbrhopl_s0 --files 299 --limit 2048 --precision $prec --tag $prec --splits test --resident 3 $F
  python -m dynamics.rawlens --runs ftblrm_s0 --files 200 --limit 2048 --precision $prec --tag $prec --splits test $F
done
fi
if want hist; then
echo "=== hist validation: full-size test rawlens at files 99 and 299 vs the training-time lens ==="
python -m dynamics.rawlens --runs ftbqmlnvo_s0,ftblrm_s0 --files 99,299 --splits test --resident 4 $F
python -m dynamics.hist --runs ftbqmlnvo_s0,ftblrm_s0 --validate $F
fi
if want bypass; then
echo "=== bypass (limit 512, files 0/149/299) ==="
python -m dynamics.bypass --runs ftbqmlnvo_s0 --files 0,149,299 --limit 512 $F
fi
if want funcchange; then
echo "=== funcchange (pairs 0,149,297,298 + floor, 256 images per set) ==="
python -m dynamics.funcchange --runs ftbqmlnvo_s0 --pairs 0,149,297,298 --floor --limit 256 $F
python -m dynamics.funcchange --runs ftbrhopl_s0 --pairs 298 --floor --limit 256 $F
fi
if want grads; then
echo "=== grads (files 0/149/299, 512 images per set) ==="
python -m dynamics.grads --runs ftbqmlnvo_s0 --files 0,149,299 --limit 512 $F
fi
if want lens_probe; then
echo "=== lens + probes (files 0/299, limit 512) ==="
python -m dynamics.lens_probe --runs ftbqmlnvo_s0 --files 0,299 --limit 512 --lens_steps 300 --probe_steps 300 $F
fi
echo "=== smoke done ==="
