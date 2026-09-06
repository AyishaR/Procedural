#!/bin/bash
# Submit 3 seeds per arm: seed 0 gets its own job id, seeds 1-2 share it as SLURM_ID (results dir), as for every other arm.
cd /home/schrodi/Procedural
for script in vitbase_runs/run_train_ftbqmlnvog.sh vitbase_runs/run_train_ftbrhos.sh; do
  j0=$(sbatch --parsable --export=SEED=0 $script)
  j1=$(sbatch --parsable --export=SLURM_ID=$j0,SEED=1 $script)
  j2=$(sbatch --parsable --export=SLURM_ID=$j0,SEED=2 $script)
  echo "$script -> SLURM_ID $j0 (jobs $j0 $j1 $j2)"
done
squeue -u schrodi -h -o "%.10i %.12j %.2t %.8M %.10l %R" | grep -E "ftbqmlnvog|ftbrhos"
