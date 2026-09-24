#!/bin/bash
# Submit the four E = 0.2 unfreezing arms (docs/unfreeze_vit_handoff.md), seed $SEED (default 0):
#   bash unfreeze/sbatch/submit.sh                 # all four
#   SEED=1 bash unfreeze/sbatch/submit.sh u4f-one0.2
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
ARMS=${@:-u4f-lin0.2 u4b-lin0.2 u4f-one0.2 u4b-one0.2}
SEED=${SEED:-0}
for arm in $ARMS; do
    JOB=$(sbatch --parsable -J "uf-$arm-s$SEED" --export=ALL,ARM=$arm,SEED=$SEED "$(pwd)/unfreeze/sbatch/train_unfreeze.sh")   # absolute: swrap does not find relative script paths
    echo "submitted $arm seed $SEED -> job $JOB"
done
