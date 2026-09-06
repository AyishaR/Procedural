#!/bin/bash
#SBATCH --job-name imnetcopy
#SBATCH --partition lmbdlc2_gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time 24:00:00
#SBATCH -o /home/schrodi/Procedural/logs/imnetcopy_%j.out
# One-off, resumable copy of ILSVRC2012 (train 1.28M JPEGs ~153 GB + val 50k ~6.4 GB) from the HDD
# NFS server (dlc-hdd1, the source of the 2026-09-05/06 epoch stalls) to the SSD workspace
# /work/dlcsmall2/schrodi-imagenet (ws_allocate -F dlcsmall imagenet; extend with ws_extend).
# Per-class rsync so a killed job resumes where it stopped; P parallel streams because the source
# is latency-bound (small files). Re-submit until the final counts match.
SRC=/data/datasets/ILSVRC2012; DST=/work/dlcsmall2/schrodi-imagenet; P=${P:-16}
mkdir -p $DST/train $DST/val; echo "Started $(date) on $(hostname), P=$P"
ls $SRC/val | xargs -P $P -I{} sh -c 'rsync -a --no-inc-recursive '"$SRC"'/val/{}/ '"$DST"'/val/{}/'; echo "val done $(date): $(find $DST/val -type f | wc -l) files (expect 50000)"
ls $SRC/train > $DST/classes.txt; echo "$(wc -l < $DST/classes.txt) classes"
cat $DST/classes.txt | xargs -P $P -I{} sh -c 'rsync -a --no-inc-recursive '"$SRC"'/train/{}/ '"$DST"'/train/{}/ && echo "$(date +%H:%M:%S) {} ok"'
echo "train: $(ls $DST/train | wc -l) classes, $(find $DST/train -type f | wc -l) files (expect 1000 / 1281167), $(du -sh $DST/train | cut -f1)"; echo "Finished $(date)"
