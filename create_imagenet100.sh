#!/bin/bash
#SBATCH --job-name ft
#SBATCH --partition lmbhiwidlc_gpu-rtx2080
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --time 09:59:59
#SBATCH -o /home/dawooda/code/procedural/Procedural/logs/misc_%j_%x.out
#SBATCH -e /home/dawooda/code/procedural/Procedural/logs/misc_%j_%x.err # STDERR
#SBATCH --mail-type END,FAIL 

# Download the class list
wget https://raw.githubusercontent.com/HobbitLong/CMC/master/imagenet100.txt

# Create subset directories
mkdir -p imagenet100/train imagenet100/val
while read wnid; do
    ln -s /data/datasets/ILSVRC2012/train/$wnid imagenet100/train/$wnid
    ln -s /data/datasets/ILSVRC2012/val/$wnid imagenet100/val/$wnid
done < imagenet100.txt