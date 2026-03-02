#!/bin/bash

# Download the class list
wget https://raw.githubusercontent.com/HobbitLong/CMC/master/imagenet100.txt

# Create subset directories
mkdir -p imagenet100/train imagenet100/val
while read wnid; do
    ln -s /data/datasets/ILSVRC2012//train/$wnid imagenet100/train/$wnid
    ln -s /data/datasets/ILSVRC2012//val/$wnid imagenet100/val/$wnid
done < imagenet100.txt