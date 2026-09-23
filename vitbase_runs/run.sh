#!/usr/bin/env bash
# Shorthand launcher for the b_vitb_*.sh scripts.
#
#   ./run.sh 1          ->  nohup ./b_vitb_1.sh  >> logs/b_vitb_1.log  2>&1 &
#   ./run.sh 2s         ->  nohup ./b_vitb_2s.sh >> logs/b_vitb_2s.log 2>&1 &
#   ./run.sh 1 2s 4     ->  launches all three
#   ./run.sh all        ->  launches every b_vitb_*.sh

cd "$(dirname "$0")" || exit 1

if [[ $# -eq 0 ]]; then
    echo "usage: $0 <id> [id ...]   (e.g. $0 1 2s 4, or $0 all)" >&2
    exit 1
fi

mkdir -p logs

if [[ "$1" == "all" ]]; then
    set -- $(ls b_vitb_*.sh | sed -e 's/^b_vitb_//' -e 's/\.sh$//')
fi

for id in "$@"; do
    script="b_vitb_${id}.sh"
    log="logs/b_vitb_${id}.log"

    if [[ ! -f "$script" ]]; then
        echo "skip: $script not found" >&2
        continue
    fi

    chmod +x "$script"
    nohup "./$script" >> "$log" 2>&1 &
    echo "started $script (pid $!) -> $log"
done
