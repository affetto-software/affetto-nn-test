#!/usr/bin/env bash

cd "$(git rev-parse --show-toplevel)" || exit
TODAY=$(date +"%Y%m%d")
NOW=$(date +"%Y%m%d_%H%M%S")
OUTPUT_BASE="$HOME/Dropbox/develop/data/$TODAY"

pdm run ./apps/collect_data_periodic_traj.py -c "./apps/config.toml" -o "${OUTPUT_BASE}/data_${NOW}" -t sin -j 5 -T 20 -a 40 -p 1 2 3 -b 50 -n 10
