#!/bin/bash

for lr in $(seq 0.5 0.01 0.99); do
  pdm run ./apps/track_recorded_traj_reBASICS.py --record-data ~/Dropbox/develop/data/20220720/track_recorded_traj_mlp_20220720_102614.csv --train-data ~/Dropbox/develop/data/20220720/track_recorded_traj_mlp_20220720_102614.csv --test-data ~/Dropbox/develop/data/20220720/track_recorded_traj_mlp_20220720_102614.csv --config ./apps/config.toml --output ~/Dropbox/develop/data/20220808/track_recorded_traj_reBASICS_"$(date +'+%Y%m%d_%H%M%S')".csv --joint 2 3 4 5 --time-duration 30 --warmup 0.25 --n-train-loop 10 --n-neurons 100 --n-modules 800 --density 0.05 --input-scale 1.0 --rho 1.2 --leaking-rate "$lr" --noise-level 0.001 -s -x
done
