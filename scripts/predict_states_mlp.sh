#!/usr/bin/bash

cd "$(git rev-parse --show-toplevel)" || exit
# TODAY=$(date +"%Y%m%d")
TODAY="20220628"
NOW=$(date +"%Y%m%d_%H%M%S")
TRAIN_DATA="$HOME/Dropbox/develop/data/$TODAY/sin-T-1-10/train"
TEST_DATA="$HOME/Dropbox/develop/data/$TODAY/sin-T-5/test"
OUTPUT_BASE="$HOME/Dropbox/develop/data/$TODAY/predict_states_mlp"
JOINT=(5)
N_PREDICT=(1 3 5 10)
SCALER=("none" "minmax" "std")
LOG="${OUTPUT_BASE}/${NOW}.log"

mkdir -p "$OUTPUT_BASE"
:> "$LOG"
for j in "${JOINT[@]}"; do
  for n in "${N_PREDICT[@]}"; do
    for s in "${SCALER[@]}"; do
      echo "joint: $j, n_predict: $n, scaler: $s"
      cmd=(pdm run ./apps/predict_states_mlp.py --train-data "${TRAIN_DATA}" --test-data "${TEST_DATA}" --joint "${j}" --n-predict "${n}" --scaler "${s}" -d "${OUTPUT_BASE}"/joint-"$j"_n-predict-"$n"_scaler-"$s" -e png -s -x)
      echo "${cmd[@]}" >>"$LOG"
      "${cmd[@]}" >>"$LOG" 2>&1
    done
  done
done
