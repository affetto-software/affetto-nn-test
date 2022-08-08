#!/bin/bash

cd "$(git rev-parse --show-toplevel)" || exit
NOW=$(date +"%Y%m%d_%H%M%S")

TRAIN_DATA_SET_A=("$HOME/Dropbox/develop/data/20220712/data_joints-2-3-4-5")
TRAIN_DATA_SET_B=("${TRAIN_DATA_SET_A[@]}" "$HOME/Dropbox/develop/data/20220719/data_joints-2-3-4-5_time-1-2")
TRAIN_DATA_SET_C=("${TRAIN_DATA_SET_B[@]}" "$HOME/Dropbox/develop/data/20220719/data_joints-2-3-4-5_time-2-4")
TRAIN_DATA_SET_N_1=("$HOME/Dropbox/develop/data/20220805/data_N_1/time_0.1-0.5" "$HOME/Dropbox/develop/data/20220805/data_N_1/time_1-2" "$HOME/Dropbox/develop/data/20220805/data_N_1/time_2-4")
TRAIN_DATA_SET_N_5=("$HOME/Dropbox/develop/data/20220805/data_N_5/time_0.1-0.5" "$HOME/Dropbox/develop/data/20220805/data_N_5/time_1-2" "$HOME/Dropbox/develop/data/20220805/data_N_5/time_2-4")
TRAIN_DATA_SET_N_10=("$HOME/Dropbox/develop/data/20220805/data_N_10/time_0.1-0.5" "$HOME/Dropbox/develop/data/20220805/data_N_10/time_1-2" "$HOME/Dropbox/develop/data/20220805/data_N_10/time_2-4")
TRAIN_DATA_SET_N_25=("$HOME/Dropbox/develop/data/20220805/data_N_25/time_0.1-0.5" "$HOME/Dropbox/develop/data/20220805/data_N_25/time_1-2" "$HOME/Dropbox/develop/data/20220805/data_N_25/time_2-4")
TRAIN_DATA_SET_N_50=("$HOME/Dropbox/develop/data/20220805/data_N_50/time_0.1-0.5" "$HOME/Dropbox/develop/data/20220805/data_N_50/time_1-2" "$HOME/Dropbox/develop/data/20220805/data_N_50/time_2-4")
OUTPUT_BASE="$HOME/Dropbox/develop/data/20220808_2/track_periodic_traj_esn"

JOINT=(5)
# N_PREDICT=(1 3 5 7 10 15 20 30)
N_PREDICT=(10)
SCALER=("minmax")
DURATION=30

TRAJECTORY=("sin" "step")
AMPLITUDE=(10 20 30 40)
PERIOD=(1 3 5 10)

LOG="${OUTPUT_BASE}/${NOW}.log"

mkdir -p "$OUTPUT_BASE"
:> "$LOG"

function do_all(){
  dataset=$(echo "$1" | tr '[:lower:]' '[:upper:]'); shift 1
  cmd=(pdm run ./apps/track_periodic_traj_esn.py --train-data "$@" --joint 5 --output "data.csv" --time-duration 10 --n-predict 10 --n-ctrl-period 1 --ctrl esn)
  echo "${cmd[@]}"
  "${cmd[@]}"
  for j in "${JOINT[@]}"; do
    for k in "${N_PREDICT[@]}"; do
      for s in "${SCALER[@]}"; do
        for t in "${TRAJECTORY[@]}"; do
          for a in "${AMPLITUDE[@]}"; do
            for p in "${PERIOD[@]}"; do
              echo "SET: ${dataset}, joint: $j, n_predict: $k, scaler: $s, traj: $t, A:$a, T:$p"
              output="${OUTPUT_BASE}"/SET-"${dataset}"_joint-"$j"_n-predict-"$k"_scaler-"$s"_"$t"_amp-"$a"_period-"$p".csv
              cmd=(pdm run ./apps/track_periodic_traj_esn.py --train-data "$@" --joint "${j}" --n-predict "${k}" --n-ctrl-period 1 --scaler "${s}" --output "${output}" --trajectory "${t}" --time-duration "$DURATION" --amplitude "${a}" --period "${p}" --ctrl esn --model "model.joblib")
              echo "${cmd[@]}" >>"$LOG"
              "${cmd[@]}"
              cmd=(python ./apps/scripts/plot_track_traj.py "$output" --joint 5 -d "${output%.*}" -s -x)
              echo "${cmd[@]}" >>"$LOG"
              "${cmd[@]}"
            done
          done
        done
      done
    done
  done
}

do_all A "${TRAIN_DATA_SET_A[@]}"
do_all B "${TRAIN_DATA_SET_B[@]}"
do_all C "${TRAIN_DATA_SET_C[@]}"
do_all N_1 "${TRAIN_DATA_SET_N_1[@]}"
do_all N_5 "${TRAIN_DATA_SET_N_5[@]}"
do_all N_10 "${TRAIN_DATA_SET_N_10[@]}"
do_all N_25 "${TRAIN_DATA_SET_N_25[@]}"
do_all N_50 "${TRAIN_DATA_SET_N_50[@]}"
