#!/bin/bash

cd "$(git rev-parse --show-toplevel)" || exit
NOW=$(date +"%Y%m%d_%H%M%S")

TRAIN_DATA_SET_A=("$HOME/Dropbox/develop/data/20220712/data_joints-2-3-4-5")
TRAIN_DATA_SET_B=("${TRAIN_DATA_SET_A[@]}" "$HOME/Dropbox/develop/data/20220719/data_joints-2-3-4-5_time-1-2")
TRAIN_DATA_SET_C=("${TRAIN_DATA_SET_B[@]}" "$HOME/Dropbox/develop/data/20220719/data_joints-2-3-4-5_time-2-4")
OUTPUT_BASE="$HOME/Dropbox/develop/data/20220808/track_periodic_traj_esn"

JOINT=(5)
# N_PREDICT=(1 3 5 7 10 15 20 30)
N_PREDICT=(10)
SCALER=("minmax")
DURATION=30

DATASET=("A" "B" "C")
TRAJECTORY=("sin" "step")
AMPLITUDE=(10 20 30 40)
PERIOD=(1 3 5 10)

LOG="${OUTPUT_BASE}/${NOW}.log"

mkdir -p "$OUTPUT_BASE"
:> "$LOG"

# TRAIN DATA SET A
if [[ " ${DATASET[*]} " =~ " A " ]]; then
  output="${OUTPUT_BASE}"/SET-A_joint-5_n-predict-10_scaler-minmax_sin_amp-10_period-1.csv
  cmd=(pdm run ./apps/track_periodic_traj_esn.py --train-data "${TRAIN_DATA_SET_A[@]}" --joint 5 --n-predict 10 --n-ctrl-period 1 --scaler minmax --output "${output}" --trajectory sin --time-duration "$DURATION" --amplitude 10 --period 1 --ctrl esn)
  echo "${cmd[@]}"
  "${cmd[@]}"
  for j in "${JOINT[@]}"; do
    for k in "${N_PREDICT[@]}"; do
      for s in "${SCALER[@]}"; do
        for t in "${TRAJECTORY[@]}"; do
          for a in "${AMPLITUDE[@]}"; do
            for p in "${PERIOD[@]}"; do
              echo "TRAIN SET: A, joint: $j, n_predict: $k, scaler: $s, traj: $t, A:$a, T:$p"
              output="${OUTPUT_BASE}"/SET-A_joint-"$j"_n-predict-"$k"_scaler-"$s"_"$t"_amp-"$a"_period-"$p".csv
              cmd=(pdm run ./apps/track_periodic_traj_esn.py --train-data "${TRAIN_DATA_SET_A[@]}" --joint "${j}" --n-predict "${k}" --n-ctrl-period 1 --scaler "${s}" --output "${output}" --trajectory "${t}" --time-duration "$DURATION" --amplitude "${a}" --period "${p}" --ctrl esn --model "model.joblib")
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
fi

# TRAIN DATA SET B
if [[ " ${DATASET[*]} " =~ " B " ]]; then
  output="${OUTPUT_BASE}"/SET-B_joint-5_n-predict-10_scaler-minmax_sin_amp-10_period-1.csv
  cmd=(pdm run ./apps/track_periodic_traj_esn.py --train-data "${TRAIN_DATA_SET_B[@]}" --joint 5 --n-predict 10 --n-ctrl-period 1 --scaler minmax --output "${output}" --trajectory sin --time-duration "$DURATION" --amplitude 10 --period 1 --ctrl esn)
  echo "${cmd[@]}"
  "${cmd[@]}"
  for j in "${JOINT[@]}"; do
    for k in "${N_PREDICT[@]}"; do
      for s in "${SCALER[@]}"; do
        for t in "${TRAJECTORY[@]}"; do
          for a in "${AMPLITUDE[@]}"; do
            for p in "${PERIOD[@]}"; do
              echo "TRAIN SET: B, joint: $j, n_predict: $k, scaler: $s, traj: $t, A:$a, T:$p"
              output="${OUTPUT_BASE}"/SET-B_joint-"$j"_n-predict-"$k"_scaler-"$s"_"$t"_amp-"$a"_period-"$p".csv
              cmd=(pdm run ./apps/track_periodic_traj_esn.py --train-data "${TRAIN_DATA_SET_B[@]}" --joint "${j}" --n-predict "${k}" --n-ctrl-period 1 --scaler "${s}" --output "${output}" --trajectory "${t}" --time-duration "$DURATION" --amplitude "${a}" --period "${p}" --ctrl esn --model "model.joblib")
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
fi

# TRAIN DATA SET C
if [[ " ${DATASET[*]} " =~ " C " ]]; then
  output="${OUTPUT_BASE}"/SET-C_joint-5_n-predict-10_scaler-minmax_sin_amp-10_period-1.csv
  cmd=(pdm run ./apps/track_periodic_traj_esn.py --train-data "${TRAIN_DATA_SET_C[@]}" --joint 5 --n-predict 10 --n-ctrl-period 1 --scaler minmax --output "${output}" --trajectory sin --time-duration "$DURATION" --amplitude 10 --period 1 --ctrl esn)
  "${cmd[@]}"
  for j in "${JOINT[@]}"; do
    for k in "${N_PREDICT[@]}"; do
      for s in "${SCALER[@]}"; do
        for t in "${TRAJECTORY[@]}"; do
          for a in "${AMPLITUDE[@]}"; do
            for p in "${PERIOD[@]}"; do
              echo "TRAIN SET: C, joint: $j, n_predict: $k, scaler: $s, traj: $t, A:$a, T:$p"
              output="${OUTPUT_BASE}"/SET-C_joint-"$j"_n-predict-"$k"_scaler-"$s"_"$t"_amp-"$a"_period-"$p".csv
              cmd=(pdm run ./apps/track_periodic_traj_esn.py --train-data "${TRAIN_DATA_SET_C[@]}" --joint "${j}" --n-predict "${k}" --n-ctrl-period 1 --scaler "${s}" --output "${output}" --trajectory "${t}" --time-duration "$DURATION" --amplitude "${a}" --period "${p}" --ctrl esn --model "model.joblib")
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
fi
