#!/bin/bash

cd "$(git rev-parse --show-toplevel)" || exit
NOW=$(date +"%Y%m%d_%H%M%S")

TRAIN_DATA_SET_A=("$HOME/Dropbox/develop/data/20220712/data_joints-2-3-4-5")
TRAIN_DATA_SET_B=("${TRAIN_DATA_SET_A[@]}" "$HOME/Dropbox/develop/data/20220719/data_joints-2-3-4-5_time-1-2")
TRAIN_DATA_SET_C=("${TRAIN_DATA_SET_B[@]}" "$HOME/Dropbox/develop/data/20220719/data_joints-2-3-4-5_time-2-4")
TEST_DATA_BASE="$HOME/Dropbox/develop/data/20220719/test_data_joints-5"
# TEST_DATA_SET=("${TEST_DATA_BASE}/sin_joint-5_A-40.0_T-5.0_b-50.0_00.csv" "${TEST_DATA_BASE}/step_joint-5_A-40.0_T-5.0_b-50.0_00.csv")
TEST_DATA_SET=("${TEST_DATA_BASE}/sin_joint-5_A-40.0_T-5.0_b-50.0_00.csv")
TEST_TRAJ_LABELS=("sin" "step")
OUTPUT_BASE="$HOME/Dropbox/develop/data/20220719/predict_ctrl_input_mlp"


JOINT=(5)
# N_PREDICT=(1 3 5 7 10 15 20 30)
N_PREDICT=(10)
SCALER=("minmax")
LOG="${OUTPUT_BASE}/${NOW}.log"

mkdir -p "$OUTPUT_BASE"
:> "$LOG"

# TRAIN DATA SET A
ii=0
for i in "${TEST_DATA_SET[@]}"; do
  for j in "${JOINT[@]}"; do
    for k in "${N_PREDICT[@]}"; do
      for s in "${SCALER[@]}"; do
        echo "TRAIN SET: A, test: ${TEST_TRAJ_LABELS[$ii]} joint: $j, n_predict: $k, scaler: $s"
        cmd=(pdm run ./apps/predict_ctrl_input_mlp.py --train-data "${TRAIN_DATA_SET_A[@]}" --test-data "${i}" --joint "${j}" --n-predict "${k}" --scaler "${s}" -d "${OUTPUT_BASE}"/SET-A_joint-"$j"_n-predict-"$k"_scaler-"$s"-"${TEST_TRAJ_LABELS[$ii]}" -e png -s -x)
        echo "${cmd[@]}" >>"$LOG"
        "${cmd[@]}" >>"$LOG" 2>&1
      done
    done
  done
  ii=$((ii + 1))
done

# TRAIN DATA SET B
ii=0
for i in "${TEST_DATA_SET[@]}"; do
  for j in "${JOINT[@]}"; do
    for k in "${N_PREDICT[@]}"; do
      for s in "${SCALER[@]}"; do
        echo "TRAIN SET: B, test: ${TEST_TRAJ_LABELS[$ii]} joint: $j, n_predict: $k, scaler: $s"
        cmd=(pdm run ./apps/predict_ctrl_input_mlp.py --train-data "${TRAIN_DATA_SET_B[@]}" --test-data "${i}" --joint "${j}" --n-predict "${k}" --scaler "${s}" -d "${OUTPUT_BASE}"/SET-B_joint-"$j"_n-predict-"$k"_scaler-"$s"-"${TEST_TRAJ_LABELS[$ii]}" -e png -s -x)
        echo "${cmd[@]}" >>"$LOG"
        "${cmd[@]}" >>"$LOG" 2>&1
      done
    done
  done
  ii=$((ii + 1))
done

# TRAIN DATA SET C
ii=0
for i in "${TEST_DATA_SET[@]}"; do
  for j in "${JOINT[@]}"; do
    for k in "${N_PREDICT[@]}"; do
      for s in "${SCALER[@]}"; do
        echo "TRAIN SET: C, test: ${TEST_TRAJ_LABELS[$ii]} joint: $j, n_predict: $k, scaler: $s"
        cmd=(pdm run ./apps/predict_ctrl_input_mlp.py --train-data "${TRAIN_DATA_SET_C[@]}" --test-data "${i}" --joint "${j}" --n-predict "${k}" --scaler "${s}" -d "${OUTPUT_BASE}"/SET-C_joint-"$j"_n-predict-"$k"_scaler-"$s"-"${TEST_TRAJ_LABELS[$ii]}" -e png -s -x)
        echo "${cmd[@]}" >>"$LOG"
        "${cmd[@]}" >>"$LOG" 2>&1
      done
    done
  done
  ii=$((ii + 1))
done
