#!/bin/bash

TARGET_DIR="$HOME/Dropbox/develop/data/20220719/predict_ctrl_input_mlp"
FILENAME_BASE="pressure_at_valve"

mkdir -p "${TARGET_DIR}/${FILENAME_BASE}"
for dir in "$TARGET_DIR/"*; do
  if [[ -d "$dir" ]]; then
    dirname=$(basename -- "$dir")
    cp "${dir}/${FILENAME_BASE}"*.png "${TARGET_DIR}/${FILENAME_BASE}/${dirname}.png"
  fi
done
