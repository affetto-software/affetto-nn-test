#!/bin/bash

TARGET_DIR="$HOME/Dropbox/develop/data/20220808/track_periodic_traj_esn"
# TARGET_DIR="$HOME/Dropbox/develop/data/20220808_2/track_periodic_traj_esn"
FILENAME_BASE="q"

mkdir -p "${TARGET_DIR}/${FILENAME_BASE}"
for dir in "$TARGET_DIR/"*; do
  if [[ -d "$dir" ]]; then
    dirname=$(basename -- "$dir")
    if [[ "$dirname" != "$FILENAME_BASE" ]]; then
      cp "${dir}/${FILENAME_BASE}"*.png "${TARGET_DIR}/${FILENAME_BASE}/${dirname}.png"
      echo "${TARGET_DIR}/${FILENAME_BASE}/${dirname}.png"
    fi
  fi
done
