#!/bin/bash
# Retrain LiDAR sidecars with FROZEN RGB geometry.
#
#   conda activate aether && cd /path/to/Aether_os
#   bash scripts/retrain_lidar_sidecar_frozen.sh 120          # pandaset 120
#   bash scripts/retrain_lidar_sidecar_frozen.sh 015
#   bash scripts/retrain_lidar_sidecar_frozen.sh nusc 88
#   bash scripts/retrain_lidar_sidecar_frozen.sh nusc 383
#
# Writes: outputs/train/<dataset>-<scene>/joint/lidar_sidecar_frozen.pt
# Does NOT overwrite the original RGB checkpoint_final.pth used for rendering.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AETHER="${AETHER_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "$AETHER"
export PYTHONPATH="${AETHER}${PYTHONPATH:+:${PYTHONPATH}}"
DS_ROOT="${AETHER}/third_party/drivestudio"
OUT_ROOT="outputs/train"
PANDA_DATA="data/pandaset/processed"
NUSC_DATA="data/nuscenes/processed_10Hz/trainval"
NUM_ITERS="${NUM_ITERS:-10000}"

arg1="${1:?scene id (e.g. 120) or 'nusc'}"
if [[ "$arg1" == "nusc" || "$arg1" == "nuscenes" ]]; then
  DATASET=nuscenes
  SCENE_RAW="${2:?nuscenes scene idx}"
else
  DATASET=pandaset
  SCENE_RAW="$arg1"
fi

SCENE_IDX=$((10#$SCENE_RAW))

if [[ "$DATASET" == "pandaset" ]]; then
  SID=$(printf "%03d" "$SCENE_IDX")
  PROJECT="pandaset-${SID}"
  RUN_NAME=joint
  RGB=$(ls -1 data/scenes/pandaset-${SID}/recon/checkpoint*.pth outputs/train/pandaset-${SID}/recon/checkpoint_final.pth 2>/dev/null | head -1 || true)
  EXTRA=(
    dataset=pandaset/6cams
    "data.data_root=${PANDA_DATA}"
    "data.scene_idx=${SCENE_IDX}"
  )
else
  RUN_NAME=joint
  SID=$(printf "%04d" "$SCENE_IDX")
  PROJECT="nuscenes-${SID}"
  RGB=$(ls -1 data/scenes/nuscenes-${SID}/recon/checkpoint*.pth 2>/dev/null | head -1 || true)
  # DriveStudio processed index of the scene (scene-0383 is index 300).
  PROC_IDX="${PROC_IDX:-$([[ "$SCENE_IDX" -eq 383 ]] && echo 300 || echo "$SCENE_IDX")}"
  EXTRA=(
    "dataset=nuscenes/6cams"
    "data.data_root=${NUSC_DATA}"
    "data.scene_idx=${PROC_IDX}"
  )
fi

[[ -n "${RGB}" && -f "${RGB}" ]] || { echo "ERROR: RGB ckpt not found"; exit 1; }

echo "[frozen-lidar] dataset=$DATASET run=$RUN_NAME"
echo "[frozen-lidar] RGB=$RGB"
echo "[frozen-lidar] out=${OUT_ROOT}/${PROJECT}/${RUN_NAME}"
echo "[frozen-lidar] iters=$NUM_ITERS"

python aether/train/joint_phase.py \
  --config_file "${DS_ROOT}/configs/omnire.yaml" \
  --resume_from "${RGB}" \
  --output_root "${OUT_ROOT}" \
  --project "${PROJECT}" \
  --run_name "${RUN_NAME}" \
  --num_iters "${NUM_ITERS}" \
  --save_freq "${NUM_ITERS}" \
  --print_freq 200 \
  --lidar_lr 5e-2 --lidar_every 1 \
  --w_depth 0.1 --w_intensity 0.05 --w_raydrop 0.05 --w_los 0.001 \
  --freeze_geometry \
  "${EXTRA[@]}"

echo "[frozen-lidar] DONE -> ${OUT_ROOT}/${PROJECT}/${RUN_NAME}/lidar_sidecar_frozen.pt"
