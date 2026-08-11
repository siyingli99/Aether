#!/usr/bin/env bash
# LiDAR refiner v2 pipeline — real 3DGS renders only (no synthetic GT degradation).
#
#   conda activate aether
#   cd /path/to/Aether_os
#   bash scripts/run_lidar_refiner_v2_pipeline.sh [all|1|2|3|4|smoke]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AETHER_ROOT="${AETHER_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "$AETHER_ROOT"

CORPUS="${CORPUS_ROOT:-$AETHER_ROOT/outputs/multimodal_refine_v2}"
STEP="${1:-all}"

banner() {
  echo ""
  echo "================================================================"
  echo "  $1"
  echo "================================================================"
  echo ""
}

run_step1() {
  banner "STEP 1/4 — 真实 3DGS LiDAR：标称位姿 (registry scenes)"
  python aether/refine/gen_lidar_pairs.py \
    --registry "${REGISTRY:-configs/refine/scene_registry.yaml}" \
    --out "$CORPUS/render_corpus" \
    --stride "${PAIR_STRIDE:-2}"
  echo "[STEP 1 DONE] $CORPUS/render_corpus/*_nominal/"
}

run_step2() {
  banner "STEP 2/4 — 真实 3DGS LiDAR：闭环位姿扰动 (registry scenes)"
  python aether/refine/gen_multimodal_pairs.py \
    --registry configs/refine/scene_registry.yaml \
    --out "$CORPUS" \
    --stride 2
  echo "[STEP 2 DONE] $CORPUS/render_corpus/"
}

run_step3() {
  banner "STEP 3/4 — 合并 train/test（仅 real_render，排除 synthetic）"
  python aether/refine/merge_pair_manifests.py \
    --corpus_root "$CORPUS" \
    --sources render_corpus \
    --real_only \
    --val_ratio 0.1 \
    --out "$CORPUS/global_pairs.json"
  echo "[STEP 3 DONE] $CORPUS/global_pairs.json"
}

run_step4() {
  banner "STEP 4/4 — 训练 CrossModalLidarRefinerV2"
  python aether/refine/train_lidar_refiner_v2.py \
    --pairs "$CORPUS/global_pairs.json" \
    --out "$CORPUS/unified" \
    --epochs 120 \
    --bs 8 \
    --lr 1.5e-4 \
    --inpaint_cap 0.28
  echo "[STEP 4 DONE] bundle: $CORPUS/unified/unified_multimodal.bundle.pth"
}

case "$STEP" in
  1) run_step1 ;;
  2) run_step2 ;;
  3) run_step3 ;;
  4) run_step4 ;;
  all)
    run_step1
    run_step2
    run_step3
    run_step4
    ;;
  smoke)
    banner "SMOKE — 1 scene, 8 frames, 2 epochs"
    python aether/refine/gen_lidar_pairs.py \
      --scene pandaset017 --out "$CORPUS/render_corpus" --stride 8 --limit 8
    python aether/refine/merge_pair_manifests.py \
      --corpus_root "$CORPUS" --sources render_corpus --real_only \
      --out "$CORPUS/global_pairs.json"
    python aether/refine/train_lidar_refiner_v2.py \
      --pairs "$CORPUS/global_pairs.json" --out "$CORPUS/unified_smoke" \
      --epochs 2 --bs 2
    echo "[SMOKE DONE]"
    ;;
  *)
    echo "Usage: bash scripts/run_lidar_refiner_v2_pipeline.sh [all|1|2|3|4|smoke]"
    exit 1
    ;;
esac
