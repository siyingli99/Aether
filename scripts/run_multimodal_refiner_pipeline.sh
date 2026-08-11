#!/usr/bin/env bash
# Unified multimodal LiDAR refiner pipeline (RGB = frozen Difix pretrained).
#
# All code and data paths are inside the AETHER repo. Run from AETHER root:
#
#   conda activate aether
#   cd /path/to/Aether_os
#   bash scripts/run_multimodal_refiner_pipeline.sh [step]
#
# step: all | 1 | 2 | 3 | 4 | 5 | smoke

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AETHER_ROOT="${AETHER_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export AETHER_ROOT
cd "$AETHER_ROOT"

CORPUS="${CORPUS_ROOT:-$AETHER_ROOT/outputs/multimodal_refine}"
STEP="${1:-all}"

banner() {
  echo ""
  echo "================================================================"
  echo "  $1"
  echo "================================================================"
  echo ""
}

run_step1() {
  banner "STEP 1/5 — GT LiDAR 语料（PandaSet 103 + nuScenes 场景，无需 3DGS checkpoint）"
  echo "预计耗时：数小时（取决于 GPU / stride）。可先 smoke test："
  echo "  python aether/refine/gen_lidar_gt_corpus.py --scene_limit 2 --stride 4"
  echo ""
  cd "$AETHER_ROOT"
  python aether/refine/gen_lidar_gt_corpus.py \
    --datasets pandaset,nuscenes \
    --stride 2 \
    --severities 0.8,1.0,1.2 \
    --out "$CORPUS"
  echo "[STEP 1 DONE] 摘要: $CORPUS/gt_corpus_summary.json"
}

run_step2() {
  banner "STEP 2/5 — 真实 3DGS 渲染 LiDAR 对（可选，需 joint checkpoint）"
  echo "场景列表见 configs/refine/scene_registry.yaml"
  echo "若 joint 不存在可跳过本步，仅用 GT 语料训练。"
  echo ""
  cd "$AETHER_ROOT"
  python aether/refine/gen_multimodal_pairs.py \
    --registry configs/refine/scene_registry.yaml \
    --out "$CORPUS" \
    --stride 2
  echo "[STEP 2 DONE] 输出: $CORPUS/render_corpus/"
}

run_step3() {
  banner "STEP 3/5 — 合并 train/test manifest"
  cd "$AETHER_ROOT"
  python aether/refine/merge_pair_manifests.py \
    --corpus_root "$CORPUS" \
    --sources gt_corpus,render_corpus \
    --val_ratio 0.1 \
    --out "$CORPUS/global_pairs.json"
  echo "[STEP 3 DONE] $CORPUS/global_pairs.json"
}

run_step4() {
  banner "STEP 4/5 — 训练 CrossModal LiDAR Refiner（RGB 不训练，bundle 仅记录 nvidia/difix_ref）"
  echo "预计耗时：数小时（80 epochs，~数万对）。"
  echo ""
  cd "$AETHER_ROOT"
  python aether/refine/train_unified_lidar.py \
    --pairs "$CORPUS/global_pairs.json" \
    --out "$CORPUS/unified" \
    --epochs 80 \
    --bs 8
  echo "[STEP 4 DONE] bundle: $CORPUS/unified/unified_multimodal.bundle.pth"
}

run_step5() {
  banner "STEP 5/5 — 验证 bundle 加载"
  cd "$AETHER_ROOT"
  python - <<'PY'
import torch
import os
from aether.refine.unified_multimodal_enhancer import load_unified_bundle

path = os.path.join(
    os.environ["AETHER_ROOT"],
    "outputs/multimodal_refine/unified/unified_multimodal.bundle.pth",
)
enc = load_unified_bundle(path, device="cuda" if torch.cuda.is_available() else "cpu")
print(f"RGB model (pretrained, not in bundle): {enc.rgb_model_id}")
x = torch.randn(1, 3, 64, 1800)
xr = torch.randn(1, 3, 64, 1800)
with torch.no_grad():
    y = enc.lidar(x.cuda() if torch.cuda.is_available() else x,
                  xr.cuda() if torch.cuda.is_available() else xr)
print(f"LiDAR forward OK, output shape={tuple(y.shape)}")
PY
  echo "[STEP 5 DONE] 统一多模态增强网络已就绪。"
  echo "  LiDAR: $CORPUS/unified/unified_multimodal.bundle.pth"
  echo "  RGB:   HuggingFace nvidia/difix_ref（推理时由 aether/refine 桥接加载）"
}

case "$STEP" in
  1) run_step1 ;;
  2) run_step2 ;;
  3) run_step3 ;;
  4) run_step4 ;;
  5) run_step5 ;;
  all)
    run_step1
    run_step2 || echo "[WARN] Step 2 skipped or failed — continuing with GT corpus only"
    run_step3
    run_step4
    run_step5
    ;;
  smoke)
    banner "SMOKE TEST — 2 场景 GT 语料 + 合并 + 2 epoch 训练"
    cd "$AETHER_ROOT"
    python aether/refine/gen_lidar_gt_corpus.py --scene_limit 2 --stride 8 --severities 1.0 --out "$CORPUS"
    python aether/refine/merge_pair_manifests.py --corpus_root "$CORPUS" --sources gt_corpus --out "$CORPUS/global_pairs.json"
    python aether/refine/train_unified_lidar.py --pairs "$CORPUS/global_pairs.json" --out "$CORPUS/unified_smoke" --epochs 2 --bs 2
    echo "[SMOKE DONE]"
    ;;
  *)
    echo "Unknown step: $STEP  (use all|1|2|3|4|5|smoke)"
    exit 1
    ;;
esac
