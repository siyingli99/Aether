#!/bin/bash
# =============================================================================
# AETHER 双模态(RGB+LiDAR)一键训练 + 可视化脚本
#
# 输出: outputs/train/pandaset-<SCENE>/{recon,joint}; 之后用 python -m aether import 生成场景包
#   Phase A : DriveStudio 纯 RGB 几何重建 (tools/train.py, 30k 步)
#   Phase B : AETHER 联合 RGB+LiDAR 微调   (aether/train/joint_phase.py, 默认 10k 步)
#   Phase C : 可视化 (viz_joint 单帧四联图 + viz_joint_video 全程视频)
#
# 关键路径处理:
#   - 处理后的数据在仓库下: data/pandaset/processed
#   - 但 DriveStudio 的 tools/train.py 必须在 drivestudio 目录下运行(SMPL/sky 等用相对该
#     目录的路径), 所以所有阶段都 cd 到 drivestudio, 并用绝对路径覆盖 data.data_root,
#     指向 AETHER 的数据, 从而消除"相对路径找不到数据"的问题。
#
# 用法:
#   bash scripts/train_pandaset_dualmodal.sh [SCENE] [PHASE_A_CKPT]
#     SCENE         : 场景号, 默认 001 (注意前导零, 内部已做八进制保护)
#     PHASE_A_CKPT  : 可选. 若已有 Phase-A 的 checkpoint_final.pth, 传入则跳过 Phase A
#
# 例:
#   bash scripts/train_pandaset_dualmodal.sh 001
#   bash scripts/train_pandaset_dualmodal.sh 001 /path/to/checkpoint_final.pth
# =============================================================================
set -euo pipefail

# ----------------------------- 可调参数 -------------------------------------
SCENE="${1:-001}"
PHASE_A_CKPT="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AETHER_ROOT="${AETHER_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DS_ROOT="${DRIVESTUDIO_ROOT:-${AETHER_ROOT}/third_party/drivestudio}"
# 处理后的 pandaset 数据根 (绝对路径). 包含 <SCENE>/{ego_pose,intrinsics,extrinsics,lidar,instances}
DATA_ROOT="${DATA_ROOT:-${AETHER_ROOT}/data/pandaset/processed}"
CONDA_ENV="${CONDA_ENV:-aether}"                       # Phase B / 可视化使用的环境
CONDA_ENV_PHASE_A="${CONDA_ENV_PHASE_A:-$CONDA_ENV}"   # Phase A 若需要不同环境可单独设
JOINT_ITERS="${JOINT_ITERS:-10000}"
VIZ_FRAME="${VIZ_FRAME:-40}"                           # 单帧可视化使用的帧号
VIZ_CAM="${VIZ_CAM:-0}"                                # 0 = front_camera
DATASET_CFG="pandaset/6cams"                          # 最完整: 6 路相机 + load_lidar=True
RUN_NAME="scene_${SCENE}_6cams_with_humans"

# ----------------------------- 激活 conda -----------------------------------
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "==================================================================="
echo " AETHER dualmodal pipeline | scene=${SCENE} | joint_iters=${JOINT_ITERS}"
echo " DS_ROOT   = ${DS_ROOT}"
echo " AETHER    = ${AETHER_ROOT}"
echo " DATA_ROOT = ${DATA_ROOT}"
echo "==================================================================="

# ----------------------------- 数据存在性检查 -------------------------------
SCENE_DATA="${DATA_ROOT}/${SCENE}"
if [ ! -d "${SCENE_DATA}" ]; then
    echo "!!! 找不到处理后的场景数据: ${SCENE_DATA}"
    exit 1
fi
if [ ! -d "${SCENE_DATA}/lidar" ] || [ -z "$(ls -A "${SCENE_DATA}/lidar" 2>/dev/null)" ]; then
    echo "!!! ${SCENE_DATA}/lidar 不存在或为空, 联合训练阶段需要 LiDAR GT。"
    exit 1
fi
echo ">>> 数据检查通过: ${SCENE_DATA} (含 lidar)"

# ============================================================================
# Phase A : DriveStudio 纯 RGB 重建 (30k 步, 较耗时)
#   - 输出到 AETHER/outputs/train/pandaset-<SCENE>/recon
#     (与现有 pandaset 场景的 checkpoint 约定一致)
# ============================================================================
PHASE_A_OUT_ROOT="${AETHER_ROOT}/outputs/train"
PHASE_A_OUT="${PHASE_A_OUT_ROOT}/pandaset-${SCENE}/recon"

if [ -z "${PHASE_A_CKPT}" ]; then
    echo ">>> [Phase A] 未提供已有 checkpoint, 开始 DriveStudio 重建训练..."
    conda activate "${CONDA_ENV_PHASE_A}"
    cd "${DS_ROOT}"                       # 必须在 drivestudio 下(SMPL/sky 相对路径)
    export PYTHONPATH="${DS_ROOT}"
    # data.scene_idx 用 "'$SCENE'" 双重引号: 防止 001/032 被 YAML 当成八进制
    # data.data_root 用绝对路径: 数据实际在 AETHER 下
    python tools/train.py \
        --config_file configs/omnire.yaml \
        --output_root "${PHASE_A_OUT_ROOT}" \
        --project "pandaset-${SCENE}" \
        --run_name recon \
        dataset="${DATASET_CFG}" \
        data.data_root="${DATA_ROOT}" \
        data.scene_idx="'${SCENE}'" \
        data.start_timestep=0 \
        data.end_timestep=-1 \
        render.render_full=False

    # 自动定位 Phase-A 的 checkpoint_final.pth (兼容是否带时间戳子目录)
    PHASE_A_CKPT="$(find "${PHASE_A_OUT}" -name 'checkpoint_final.pth' 2>/dev/null | sort | tail -n 1)"
fi

if [ -z "${PHASE_A_CKPT}" ] || [ ! -f "${PHASE_A_CKPT}" ]; then
    echo "!!! 未找到 Phase-A checkpoint_final.pth (查找路径: ${PHASE_A_OUT})"
    exit 1
fi
echo ">>> [Phase A] 完成, checkpoint = ${PHASE_A_CKPT}"

# ============================================================================
# Phase B : AETHER 联合 RGB+LiDAR 微调
# ============================================================================
echo ">>> [Phase B] 开始联合 RGB+LiDAR 微调 (${JOINT_ITERS} 步)..."
conda activate "${CONDA_ENV}"

JOINT_OUT_ROOT="${AETHER_ROOT}/outputs/train"
JOINT_RUN="joint"
JOINT_DIR="${JOINT_OUT_ROOT}/pandaset-${SCENE}/${JOINT_RUN}"

# 仍在 drivestudio 下运行(SMPL 等相对路径), data_root 用绝对路径覆盖。
cd "${DS_ROOT}"
python "${AETHER_ROOT}/aether/train/joint_phase.py" \
    --config_file "${DS_ROOT}/configs/omnire.yaml" \
    --resume_from "${PHASE_A_CKPT}" \
    --output_root "${JOINT_OUT_ROOT}" \
    --project "pandaset-${SCENE}" \
    --run_name "${JOINT_RUN}" \
    --num_iters "${JOINT_ITERS}" \
    dataset="${DATASET_CFG}" \
    data.data_root="${DATA_ROOT}" \
    data.scene_idx="'${SCENE}'"

JOINT_CKPT="${JOINT_DIR}/checkpoint_final.pth"
SIDECAR="$(find "${JOINT_DIR}" -name 'lidar_sidecar_step*.pt' 2>/dev/null | sort | tail -n 1)"

if [ ! -f "${JOINT_CKPT}" ] || [ -z "${SIDECAR}" ]; then
    echo "!!! 联合训练产物缺失: ckpt=${JOINT_CKPT} sidecar=${SIDECAR}"
    exit 1
fi
echo ">>> [Phase B] 完成"
echo "    joint ckpt    = ${JOINT_CKPT}"
echo "    lidar sidecar = ${SIDECAR}"

# ============================================================================
# Phase C : 可视化 (仍在 drivestudio 下运行以满足 SMPL 等相对路径)
# ============================================================================
echo ">>> [Phase C] 生成可视化..."
cd "${DS_ROOT}"
VIZ_DIR="${JOINT_DIR}/viz"
mkdir -p "${VIZ_DIR}"

# 单帧四联图: GT RGB | Pred RGB | GT LiDAR 叠加 | Pred LiDAR 叠加
python "${AETHER_ROOT}/aether/tools/viz_joint.py" \
    --resume_from "${JOINT_CKPT}" \
    --lidar_sidecar "${SIDECAR}" \
    --frame_idx "${VIZ_FRAME}" \
    --cam_idx "${VIZ_CAM}" \
    --out_png "${VIZ_DIR}/frame${VIZ_FRAME}_cam${VIZ_CAM}.png"

# 全程视频: 6 路 Pred RGB + LiDAR 叠加 + BEV
python "${AETHER_ROOT}/aether/tools/viz_joint_video.py" \
    --resume_from "${JOINT_CKPT}" \
    --lidar_sidecar "${SIDECAR}" \
    --out_dir "${VIZ_DIR}" \
    --out_name "dualmodal_scene${SCENE}.mp4" \
    --fps 10

echo "==================================================================="
echo " 全部完成!"
echo "   联合权重 : ${JOINT_CKPT}"
echo "   可视化   : ${VIZ_DIR}/"
echo "==================================================================="
