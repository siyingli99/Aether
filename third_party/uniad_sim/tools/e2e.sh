#!/usr/bin/env bash
# -------------------------------------------------------------------------
# e2e.sh for launching UniAD_SIM in closed-loop mode
# Usage: e2e.sh <GPU_ID> <OUTPUT_DIR>
# -------------------------------------------------------------------------

# Resolve everything from this vendored checkout. Each path can be overridden
# without editing this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIAD_PATH="${UNIAD_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
AETHER_ROOT="${AETHER_ROOT:-$(cd "${UNIAD_PATH}/../.." && pwd)}"
cd "${UNIAD_PATH}" || { echo "Cannot cd to ${UNIAD_PATH}"; exit 1; }

# 4. 设置 PYTHONPATH (彻底抛弃父进程的路径)
export PYTHONPATH="${UNIAD_PATH}:${UNIAD_PATH}/tools/closeloop"

CONFIG_PATH="${UNIAD_CONFIG:-${UNIAD_PATH}/projects/configs/stage2_e2e/base_e2e.py}"
CHECKPOINT_PATH="${UNIAD_CHECKPOINT:-${AETHER_ROOT}/checkpoints/uniad_base_e2e.pth}"
UNIAD_PYTHON="${UNIAD_PYTHON:-${AETHER_ROOT}/.conda/uniad/bin/python}"

if [ ! -x "${UNIAD_PYTHON}" ]; then
  echo "UniAD Python not found: ${UNIAD_PYTHON}" >&2
  echo "Create .conda/uniad as shown in README.md, or set UNIAD_PYTHON." >&2
  exit 1
fi
if [ ! -f "${CHECKPOINT_PATH}" ]; then
  echo "UniAD checkpoint not found: ${CHECKPOINT_PATH}" >&2
  echo "Set UNIAD_CHECKPOINT after downloading the weight." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="$1" \
  "${UNIAD_PYTHON}" tools/closeloop/e2e.py \
    "${CONFIG_PATH}" "${CHECKPOINT_PATH}" \
    --output "$2"
EXIT_CODE=$?

cd - > /dev/null
exit $EXIT_CODE