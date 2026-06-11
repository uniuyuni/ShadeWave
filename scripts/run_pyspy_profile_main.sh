#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${ROOT_DIR}/.profiles"
mkdir -p "${OUT_DIR}"
OUT_FILE="${OUT_DIR}/main-$(date +%Y%m%d-%H%M%S).svg"

PYSPY_BIN="${ROOT_DIR}/.pixi/envs/default/bin/py-spy"
PYTHON_BIN="${ROOT_DIR}/.pixi/envs/default/bin/python"
MAIN_PY="${ROOT_DIR}/main.py"

# Keep launch environment close to existing debug settings.
export PATH="${ROOT_DIR}/.pixi/envs/default/bin:${PATH}"
export CONDA_PREFIX="${ROOT_DIR}/.pixi/envs/default"
export CONDA_DEFAULT_ENV="platypus"
export PIXI_PROJECT_ROOT="${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export LIBRAW_LOCAL_PREFIX="${ROOT_DIR}/.pixi/libraw-install"
export PKG_CONFIG_PATH="${ROOT_DIR}/.pixi/libraw-install/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export PYDEVD_DISABLE_FILE_VALIDATION="1"
export OMP_DISPLAY_ENV="FALSE"
export KMP_WARNINGS="0"
export LIBOMP_VERBOSE="0"
export ENABLE_PJRT_COMPATIBILITY="1"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="0.0"
export PYTORCH_ENABLE_MPS_FALLBACK="1"
export PYTORCH_MPS_FAST_MATH="1"
export KMP_DUPLICATE_LIB_OK="FALSE"
export DYLD_FALLBACK_LIBRARY_PATH="${ROOT_DIR}/.pixi/libraw-install/lib:${ROOT_DIR}/.pixi/envs/default/lib:/usr/lib:/usr/local/lib:/opt/homebrew/lib"

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "py-spy on macOS requires sudo; enter your password if prompted."
  sudo "${PYSPY_BIN}" record --subprocesses --output "${OUT_FILE}" -- "${PYTHON_BIN}" -X frozen_modules=off "${MAIN_PY}"
else
  "${PYSPY_BIN}" record --subprocesses --output "${OUT_FILE}" -- "${PYTHON_BIN}" -X frozen_modules=off "${MAIN_PY}"
fi

echo "py-spy profile written to: ${OUT_FILE}"
