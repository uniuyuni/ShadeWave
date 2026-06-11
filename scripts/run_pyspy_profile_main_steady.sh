#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${ROOT_DIR}/.profiles"
mkdir -p "${OUT_DIR}"

WARMUP_SEC="${PYSPY_WARMUP_SEC:-10}"
DURATION_SEC="${PYSPY_DURATION_SEC:-45}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_FILE="${OUT_DIR}/main-steady-w${WARMUP_SEC}-d${DURATION_SEC}-${STAMP}.svg"

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

echo "Starting target process: ${MAIN_PY}"
"${PYTHON_BIN}" -X frozen_modules=off "${MAIN_PY}" &
TARGET_PID=$!

cleanup() {
  if kill -0 "${TARGET_PID}" >/dev/null 2>&1; then
    kill "${TARGET_PID}" >/dev/null 2>&1 || true
    wait "${TARGET_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "Warm-up: ${WARMUP_SEC}s"
sleep "${WARMUP_SEC}"

if ! kill -0 "${TARGET_PID}" >/dev/null 2>&1; then
  echo "Target process exited before profiling started."
  exit 1
fi

echo "Profiling steady-state for ${DURATION_SEC}s -> ${OUT_FILE}"
if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "py-spy on macOS requires sudo; enter your password if prompted."
  sudo "${PYSPY_BIN}" record --pid "${TARGET_PID}" --subprocesses --duration "${DURATION_SEC}" --output "${OUT_FILE}"
else
  "${PYSPY_BIN}" record --pid "${TARGET_PID}" --subprocesses --duration "${DURATION_SEC}" --output "${OUT_FILE}"
fi

echo "py-spy steady profile written to: ${OUT_FILE}"
