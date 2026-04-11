#!/usr/bin/env bash
# pixi の activation で読み込み: ローカルビルド LibRaw を参照（システム libraw に依存しない）
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LIBRAW_LOCAL_PREFIX="${_ROOT}/third_party/libraw-install"
export PKG_CONFIG_PATH="${LIBRAW_LOCAL_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
# 実行時にローカル libraw / libomp を解決（rpath でも足りない環境向け）
if [[ "$(uname -s)" == "Darwin" ]]; then
  export DYLD_FALLBACK_LIBRARY_PATH="${LIBRAW_LOCAL_PREFIX}/lib:${CONDA_PREFIX:+$CONDA_PREFIX/lib}:${DYLD_FALLBACK_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${LIBRAW_LOCAL_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
