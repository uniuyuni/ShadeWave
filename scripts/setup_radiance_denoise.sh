#!/usr/bin/env bash
# radiance_denoise を platypus に取り込むセットアップ。
#   1) external/radiance_denoise/ が無ければ GitHub から clone
#   2) ネイティブ拡張 (_native*.so) を cmake/ninja でビルドし、Python パッケージ内に配置
#
# pixi 環境内のツールチェーン (cmake / ninja / pybind11 / cxx-compiler / llvm-openmp)
# を使うため、`pixi run setup-denoise` 経由での実行を想定している。
# platypus 側は effects.py の `from radiance_denoise.radiance_denoise.native import ...`
# という入れ子パスで参照する（リポジトリ直下の namespace ディレクトリ経由）ため、
# 別途 editable install は必須ではないが、pixi install で行われる。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="${ROOT_DIR}/external/radiance_denoise"
REPO_URL="https://github.com/uniuyuni/radiance_denoise.git"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  echo "radiance_denoise が見つからないので clone します: ${REPO_URL}"
  git clone "${REPO_URL}" "${REPO_DIR}"
else
  echo "radiance_denoise は既に存在します: ${REPO_DIR}"
fi

# pixi 環境のツール／cmake config を優先的に使う
PIXI_ENV="${ROOT_DIR}/.pixi/envs/default"
if [[ -d "${PIXI_ENV}" ]]; then
  export PATH="${PIXI_ENV}/bin:${PATH}"
  export CMAKE_PREFIX_PATH="${PIXI_ENV}:${CMAKE_PREFIX_PATH:-}"
else
  echo "警告: ${PIXI_ENV} がありません。先に 'pixi install' を実行してください。" >&2
fi

BUILD_DIR="${REPO_DIR}/native/build"
echo "ネイティブ拡張をビルドします (CPU/OpenMP + Metal)..."
cmake -G Ninja -S "${REPO_DIR}/native" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}"

# 生成物（_native 拡張）の確認
if compgen -G "${REPO_DIR}/radiance_denoise/_native"*.so >/dev/null; then
  echo "OK: $(ls "${REPO_DIR}/radiance_denoise/_native"*.so)"
else
  echo "エラー: _native 拡張が生成されませんでした。" >&2
  echo "       pixi 環境に pybind11 / cxx-compiler があるか確認してください。" >&2
  exit 1
fi

echo "完了。'pixi run python -c \"from radiance_denoise.radiance_denoise.native import metal_available; print(metal_available())\"' で確認できます。"
