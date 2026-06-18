#!/usr/bin/env bash
# upstream LibRaw を取得して autotools でビルドし .pixi/libraw-install に入れる
# デフォルトで GitHub の最新リリースを取得する。オフライン時は
# external/libraw_enhanced/external/LibRaw-master を置けばそれを使う。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Source directory: prefer workspace copy for offline dev, otherwise fetch into .pixi
LOCAL_SRC="${ROOT_DIR}/external/libraw_enhanced/external/LibRaw-master"
FETCH_DIR="${ROOT_DIR}/.pixi/libraw-src"
PREFIX="${ROOT_DIR}/.pixi/libraw-install"
PIXI_ENV="${ROOT_DIR}/.pixi/envs/default"

if [[ -f "${LOCAL_SRC}/configure.ac" ]]; then
  SRC="${LOCAL_SRC}"
else
  # try to fetch latest release tag and clone it shallowly
  echo "ローカルソースが見つかりません。GitHub から最新リリースを取得します..."
  mkdir -p "${ROOT_DIR}/.pixi"

  # prefer system git binary that has https remote helper (git-remote-https)
  GIT_CMD=""
  for candidate in "/usr/bin/git" "$(command -v git 2>/dev/null || true)"; do
    if [[ -z "${candidate}" ]]; then
      continue
    fi
    if [[ -x "${candidate}" ]]; then
      exec_path="$(${candidate} --exec-path 2>/dev/null || true)"
      if [[ -n "${exec_path}" ]] && { [[ -x "${exec_path}/git-remote-https" ]] || [[ -x "${exec_path}/git-remote-http" ]]; }; then
        GIT_CMD="${candidate}"
        break
      fi
    fi
  done
  if [[ -z "${GIT_CMD}" ]]; then
    echo "エラー: git の https リモートヘルパーが利用できません。git をインストールまたは Xcode コマンドラインツールを有効化してください。" >&2
    exit 1
  fi
  echo "Using git: ${GIT_CMD}"

  # get latest release tag via GitHub API (best-effort)
  LATEST_TAG=$(curl -s https://api.github.com/repos/LibRaw/LibRaw/releases/latest | grep '"tag_name"' | sed -E 's/.*"([^\"]+)".*/\1/') || true

  # ensure clean fetch dir
  rm -rf "${FETCH_DIR}"

  if [[ -z "${LATEST_TAG}" ]]; then
    echo "警告: GitHub API で最新タグが取得できませんでした。master ブランチを使用します。"
    "${GIT_CMD}" clone --depth 1 https://github.com/LibRaw/LibRaw.git "${FETCH_DIR}"
  else
    echo "取得する LibRaw タグ: ${LATEST_TAG}"
    if ! "${GIT_CMD}" clone --depth 1 --branch "${LATEST_TAG}" https://github.com/LibRaw/LibRaw.git "${FETCH_DIR}"; then
      echo "タグ ${LATEST_TAG} で clone に失敗しました。master を試します。"
      rm -rf "${FETCH_DIR}"
      "${GIT_CMD}" clone --depth 1 https://github.com/LibRaw/LibRaw.git "${FETCH_DIR}"
    fi
  fi

  SRC="${FETCH_DIR}"
fi

if [[ ! -d "${PIXI_ENV}" ]]; then
  echo "エラー: ${PIXI_ENV} がありません。先に pixi install を実行してください。" >&2
  exit 1
fi

# pixi の bin を優先しつつ、Homebrew の autotools も使えるようにする（libtoolize 用）
export PATH="${PIXI_ENV}/bin:/opt/homebrew/bin:/usr/local/bin:${PATH}"

# configure だけあって compile / config.guess 等が無いと
# 「cannot find required auxiliary files: compile missing」になるため autoreconf が必要
needs_autoreconf=0
if [[ -f "${SRC}/configure" ]] && grep -q '^AC_PROG_LIBTOOL' "${SRC}/configure" 2>/dev/null; then
  echo "警告: ${SRC}/configure が壊れています（AC_PROG_LIBTOOL 等がシェルに展開されていません）。" >&2
  needs_autoreconf=1
fi
if [[ -f "${SRC}/configure" ]] && { [[ ! -f "${SRC}/compile" ]] || [[ ! -f "${SRC}/config.guess" ]] || [[ ! -f "${SRC}/install-sh" ]]; }; then
  echo "補助ファイル（compile 等）が不足しています。autoreconf を実行します。" >&2
  needs_autoreconf=1
fi

if [[ ! -x "${SRC}/configure" ]] || [[ "${needs_autoreconf}" -eq 1 ]]; then
  if ! command -v autoreconf >/dev/null 2>&1; then
    echo "エラー: autoreconf がありません。次のいずれかで入れてください:" >&2
    echo "  - pixi.toml の autoconf / automake / libtool を有効にして pixi install" >&2
    echo "  - macOS: brew install autoconf automake libtool" >&2
    exit 1
  fi
  if ! command -v libtoolize >/dev/null 2>&1 && ! command -v glibtoolize >/dev/null 2>&1; then
    echo "エラー: libtoolize/glibtoolize がありません。LibRaw の configure を生成するのに必須です。" >&2
    echo "  macOS: brew install libtool" >&2
    echo "  pixi: libtool を dependencies に追加して pixi install" >&2
    exit 1
  fi
fi

export CPPFLAGS="-I${PIXI_ENV}/include${CPPFLAGS:+ $CPPFLAGS}"
export LDFLAGS="-L${PIXI_ENV}/lib -Wl,-rpath,${PIXI_ENV}/lib${LDFLAGS:+ $LDFLAGS}"
export PKG_CONFIG_PATH="${PIXI_ENV}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

cd "${SRC}"

if [[ ! -x ./configure ]] || [[ "${needs_autoreconf}" -eq 1 ]]; then
  echo "configure を生成しています (autoreconf -fiv)..."
  autoreconf -fiv
fi

rm -rf "${PREFIX}"
mkdir -p "${PREFIX}"

echo "configure --prefix=${PREFIX} (source=${SRC})"
# LibRaw の configure は --disable-static を認識しない場合があるため付けない
if ! ./configure --prefix="${PREFIX}" --disable-examples --enable-openmp; then
  echo "openmp 付きで失敗したため openmp なしで再試行します..."
  ./configure --prefix="${PREFIX}" --disable-examples --disable-openmp
fi

CORES="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
make -j"${CORES}"
make install

if [[ "${SRC}" == "${FETCH_DIR}" ]]; then
  rm -rf "${FETCH_DIR}"
fi

echo
echo "LibRaw を ${PREFIX} にインストールしました。"
echo "注意: 従来の third_party/libraw-install は不要になります。"

echo
echo "完了: ${PREFIX}"
echo "次: pixi run install-libraw-enhanced"
