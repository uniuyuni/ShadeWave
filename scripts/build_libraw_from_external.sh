#!/usr/bin/env bash
# external/LibRaw-master を autotools でビルドし third_party/libraw-install に入れる（pixi 内のツールのみ使用）
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT_DIR}/libraw_enhanced/external/LibRaw-master"
PREFIX="${ROOT_DIR}/third_party/libraw-install"
PIXI_ENV="${ROOT_DIR}/.pixi/envs/default"

if [[ ! -f "${SRC}/configure.ac" ]]; then
  echo "エラー: ${SRC} が見つかりません。" >&2
  exit 1
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

echo "configure --prefix=${PREFIX}"
# LibRaw の configure は --disable-static を認識しない場合があるため付けない
if ! ./configure --prefix="${PREFIX}" --disable-examples --enable-openmp; then
  echo "openmp 付きで失敗したため openmp なしで再試行します..."
  ./configure --prefix="${PREFIX}" --disable-examples --disable-openmp
fi

CORES="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
make -j"${CORES}"
make install

echo
echo "完了: ${PREFIX}"
echo "次: pixi run install-libraw-enhanced"
