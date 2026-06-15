#!/usr/bin/env bash
# PyInstaller で Shade Wave.app をビルドする（現在の python / pixi 環境を使用）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec python3 scripts/build_macos_app_pyinstaller.py "$@"
