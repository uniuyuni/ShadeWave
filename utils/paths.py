from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """
    実行カレントディレクトリに依存せず、リポジトリ（platypus）ルートを返す。
    """
    # utils/paths.py -> utils -> repo root
    return Path(__file__).resolve().parents[1]


def rel(*parts: str) -> str:
    """
    リポジトリルート基準のパスを返す（Kivy等がstrを要求する箇所向け）。
    """
    return str(repo_root().joinpath(*parts))

