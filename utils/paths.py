from __future__ import annotations

import shutil
from pathlib import Path

APP_DIR_NAME = "Platypus"
CONFIG_FILE_NAME = "config.json"
EXPORT_PRESETS_FILE_NAME = "export_presets.json"
FILM_PRESETS_FILE_NAME = "film_presets.json"
PRESETS_DIR_NAME = "presets"

LEGACY_EXPORT_PRESETS_FILE_NAME = "export_preset.json"
LEGACY_PRESETS_DIR_NAME = "preset"


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


def user_data_dir() -> Path:
    """
    ユーザーが編集する Platypus 設定フォルダを返す。
    """
    return Path.home() / "Pictures" / APP_DIR_NAME


def internal_resource_path(*parts: str) -> Path:
    """
    アプリに同梱される初期設定・リソースへのパスを返す。
    """
    return repo_root().joinpath(*parts)


def _copy_file_if_missing(source: Path, destination: Path) -> None:
    if destination.exists() or not source.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_dir_if_missing(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    if source.is_dir():
        shutil.copytree(source, destination)
        return
    destination.mkdir(parents=True, exist_ok=True)


def _rename_if_needed(folder: Path, old_name: str, new_name: str) -> None:
    old_path = folder / old_name
    new_path = folder / new_name
    if new_path.exists() or not old_path.exists():
        return
    old_path.rename(new_path)


def ensure_user_data_dir() -> Path:
    """
    ~/Pictures/Platypus を作成し、初回起動に必要な設定ファイルを内蔵リソースからコピーする。
    旧名の preset / export_preset.json があれば、新名へ移行する。
    """
    folder = user_data_dir()
    folder.mkdir(parents=True, exist_ok=True)

    _rename_if_needed(folder, LEGACY_PRESETS_DIR_NAME, PRESETS_DIR_NAME)
    _rename_if_needed(folder, LEGACY_EXPORT_PRESETS_FILE_NAME, EXPORT_PRESETS_FILE_NAME)

    _copy_file_if_missing(internal_resource_path(CONFIG_FILE_NAME), folder / CONFIG_FILE_NAME)
    _copy_file_if_missing(internal_resource_path(FILM_PRESETS_FILE_NAME), folder / FILM_PRESETS_FILE_NAME)

    export_source = internal_resource_path(EXPORT_PRESETS_FILE_NAME)
    if not export_source.is_file():
        export_source = internal_resource_path(LEGACY_EXPORT_PRESETS_FILE_NAME)
    _copy_file_if_missing(export_source, folder / EXPORT_PRESETS_FILE_NAME)

    presets_source = internal_resource_path(PRESETS_DIR_NAME)
    if not presets_source.is_dir():
        presets_source = internal_resource_path(LEGACY_PRESETS_DIR_NAME)
    _copy_dir_if_missing(presets_source, folder / PRESETS_DIR_NAME)

    return folder


def user_data_path(*parts: str) -> Path:
    return ensure_user_data_dir().joinpath(*parts)


def config_path() -> Path:
    return user_data_path(CONFIG_FILE_NAME)


def export_presets_path() -> Path:
    return user_data_path(EXPORT_PRESETS_FILE_NAME)


def film_presets_path() -> Path:
    return user_data_path(FILM_PRESETS_FILE_NAME)


def presets_dir() -> Path:
    return user_data_path(PRESETS_DIR_NAME)
