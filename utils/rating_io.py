"""
レーティングのディスク永続化（RGB: ExifTool、RAW: .pmck マージ）
"""
from __future__ import annotations

import logging
import os
import subprocess
import shutil
from typing import Any, Optional

import json
from cores import pmck_store
from utils import rating_utils
from utils.exiftool_safe import safe_run_exiftool

_EXIF = shutil.which("exiftool") or "exiftool"


def merge_xmp_star_tags_into_exif(file_path: str, exif: Any) -> None:
    """
    PyExifTool 取得の exif 辞書に、exiftool -j 追読で得た XMP/Composite 由来の星を上書きマージ（インプレース）。
    左ペイン・キャッシュ経路（viewer の二段目 exiftool と同趣旨）。
    """
    # exif は空 dict でも可（PyExifTool が何も出さない新規ファイル）。{} は falsy なので not exif だと常に不達になり星が取れない
    if not file_path or exif is None or not os.path.isfile(file_path):
        return
    if not isinstance(exif, dict):
        return
    if not rating_utils.is_rgb_path(file_path):
        return
    try:
        p = subprocess.run(
            [
                _EXIF,
                "-j",
                "-n",
                "-m",
                "-XMP:Rating",
                "-XMP-xmp:Rating",
                "-Rating",
                "-Composite:Rating",
                file_path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired) as e:
        logging.debug("merge_xmp_star_tags_into_exif: %s", e)
        return
    if not (p.stdout or "").strip():
        return
    try:
        arr = json.loads(p.stdout)
    except json.JSONDecodeError:
        return
    if not arr or not isinstance(arr[0], dict):
        return
    row = arr[0]
    rating_utils.merge_exiftool_j_row_into_exif(exif, row)


# msgpack ルートの RAW 専用（primary_param には入れない）
PMCK_RAW_RATING_KEY = "platypus_raw_rating"


def read_pmck_dict(pmck_path: str) -> Optional[dict[str, Any]]:
    return pmck_store.read_path(pmck_path)


def _clamp_r(v) -> int:
    try:
        return max(0, min(5, int(v)))
    except (TypeError, ValueError):
        return 0


def raw_rating_from_pmck_dict(d: Optional[dict]) -> int:
    """.pmck 辞書から RAW 用レーティング。ルート key 優先、なければ legacy primary_param.rating。"""
    if not d:
        return 0
    v = d.get(PMCK_RAW_RATING_KEY)
    if v not in (None, ""):
        return _clamp_r(v)
    pp = d.get("primary_param") or {}
    v2 = pp.get("rating", None)
    if v2 in (None, ""):
        return 0
    return _clamp_r(v2)


def read_raw_pmck_rating_value(file_path: str) -> int:
    d = pmck_store.read_image(file_path)
    return raw_rating_from_pmck_dict(d)


def _pmck_has_substance(d: dict) -> bool:
    r = d.get(PMCK_RAW_RATING_KEY, 0)
    try:
        if int(r) > 0:
            return True
    except (TypeError, ValueError):
        pass
    pp = d.get("primary_param") or {}
    if len(pp) > 0:
        return True
    m2 = d.get("mask2")
    if m2 and len(m2) > 0:
        return True
    return False


def merge_raw_pmck_rating(file_path: str, rating: int) -> bool:
    """
    RAW: .pmck ルートに platypus_raw_rating（1～5）のみ。primary_param.rating は廃止のため除去。
    """
    def _merge(d):
        d = pmck_store.ensure_primary_param(d)
        pp = d["primary_param"]
        if isinstance(pp, dict):
            pp.pop("rating", None)
        d["primary_param"] = pp
        if int(rating) == 0:
            d.pop(PMCK_RAW_RATING_KEY, None)
        else:
            d[PMCK_RAW_RATING_KEY] = int(rating)
        if not _pmck_has_substance(d):
            return pmck_store.DELETE
        return d

    try:
        return pmck_store.update_image(file_path, _merge, default_empty=True)
    except OSError as e:
        raise RuntimeError(str(e)) from e


def _run_exiftool(argv: list[str]) -> None:
    r = safe_run_exiftool(argv, timeout=120, retries=2, backoff=0.3)
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip() or f"exiftool exit {r.returncode}"
        raise RuntimeError(err)


def write_exported_file_rating(file_path: str, new_rating: int) -> bool:
    """
    エクスポート直後の画像へ星を書き込み。メタデータ一括 ON/OFF とは独立して呼ぶ。
    VIPS 出力で XMP ブロックが無い場合、複合 -Rating= が有効（exiftool 挙動）。
    """
    if not file_path or not os.path.isfile(file_path):
        raise RuntimeError("invalid file")
    n = max(0, min(5, int(new_rating)))
    base = [
        _EXIF,
        "-P",
        "-overwrite_original",
        "-m",
        "-q",
    ]
    if n > 0:
        base.append(f"-Rating={n}")
        base.append(f"-XMP:Rating={n}")
        base.append(file_path)
    else:
        base.append("-XMP:Rating=")  # 空 = タグ除去（元ファイル編集用 write とは別）
        base.append(file_path)
    _run_exiftool(base)
    return True


def write_rgb_file_xmp_rating(
    file_path: str,
    new_rating: int,
    had_xmp_rating_before: bool,
) -> bool:
    """
    RGB: 元ファイルに XMP:Rating を書き込み。
    0 かつ had_xmp: XMP:Rating=0
    0 かつ not had: タグ削除（-XMP:Rating=）
    """
    if not file_path or not os.path.isfile(file_path):
        raise RuntimeError("invalid file")
    base = [
        _EXIF,
        "-P",
        "-overwrite_original",
        "-m",
        "-q",
    ]
    if new_rating > 0:
        base.append(f"-XMP:Rating={int(new_rating)}")
        base.append(file_path)
    elif new_rating == 0:
        if had_xmp_rating_before:
            base.append("-XMP:Rating=0")
        else:
            base.append("-XMP:Rating=")  # 空値 = 削除
        base.append(file_path)
    _run_exiftool(base)
    return True


def update_exif_dict_after_rgb_write(
    exif: dict, new_rating: int, had_before: bool
) -> None:
    """メモリ上 exif 辞書を書き込み結果に揃える。"""
    if new_rating > 0:
        exif["Rating"] = new_rating
        return
    if had_before:
        exif["Rating"] = 0
    else:
        exif.pop("Rating", None)


def notify_write_error(err: str, title: str = "Platypus") -> None:
    try:
        import macos

        macos.alert(str(err)[:2000], title=title, icon="stop")
    except Exception:
        logging.error("レーティング保存エラー: %s", err)
