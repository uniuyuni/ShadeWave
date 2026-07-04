"""
Viewer のソート/フィルタ設定と、item dict リストへの適用ロジック。
Kivy 非依存（headless テスト可能）。item dict は widgets/viewer.py の
ViewerWidget._new_image_item が生成する形式を前提とする。
"""
import os

from utils import rating_utils

SORT_KEYS = ("filename", "date", "rating", "edited")
FILTER_EDITED_VALUES = ("all", "edited", "unedited")
FILTER_TYPE_VALUES = ("all", "raw", "rgb")

DEFAULT_SETTINGS = {
    "sort_key": "filename",
    "sort_descending": False,
    "filter_rating_min": 0,
    "filter_edited": "all",
    "filter_type": "all",
    "filter_text": "",
}

# EXIF 撮影日時の優先順（viewer のホバーヒントと同じ）。exiftool の
# "YYYY:MM:DD HH:MM:SS" 形式は文字列比較で時系列順になる。
_DATE_EXIF_KEYS = ("CreateDate", "DateCreated", "FileModifyDate", "ModifyDate")


def is_default_settings(settings) -> bool:
    return all(settings.get(k, v) == v for k, v in DEFAULT_SETTINGS.items())


def _norm_path_key(p: str) -> str:
    try:
        return os.path.normcase(os.path.abspath(p))
    except OSError:
        return os.path.normcase(p or "")


def _date_key(item) -> str:
    exif = item.get("exif_data")
    if isinstance(exif, dict):
        for key in _DATE_EXIF_KEYS:
            value = exif.get(key)
            if value not in (None, ""):
                return str(value)
    return ""


def passes_filters(item, settings) -> bool:
    file_path = item.get("file_path") or ""

    text = (settings.get("filter_text") or "").strip().lower()
    if text and text not in os.path.basename(file_path).lower():
        return False

    ftype = settings.get("filter_type", "all")
    if ftype == "raw" and not rating_utils.is_raw_path(file_path):
        return False
    if ftype == "rgb" and rating_utils.is_raw_path(file_path):
        return False

    edited = settings.get("filter_edited", "all")
    if edited == "edited" and not item.get("pmck_exists"):
        return False
    if edited == "unedited" and item.get("pmck_exists"):
        return False

    rating_min = int(settings.get("filter_rating_min", 0) or 0)
    if rating_min > 0:
        # メタデータ未取得（exif_data=None）の item はまだ星が確定していないので隠さない。
        if item.get("exif_data") is not None and int(item.get("rating", 0) or 0) < rating_min:
            return False

    return True


def _sort_key_func(sort_key):
    def path_key(item):
        return _norm_path_key(item.get("file_path") or "")

    if sort_key == "date":
        return lambda item: (_date_key(item), path_key(item))
    if sort_key == "rating":
        return lambda item: (int(item.get("rating", 0) or 0), path_key(item))
    if sort_key == "edited":
        return lambda item: (bool(item.get("pmck_exists")), path_key(item))
    return path_key


def sort_items(items, settings):
    return sorted(
        items,
        key=_sort_key_func(settings.get("sort_key", "filename")),
        reverse=bool(settings.get("sort_descending", False)),
    )


def build_view(items, settings):
    """フィルタ通過 item をソートした新しいリストを返す（item dict は共有）。"""
    return sort_items([it for it in items if passes_filters(it, settings)], settings)
