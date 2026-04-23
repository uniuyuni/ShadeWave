"""
星レーティング（0～5）の正規化とクリック挙動。
"""
import os
import define

_MAX_R = 5

# get_metadata 等のフラット辞書で、星が付く主なタグ名（XMP を先に。Composite の Rating より優先が安全）
_RATING_EXIF_KEYS = (
    "XMP:Rating",
    "XMP-xmp:Rating",
    "Rating",
    "Composite:Rating",
    "EXIF:Rating",
    "IFD0:Rating",
)
# ビューアの2パス目（exiftool CLI）で埋めるタグ
RATING_EXIF_MERGE_KEYS = (
    "XMP:Rating",
    "XMP-xmp:Rating",
    "Rating",
    "Composite:Rating",
)


def merge_exiftool_j_row_into_exif(exif: dict, row: dict) -> None:
    """
    exiftool -getMetadata や -j の1行から、星表示用に Rating 系キーだけを exif 風 dict へ上書きマージ（インプレース）。
    viewer サムネ2パス目・rating_io 追読の共通化。
    """
    for tkey in RATING_EXIF_MERGE_KEYS:
        v = row.get(tkey)
        if v is not None and v != "" and v != "-":
            exif[tkey] = v
    for k, v in row.items():
        if not isinstance(k, str) or k == "SourceFile":
            continue
        if "Rating" not in k or "Region" in k or "iptcExt" in k:
            continue
        if v in (None, "", "-"):
            continue
        exif[k] = v


def is_raw_path(file_path: str) -> bool:
    if not file_path:
        return False
    ext = os.path.splitext(file_path)[1].lower()
    return ext in define.SUPPORTED_FORMATS_RAW


def is_rgb_path(file_path: str) -> bool:
    if not file_path:
        return False
    ext = os.path.splitext(file_path)[1].lower()
    return ext in define.SUPPORTED_FORMATS_RGB


def _coerce_star_int(v):
    if v is None or v == "" or v == "-":
        return None
    try:
        if isinstance(v, str):
            v = v.strip()
            n = int(float(v))
        else:
            n = int(v)
    except (TypeError, ValueError):
        return None
    return max(0, min(_MAX_R, n))


def _deep_find_rating_nested(obj):
    """-j のネスト例: IP 配下の Rating 等。"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            if k == "SourceFile":
                continue
            if isinstance(v, dict):
                n = _deep_find_rating_nested(v)
                if n is not None:
                    return n
                continue
            if k == "Rating" or (k.endswith(":Rating") and "Region" not in k):
                n = _coerce_star_int(v)
                if n is not None:
                    return n
    return None


def parse_exif_rating_value(exif_data: dict) -> int:
    """ExifTool get_metadata の辞書から星を 0～5 に。フラット + ネスト両対応。"""
    if not exif_data:
        return 0
    for k in _RATING_EXIF_KEYS:
        if k not in exif_data:
            continue
        n = _coerce_star_int(exif_data.get(k))
        if n is not None:
            return n
    for key, v in exif_data.items():
        if not isinstance(key, str):
            continue
        if key == "Rating" or key.endswith(":Rating"):
            if "Region" in key or "iptcExt" in key:
                continue
            n = _coerce_star_int(v)
            if n is not None:
                return n
    n2 = _deep_find_rating_nested(exif_data)
    if n2 is not None:
        return n2
    return 0


def exif_had_xmp_rating_tag(exif_data: dict) -> bool:
    if not exif_data:
        return False
    for k in _RATING_EXIF_KEYS:
        v = exif_data.get(k)
        if v not in (None, "", "-"):
            return True
    for key, v in exif_data.items():
        if isinstance(key, str) and (key == "Rating" or key.endswith(":Rating")):
            if v not in (None, "", "-"):
                return True
    return False


def new_rating_on_slot_click(current: int, slot: int) -> int:
    """
    slot: 1～5（左が1）
    同一スロットを再度クリック（= 最後の星）で 0
    """
    if slot < 1 or slot > _MAX_R:
        return current
    if current > 0 and slot == current:
        return 0
    return slot


def effective_rating_display(file_path: str, exif_data: dict, primary_param: dict) -> int:
    """左ペイン・サムネ用の表示用レーティング（整数 0～5）。RAW は .pmck ルート（primary は使わない）。"""
    if is_raw_path(file_path):
        from utils import rating_io

        return rating_io.read_raw_pmck_rating_value(file_path)
    return parse_exif_rating_value(exif_data or {})


def strip_zero_rating_key(param_slice: dict) -> dict:
    """pmck 用: rating が 0 ならキーごと外す。コピーは呼び出し側で。"""
    if not param_slice:
        return param_slice
    if param_slice.get("rating", 0) in (0, None, ""):
        param_slice = param_slice.copy()
        param_slice.pop("rating", None)
    return param_slice
