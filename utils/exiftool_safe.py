"""
exiftool 呼び出しの堅牢化ヘルパー。

設計の要点:
- 「ファイルが存在しない／壊れている／途中で消えた」はリトライしない（無限に失敗するため）
  → 事前に os.path.isfile で除外、または分割して失敗ファイルだけスキップ。
- 「タイムアウト・一時的 I/O エラー」はリトライ対象
  → 指数バックオフで小回数だけ再試行。
- 結果リストは必ず入力 files と同じ長さ・順序を返す（取れなかったファイルは空 dict）。
"""

from __future__ import annotations

import logging
import json
import os
import subprocess
import time
from typing import Iterable, Sequence

from exiftool.exceptions import (
    ExifToolOutputEmptyError,
)

_UNGROUPED_ALIAS_PRIORITY: dict[str, tuple[str, ...]] = {
    # System / file summary
    "FileName": ("System:FileName",),
    "Directory": ("System:Directory",),
    "FileSize": ("System:FileSize",),
    "FileModifyDate": ("System:FileModifyDate",),
    "FileAccessDate": ("System:FileAccessDate",),
    "FileInodeChangeDate": ("System:FileInodeChangeDate",),
    "FilePermissions": ("System:FilePermissions",),
    "FileType": ("File:FileType",),
    "FileTypeExtension": ("File:FileTypeExtension",),
    "MIMEType": ("File:MIMEType",),
    "ImageWidth": ("File:ImageWidth", "PNG:ImageWidth", "SubIFD:ImageWidth", "IFD0:ImageWidth", "SubIFD1:ImageWidth", "IFD1:ImageWidth"),
    "ImageHeight": ("File:ImageHeight", "PNG:ImageHeight", "SubIFD:ImageHeight", "IFD0:ImageHeight", "SubIFD1:ImageHeight", "IFD1:ImageHeight"),
    "ImageSize": ("Composite:ImageSize",),

    # Common camera metadata used by the app
    "Make": ("IFD0:Make", "IFD1:Make"),
    "Model": ("IFD0:Model", "IFD1:Model"),
    "Software": ("IFD0:Software", "IFD1:Software", "XMP-xmp:CreatorTool"),
    "Orientation": ("IFD0:Orientation", "XMP-tiff:Orientation", "IFD1:Orientation", "SubIFD1:Orientation"),
    "CreateDate": ("ExifIFD:CreateDate", "XMP-xmp:CreateDate"),
    "ModifyDate": ("IFD0:ModifyDate", "XMP-xmp:ModifyDate", "IFD1:ModifyDate"),
    "DateCreated": ("IPTC:DateCreated", "IPTC2:DateCreated", "IPTC3:DateCreated", "XMP-photoshop:DateCreated"),
    "ISO": ("ExifIFD:ISO", "Pentax:ISO"),
    "ApertureValue": ("ExifIFD:ApertureValue",),
    "Aperture": ("Composite:Aperture",),
    "ShutterSpeedValue": ("ExifIFD:ShutterSpeedValue",),
    "ExposureCompensation": ("ExifIFD:ExposureCompensation", "Sony:ExposureCompensation", "Pentax:ExposureCompensation"),
    "Flash": ("ExifIFD:Flash",),
    "WhiteBalance": ("FujiFilm:WhiteBalance", "ExifIFD:WhiteBalance", "Sony:WhiteBalance", "Pentax:WhiteBalance", "XMP-crs:WhiteBalance"),
    "FocalLength": ("ExifIFD:FocalLength", "Pentax:FocalLength"),
    "PictureMode": ("FujiFilm:PictureMode", "Pentax:PictureMode"),
    "LensMake": ("ExifIFD:LensMake", "XMP-aux:LensMake"),
    "LensModel": ("ExifIFD:LensModel", "XMP-exifEX:LensModel"),
    "LensInfo": ("ExifIFD:LensInfo", "XMP-aux:LensInfo", "XMP-exif:LensInfo"),
    "LensSerialNumber": ("ExifIFD:LensSerialNumber", "XMP-aux:LensSerialNumber", "XMP-exif:LensSerialNumber"),
    "SubjectDistanceRange": ("ExifIFD:SubjectDistanceRange",),

    # RAW geometry and embedded previews
    "RawImageCropTopLeft": ("RAF:RawImageCropTopLeft", "RAF2:RawImageCropTopLeft"),
    "RawImageCroppedSize": ("RAF:RawImageCroppedSize", "RAF2:RawImageCroppedSize"),
    "RawImageSize": ("RAF:RawImageSize", "Pentax:RawImageSize", "RAF2:RawImageSize"),
    "FullImageSize": ("Sony:FullImageSize",),
    "ThumbnailImage": ("IFD1:ThumbnailImage",),
    "PreviewImage": ("File:PreviewImage", "IFD0:PreviewImage", "Pentax:PreviewImage"),
    "JpgFromRaw": ("IFD2:JpgFromRaw",),
    "PreviewTIFF": ("IFD0:PreviewTIFF", "IFD1:PreviewTIFF", "SubIFD1:PreviewTIFF"),
    "OtherImage": ("All:OtherImage", "File:OtherImage", "IFD0:OtherImage", "IFD1:OtherImage", "SubIFD1:OtherImage"),
    "ThumbnailTIFF": ("All:ThumbnailTIFF", "JFXX:ThumbnailTIFF", "IFD1:ThumbnailTIFF", "IFD0:ThumbnailTIFF", "SubIFD1:ThumbnailTIFF"),
    "WB_GRBLevels": ("FujiIFD:WB_GRBLevels",),

    # Rating must prefer XMP over embedded/camera maker tags.
    "Rating": ("XMP-xmp:Rating", "XMP:Rating", "IFD0:Rating", "Sony:Rating", "FujiFilm:Rating", "Composite:Rating"),

    # Export metadata allowlist / duplicate TIFF-ish fields
    "BitsPerSample": ("FujiIFD:BitsPerSample", "SubIFD:BitsPerSample", "File:BitsPerSample", "IFD0:BitsPerSample"),
    "Compression": ("SubIFD:Compression", "IFD0:Compression", "IFD1:Compression", "PNG:Compression"),
    "PhotometricInterpretation": ("SubIFD:PhotometricInterpretation", "IFD0:PhotometricInterpretation"),
    "RowsPerStrip": ("SubIFD:RowsPerStrip", "IFD0:RowsPerStrip", "IFD1:RowsPerStrip"),
    "SamplesPerPixel": ("SubIFD:SamplesPerPixel", "IFD0:SamplesPerPixel"),
    "Sharpness": ("ExifIFD:Sharpness", "XMP-crs:Sharpness", "FujiFilm:Sharpness", "Pentax:Sharpness", "Sony:Sharpness"),
    "StripByteCounts": ("FujiIFD:StripByteCounts", "IFD0:StripByteCounts", "SubIFD:StripByteCounts", "IFD1:StripByteCounts"),
    "StripOffsets": ("FujiIFD:StripOffsets", "IFD0:StripOffsets", "SubIFD:StripOffsets", "IFD1:StripOffsets"),
    "SubfileType": ("IFD0:SubfileType", "SubIFD:SubfileType", "SubIFD1:SubfileType", "IFD1:SubfileType"),
    "Credit": ("IPTC:Credit", "XMP-photoshop:Credit"),
    "Format": ("XMP-dc:Format", "CBOR:Format"),
    "InstanceID": ("XMP-xmpMM:InstanceID", "CBOR:InstanceID"),

    # Additional duplicate short names observed in ../test_photos.
    "ApplicationRecordVersion": ("IPTC:ApplicationRecordVersion", "IPTC2:ApplicationRecordVersion", "IPTC3:ApplicationRecordVersion"),
    "Artist": ("IFD0:Artist", "IFD1:Artist"),
    "BrightnessValue": ("ExifIFD:BrightnessValue", "Sony:BrightnessValue"),
    "By-line": ("IPTC:By-line", "IPTC2:By-line", "IPTC3:By-line"),
    "ChromaticAberrationCorrParams": ("SR2SubIFD:ChromaticAberrationCorrParams",),
    "ColorMode": ("XMP-photoshop:ColorMode", "Sony:ColorMode", "SR2DataIFD3:ColorMode"),
    "ColorSpace": ("ExifIFD:ColorSpace", "Pentax:ColorSpace", "XMP-exif:ColorSpace"),
    "Contrast": ("ExifIFD:Contrast", "FujiFilm:Contrast"),
    "DependentImage1EntryNumber": ("MPImage1:DependentImage1EntryNumber", "MPImage2:DependentImage1EntryNumber"),
    "DependentImage2EntryNumber": ("MPImage1:DependentImage2EntryNumber", "MPImage2:DependentImage2EntryNumber"),
    "DistortionCorrParams": ("SR2SubIFD:DistortionCorrParams",),
    "ExifImageHeight": ("ExifIFD:ExifImageHeight", "XMP-exif:ExifImageHeight"),
    "ExifImageWidth": ("ExifIFD:ExifImageWidth", "XMP-exif:ExifImageWidth"),
    "ExposureMode": ("ExifIFD:ExposureMode",),
    "ExposureProgram": ("ExifIFD:ExposureProgram", "Sony:ExposureProgram"),
    "ExposureTime": ("ExifIFD:ExposureTime", "Pentax:ExposureTime"),
    "FNumber": ("ExifIFD:FNumber", "Pentax:FNumber"),
    "FujiLayout": ("RAF:FujiLayout", "RAF2:FujiLayout"),
    "ImageDescription": ("IFD0:ImageDescription", "IFD1:ImageDescription"),
    "Lens": ("XMP-aux:Lens", "XMP-exif:Lens"),
    "MPImageFlags": ("MPImage1:MPImageFlags", "MPImage2:MPImageFlags"),
    "MPImageFormat": ("MPImage1:MPImageFormat", "MPImage2:MPImageFormat"),
    "MPImageLength": ("MPImage2:MPImageLength",),
    "MPImageStart": ("MPImage2:MPImageStart",),
    "MPImageType": ("MPImage2:MPImageType",),
    "MeteringMode": ("ExifIFD:MeteringMode", "Sony:MeteringMode", "Pentax:MeteringMode"),
    "PlanarConfiguration": ("IFD0:PlanarConfiguration", "SubIFD:PlanarConfiguration"),
    "RawExposureBias": ("RAF:RawExposureBias", "RAF2:RawExposureBias"),
    "RawImageFullSize": ("RAF:RawImageFullSize", "RAF2:RawImageFullSize"),
    "RelativeExposure": ("RAF:RelativeExposure",),
    "ResolutionUnit": ("IFD0:ResolutionUnit", "JFIF:ResolutionUnit", "SubIFD:ResolutionUnit", "IFD1:ResolutionUnit", "IFD2:ResolutionUnit", "Photoshop:ResolutionUnit", "XMP-tiff:ResolutionUnit"),
    "Saturation": ("ExifIFD:Saturation", "FujiFilm:Saturation", "XMP-crs:Saturation"),
    "SerialNumber": ("ExifIFD:SerialNumber", "Pentax:SerialNumber", "XMP-aux:SerialNumber", "XMP-exif:SerialNumber"),
    "VignettingCorrParams": ("SR2SubIFD:VignettingCorrParams",),
    "WB_GRGBLevels": ("RAF:WB_GRGBLevels",),
    "WB_GRGBLevelsAuto": ("RAF:WB_GRGBLevelsAuto",),
    "WB_GRGBLevelsCloudy": ("RAF:WB_GRGBLevelsCloudy",),
    "WB_GRGBLevelsDayWhiteFluor": ("RAF:WB_GRGBLevelsDayWhiteFluor",),
    "WB_GRGBLevelsDaylight": ("RAF:WB_GRGBLevelsDaylight",),
    "WB_GRGBLevelsDaylightFluor": ("RAF:WB_GRGBLevelsDaylightFluor",),
    "WB_GRGBLevelsLivingRoomWarmWhiteFluor": ("RAF:WB_GRGBLevelsLivingRoomWarmWhiteFluor",),
    "WB_GRGBLevelsTungsten": ("RAF:WB_GRGBLevelsTungsten",),
    "WB_GRGBLevelsWarmWhiteFluor": ("RAF:WB_GRGBLevelsWarmWhiteFluor",),
    "WB_GRGBLevelsWhiteFluorescent": ("RAF:WB_GRGBLevelsWhiteFluorescent",),
    "XResolution": ("IFD0:XResolution", "JFIF:XResolution", "SubIFD:XResolution", "IFD1:XResolution", "IFD2:XResolution", "Photoshop:XResolution", "XMP-tiff:XResolution"),
    "YCbCrPositioning": ("IFD0:YCbCrPositioning", "IFD1:YCbCrPositioning"),
    "YResolution": ("IFD0:YResolution", "JFIF:YResolution", "SubIFD:YResolution", "IFD1:YResolution", "IFD2:YResolution", "Photoshop:YResolution", "XMP-tiff:YResolution"),
}


def safe_run_exiftool(
    argv: Sequence[str],
    *,
    timeout: float = 120,
    retries: int = 2,
    backoff: float = 0.3,
) -> subprocess.CompletedProcess:
    """
    subprocess.run のリトライ付きラッパー。

    再試行する例外: TimeoutExpired, OSError（一時的な FD 枯渇など）。
    再試行しない: FileNotFoundError（バイナリ自体がない → 設定ミス、即失敗）。
    returncode != 0 はそのまま返す（呼び出し側で判定する）。
    """
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return subprocess.run(
                list(argv),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError:
            raise
        except (subprocess.TimeoutExpired, OSError) as e:
            last_exc = e
            if attempt >= retries:
                break
            time.sleep(backoff * (2 ** attempt))
            logging.warning("safe_run_exiftool: 一時エラー %r。再試行 %d/%d", e, attempt + 1, retries)
    assert last_exc is not None
    raise last_exc


def safe_get_metadata(
    files: Iterable[str],
    *,
    common_args: Sequence[str] | None = None,
    timeout: float = 120,
) -> list[dict]:
    """
    exiftool -j の堅牢版。

    - 存在しないファイルは事前に除外し、その位置には空 dict を入れる。
    - timeout / JSON 解析失敗 / stdout 空なら chunk を二分して再帰し、最終的に
      1 件単位で試行する。1 件で失敗したファイルは空 dict。
    - exiftool バイナリなし等は全件空 dict で返し、警告ログのみ出して
      呼び出し側（スピナー解除等）を止めない。
    - 戻り値は入力 files と同じ長さ・順序。
    """
    file_list = list(files)
    n = len(file_list)
    results: list[dict] = [{} for _ in range(n)]
    if n == 0:
        return results

    alive_indices: list[int] = []
    alive_paths: list[str] = []
    for i, p in enumerate(file_list):
        try:
            if p and os.path.isfile(p):
                alive_indices.append(i)
                alive_paths.append(p)
        except OSError:
            pass
    if not alive_paths:
        return results

    _collect_metadata_subprocess(
        alive_paths,
        alive_indices,
        results,
        list(common_args) if common_args else [],
        timeout,
    )

    return results


def _collect_metadata_subprocess(
    paths: list[str],
    indices: list[int],
    results: list[dict],
    common_args: list[str],
    timeout: float,
) -> None:
    """ヘルパー: paths を取り、失敗したら二分して再試行、最終的に 1 件単位。"""
    if not paths:
        return
    try:
        rows = _run_metadata_subprocess(paths, common_args, timeout)
    except FileNotFoundError as e:
        logging.warning("safe_get_metadata: exiftool バイナリが見つかりません: %s", e)
        return
    except (OSError, subprocess.SubprocessError, ExifToolOutputEmptyError, json.JSONDecodeError, ValueError) as e:
        if len(paths) == 1:
            logging.warning("safe_get_metadata: 取得失敗（スキップ）: %s (%s)", paths[0], e)
            return
        mid = len(paths) // 2
        _collect_metadata_subprocess(paths[:mid], indices[:mid], results, common_args, timeout)
        _collect_metadata_subprocess(paths[mid:], indices[mid:], results, common_args, timeout)
        return

    if not isinstance(rows, list):
        return
    index_by_source = {}
    index_by_basename = {}
    for path, idx in zip(paths, indices):
        key = _norm_path_key(path)
        index_by_source[key] = idx
        index_by_basename.setdefault(os.path.basename(path), idx)
    fallback_iter = iter(indices)
    for row in rows:
        if not isinstance(row, dict):
            continue
        _add_ungrouped_aliases(row)
        idx = None
        source_file = row.get("SourceFile")
        if source_file:
            idx = index_by_source.get(_norm_path_key(str(source_file)))
            if idx is None:
                idx = index_by_basename.get(os.path.basename(str(source_file)))
        if idx is None:
            idx = next(fallback_iter, None)
        if idx is not None:
            results[idx] = row


def _run_metadata_subprocess(
    paths: list[str],
    common_args: list[str],
    timeout: float,
) -> list[dict]:
    argv = ["exiftool", "-j", *common_args, *paths]
    proc = safe_run_exiftool(argv, timeout=timeout, retries=0)
    stdout = proc.stdout or ""
    if not stdout.strip():
        stderr = (proc.stderr or "").strip()
        raise ExifToolOutputEmptyError(proc.returncode, stdout, stderr, argv)
    try:
        rows = json.loads(stdout)
    except json.JSONDecodeError:
        logging.warning(
            "safe_get_metadata: JSON 解析失敗 returncode=%s stderr=%s",
            proc.returncode,
            (proc.stderr or "").strip()[:400],
        )
        raise
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()[:400]
        if stderr:
            logging.info("safe_get_metadata: exiftool warning returncode=%s: %s", proc.returncode, stderr)
    if not isinstance(rows, list):
        raise ValueError("exiftool JSON root is not a list")
    return rows


def _norm_path_key(path: str) -> str:
    try:
        return os.path.normcase(os.path.abspath(path))
    except OSError:
        return os.path.normcase(path or "")


def _add_ungrouped_aliases(row: dict) -> None:
    """
    -G1 で取得したタグを既存コード用の非グループ名でも参照できるようにする。
    グループ付きキー自体は残すため、XMP-xmp:Rating などの衝突回避情報も保持される。
    """
    grouped_by_short: dict[str, list[tuple[str, object]]] = {}
    for key, value in list(row.items()):
        if not isinstance(key, str) or ":" not in key:
            continue
        short_key = key.rsplit(":", 1)[-1]
        if short_key:
            grouped_by_short.setdefault(short_key, []).append((key, value))

    for short_key, priority in _UNGROUPED_ALIAS_PRIORITY.items():
        if short_key in row:
            continue
        source_ext = os.path.splitext(str(row.get("SourceFile") or ""))[1].lower()
        if short_key in ("StripByteCounts", "StripOffsets") and source_ext == ".3fr":
            grouped_key = f"SubIFD:{short_key}"
            if grouped_key in row:
                row[short_key] = row[grouped_key]
                continue
        if short_key == "SubfileType" and source_ext in (".3fr", ".arw"):
            if "SubIFD:SubfileType" in row:
                row[short_key] = row["SubIFD:SubfileType"]
                continue
        if (
            short_key == "WhiteBalance"
            and row.get("File:FileType") == "TIFF"
            and "XMP-crs:WhiteBalance" in row
        ):
            row[short_key] = row["XMP-crs:WhiteBalance"]
            continue
        for grouped_key in priority:
            if grouped_key in row:
                row[short_key] = row[grouped_key]
                break

    for short_key, grouped_items in grouped_by_short.items():
        if short_key in row or short_key in _UNGROUPED_ALIAS_PRIORITY:
            continue
        if len(grouped_items) == 1:
            row[short_key] = grouped_items[0][1]
