"""
EXR (OpenEXR) 読み込みユーティリティ。

pyvips は EXR 非対応のため、ここでは OpenEXR 3.x の新 API (`OpenEXR.File`) を使う。
旧 API (`OpenEXR.InputFile` + `Imath.PixelType`) ではない点に注意。

EXR は基本リニア（シーンリニア）で、HDR 値 (>1.0) を含みうる。
色空間はヘッダの `chromaticities` 属性で決まる：
  - 無い場合: EXR 仕様により Rec.709/sRGB 原色のリニアとして扱う。
  - ある場合: その原色から RGB→XYZ 行列を構成し、任意の原色（ACEScg / ACES2065-1 など）を厳密に扱う。

`chromaticities` のタプル並びは (Rx, Ry, Gx, Gy, Bx, By, Wx, Wy) で、
colour_functions.RGBColourspace の primaries(=Rx..By) + whitepoint(=Wx,Wy) とそのまま一致する。
"""

import logging
import numpy as np

import OpenEXR

from effect_backends import colour_functions_adapter as cf


# chromaticities マッチング許容差（原色座標の絶対誤差）
_CHROMA_ATOL = 2.0e-3

# chromaticities が一致したら名前付き色空間を優先的に使う（ログが分かりやすく、
# 既定の高精度行列を使えるため）。順序は判定優先度。
_KNOWN_LINEAR_SPACES = (
    'sRGB',              # = Rec.709 原色（EXR の既定でもある）
    'ACEScg',
    'ACES2065-1',
    'Display P3',
    'Rec.2020',
    'Adobe RGB (1998)',
)


def _extract_rgb(part) -> np.ndarray:
    """
    OpenEXR.File の 1 パートから (H, W, 3) の RGB float32 を取り出す。

    combined モード（separate_channels=False）では R/G/B/A が別チャンネル保存でも
    自動で 'RGB'/'RGBA' に結合される。それが無いファイル（Y 単色など）は
    separate モードで読み直してフォールバックする。
    """
    chans = part.channels
    if 'RGBA' in chans:
        px = chans['RGBA'].pixels[..., :3]
    elif 'RGB' in chans:
        px = chans['RGB'].pixels
    else:
        # 個別チャンネル名で組み立てる
        names = set(chans.keys())
        if {'R', 'G', 'B'} <= names:
            px = np.stack([chans['R'].pixels, chans['G'].pixels, chans['B'].pixels], axis=-1)
        elif 'Y' in names:
            # 輝度のみ（グレースケール EXR）
            y = chans['Y'].pixels
            px = np.stack([y, y, y], axis=-1)
        else:
            raise ValueError(f"EXR にRGB/輝度チャンネルが見つかりません: {sorted(names)}")
    return np.ascontiguousarray(px, dtype=np.float32)


def read_exr(file_path):
    """
    EXR をリニア RGB float32 (H, W, 3) として読み込む。

    Returns
    -------
    (img, chromaticities)
        img : np.ndarray, shape (H, W, 3), float32, シーンリニア（HDR 値を保持）
        chromaticities : tuple[float] | None  ヘッダの chromaticities（8要素）または None
    """
    # separate_channels=False で R/G/B(/A) を自動結合
    with OpenEXR.File(file_path, separate_channels=False) as f:
        if not f.parts:
            raise ValueError(f"EXR に読み取り可能な part がありません: {file_path}")
        part = f.parts[0]
        chroma = part.header.get('chromaticities', None)
        if chroma is not None:
            chroma = tuple(float(v) for v in np.asarray(chroma).ravel())

        try:
            img = _extract_rgb(part)
        except ValueError:
            # combined で取れない場合は separate で読み直す
            with OpenEXR.File(file_path, separate_channels=True) as f2:
                if not f2.parts:
                    raise ValueError(f"EXR に読み取り可能な part がありません: {file_path}")
                img = _extract_rgb(f2.parts[0])

    return img, chroma


def resolve_input_colourspace(chromaticities) -> str:
    """
    EXR の chromaticities からソース色空間名を決める。

    - None: 'Linear sRGB'（= Rec.709 原色リニア。EXR 既定）
    - 既知の原色に一致: その名前付き色空間
    - それ以外: chromaticities から一意名の色空間を生成し RGB_COLOURSPACES に登録して返す

    いずれも「リニア」前提の名前を返す（呼び出し側は apply_cctf_decoding=False で使う）。
    """
    if chromaticities is None:
        return 'Linear sRGB'

    prim = np.asarray(chromaticities[:6], dtype=np.float64)
    wp = np.asarray(chromaticities[6:8], dtype=np.float64)

    for name in _KNOWN_LINEAR_SPACES:
        cs = cf.RGB_COLOURSPACES.get(name)
        if cs is None:
            continue
        if np.allclose(cs.primaries, prim, atol=_CHROMA_ATOL) and \
           np.allclose(cs.whitepoint, wp, atol=_CHROMA_ATOL):
            logging.info(f"EXR chromaticities matched colourspace: {name}")
            return name

    # 既知に無い任意原色（カメラ独自空間など）: chromaticities から色空間を構成
    key = ",".join(f"{v:.5f}" for v in (*prim, *wp))
    name = f"EXR[{key}]"
    if name not in cf.RGB_COLOURSPACES:
        cf.RGB_COLOURSPACES[name] = cf.RGBColourspace(
            name=name, primaries=prim, whitepoint=wp, whitepoint_name='EXR'
        )
        logging.info(f"EXR custom chromaticities registered: primaries={prim.tolist()} wp={wp.tolist()}")
    return name


def read_exr_linear(file_path, output_colourspace='ProPhoto RGB', cat='Bradford'):
    """
    EXR を読み、ソース色空間（chromaticities 由来）から output_colourspace のリニアへ変換する。
    HDR を保持するためガンマ補正もガマットマッピングも行わない。
    """
    img, chroma = read_exr(file_path)
    src = resolve_input_colourspace(chroma)
    if src == output_colourspace:
        return img, src
    out = cf.RGB_to_RGB(
        img, src, output_colourspace, cat,
        apply_cctf_decoding=False,   # 既にリニア
        apply_cctf_encoding=False,   # リニアのまま返す
        apply_gamut_mapping=False,   # HDR(>1) をクリップしない
    ).astype(np.float32)
    return out, src


def read_exr_thumbnail(file_path):
    """
    サムネイル表示用に EXR を sRGB 表示空間の [0,1] float32 へトーンマップして返す。
    Reinhard グローバルトーンマップ + sRGB ガンマ。
    """
    img, chroma = read_exr(file_path)
    src = resolve_input_colourspace(chroma)
    # 表示色域（sRGB）リニアへ
    rgb = cf.RGB_to_RGB(
        img, src, 'sRGB', 'Bradford',
        apply_cctf_decoding=False, apply_cctf_encoding=False, apply_gamut_mapping=False,
    ).astype(np.float32)
    # 負値を除去し Reinhard トーンマップ
    rgb = np.clip(rgb, 0.0, None)
    rgb = rgb / (1.0 + rgb)
    # sRGB ガンマで表示エンコード
    rgb = cf.linear_to_sRGB(rgb).astype(np.float32)
    return np.clip(rgb, 0.0, 1.0)
