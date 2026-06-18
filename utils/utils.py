
import math
import cv2
import numpy as np
import json
import logging
import zstandard as zstd

import tempfile
import io
from contextlib import redirect_stdout, redirect_stderr
import time

_IMAGE_SERIALIZE_VERSION = 1
_IMAGE_ZSTD_LEVEL = 3

def to_texture(pos, widget):
    # ウィンドウ座標からローカルイメージ座標に変換
    local_x, local_y = widget.to_widget(*pos)
    local_x = local_x - widget.pos[0]
    local_y = local_y - widget.pos[1]

    # ローカル座標をテクスチャ座標に変換
    tex_y = widget.height-local_y
    tex_x = local_x - (widget.width - widget.texture_size[0])/2
    tex_y = tex_y - (widget.height - widget.texture_size[1])/2

    return (tex_x, tex_y)

def normalize_exif_orientation(value):
    """EXIF Orientation を 1〜8 に正規化する。

    PyExifTool は英語の説明文字列のほか、整数 1〜8 だけを返すことがある。
    整数を str_to_orientation に渡すと常に 1 になっていた（縦横の入れ替えが効かない）。
    """
    if value is None:
        return 1
    if isinstance(value, int):
        return value if 1 <= value <= 8 else 1
    if isinstance(value, str):
        s = value.strip()
        if s.isdigit():
            i = int(s)
            return i if 1 <= i <= 8 else 1
        return str_to_orientation(s)
    return 1


def str_to_orientation(tag):
    """EXIF Orientation 文字列（Exif.Photo にある英語表現）から内部 orient 値へ。

    旧実装では最後の if/else が「Rotate 270 CW」専用になっており、
    「Rotate 90 CW」等は else で 1 に上書きされていて縦構図の寸法ずれになるバグだった。
    """
    if tag == "Horizontal (normal)":
        return 1
    if tag == "Mirror horizontal":
        return 2
    if tag == "Rotate 180":
        return 3
    if tag == "Mirror vertical":
        return 4
    if tag == "Mirror horizontal and rotate 90 CW":
        return 5
    if tag == "Rotate 90 CW":
        return 6
    if tag == "Mirror horizontal and rotate 270 CW":
        return 7
    if tag == "Rotate 270 CW":
        return 8
    return 1

def split_orientation(orientation):
    rad, flip = 0, 0
    if orientation == 1:
        rad, flip = 0, 0
        logging.debug("Horizontal (normal)")
    elif orientation == 2:
        rad, flip = 0, 1
        logging.debug("Mirror horizontal")
    elif orientation == 3:
        rad, flip = math.radians(180), 0
        logging.debug("Rotate 180")
    elif orientation == 4:
        rad, flip = 0, 2
        logging.debug("Mirror vertical")
    elif orientation == 5:
        rad, flip = math.radians(-90), 1
        logging.debug("Mirror horizontal and rotate 90 CW")
    elif orientation == 6:
        rad, flip = math.radians(-90), 0
        logging.debug("Rotate 90 CW")
    elif orientation == 7:
        rad, flip = math.radians(-270), 1
        logging.debug("Mirror horizontal and rotate 270 CW")
    elif orientation == 8:
        rad, flip = math.radians(-270), 0
        logging.debug("Rotate 270 CW")

    return rad, flip

def make_orientation(rotation, flip):
    """
    回転角と反転情報からEXIFオリエンテーションタグの値を生成する関数

    Args:
        rotation (int): 回転角（0, 90, 180, 270のいずれか）
        flip_horizontal (bool, optional): 水平方向の反転
        flip_vertical (bool, optional): 垂直方向の反転

    Returns:
        int: 対応するEXIFオリエンテーションタグの値
    """

    # 回転と反転の組み合わせをマッピング
    orientation_reverse_map = {
        (0, False, False): 1,
        (0, True, False): 2,
        (180, False, False): 3,
        (0, True, True): 3,
        (0, False, True): 4,
        (270, True, False): 5,
        (270, False, False): 6,
        (270, False, True): 7,
        (180, True, False): 4,
        (90, True, False): 7,
        (90, False, False): 8,
        (0, True, True): 3,
        (180, False, True): 8,
        (180, True, True): 5,
        (90, True, True): 8,
        (90, False, True): 7,
        (270, True, True): 6,
    }

    # 入力値のバリデーション
    if rotation not in [0, 90, 180, 270]:
        raise ValueError(f"無効な回転角: {rotation}")
    flip_horizontal = (flip & 1) == 1
    flip_vertical = (flip & 2) == 2

    orientation = orientation_reverse_map.get((rotation, flip_horizontal, flip_vertical))
    
    if orientation is None:
        raise ValueError(f"サポートされていない回転・反転の組み合わせ")

    return orientation

def print_nan_inf(img, label=""):
    result = np.isnan(img)
    nan_count = result.sum()
    result = np.isinf(img)
    inf_count = result.sum()
    if nan_count > 0 or inf_count > 0:
        logging.warning(f"NaN or Inf detected in {label} image. NaN={nan_count}, Inf={inf_count}")

def _byte_shuffle_array(arr: np.ndarray) -> tuple[bytes, str]:
    itemsize = arr.dtype.itemsize
    if itemsize <= 1:
        return arr.tobytes(), "none"

    byte_view = arr.view(np.uint8).reshape(-1, itemsize)
    return byte_view.T.ravel().tobytes(), "byte"


def _byte_unshuffle_array(raw: bytes, dtype: np.dtype, shape) -> np.ndarray:
    itemsize = dtype.itemsize
    if itemsize <= 1:
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()

    shuffled = np.frombuffer(raw, dtype=np.uint8).reshape(itemsize, -1)
    unshuffled = shuffled.T.copy().reshape(-1)
    return unshuffled.view(dtype).reshape(shape).copy()


def convert_image_to_list(img):
    arr = np.ascontiguousarray(img)
    raw, shuffle = _byte_shuffle_array(arr)
    buffer = zstd.ZstdCompressor(level=_IMAGE_ZSTD_LEVEL).compress(raw)

    return {
        "version": _IMAGE_SERIALIZE_VERSION,
        "codec": "zstd",
        "level": _IMAGE_ZSTD_LEVEL,
        "shuffle": shuffle,
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "data": buffer,
    }


def convert_image_from_list(save_data):
    if save_data.get("version") != _IMAGE_SERIALIZE_VERSION:
        raise ValueError(f"Unsupported image serialization version: {save_data.get('version')}")
    if save_data.get("codec") != "zstd":
        raise ValueError(f"Unsupported image serialization codec: {save_data.get('codec')}")

    dtype = np.dtype(save_data["dtype"])
    shape = tuple(save_data["shape"])
    raw = zstd.ZstdDecompressor().decompress(save_data["data"])

    if save_data.get("shuffle") == "byte":
        return _byte_unshuffle_array(raw, dtype, shape)
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()

def pack_uint8_to_uint32(uint8_arr):
    """
    uint8配列をuint32配列にパック（4の倍数でない場合も対応）
    
    Args:
        uint8_arr (np.ndarray): uint8型の入力配列
        
    Returns:
        tuple: (packed_uint32, original_length)
        packed_uint32: パックされたuint32配列
        original_length: 元の配列の長さ（パディングなし）
    """
    # 元の長さを保存
    original_length = len(uint8_arr)
    
    # 4の倍数になるように0でパディング
    pad_len = (4 - (original_length % 4)) % 4
    if pad_len > 0:
        padded = np.pad(uint8_arr, (0, pad_len), mode='constant', constant_values=0)
    else:
        padded = uint8_arr
    
    # パック
    packed = np.frombuffer(padded.tobytes(), dtype=np.uint32)
    
    return packed, original_length

def unpack_uint32_to_uint8(packed_uint32, original_length):
    """
    uint32配列を元のuint8配列に変換
    
    Args:
        packed_uint32 (np.ndarray): pack_uint8_to_uint32で作成した配列
        original_length (int): 元の配列の長さ
        
    Returns:
        np.ndarray: 復元されたuint8配列
    """
    # バイト配列に変換
    byte_arr = np.frombuffer(packed_uint32.tobytes(), dtype=np.uint8)
    
    # 元の長さでトリミング（パディング部分を除去）
    return byte_arr[:original_length]

def array_to_memmap(arr):
    """
    numpy配列をメモリマップドファイルに変換する関数
    
    Args:
        arr (np.ndarray): 入力配列
        
    Returns:
        tuple: (memmap_obj, temp_file_obj)
        memmap_obj: メモリマップされた配列
        temp_file_obj: 一時ファイルオブジェクト（これを保持していないとファイルが消える）
    """
    # 一時ファイルを作成（自動削除されるように設定）
    tfile = tempfile.TemporaryFile()
    
    # 配列の全データをファイルに書き込む
    # メモリマップを作成するために必要なサイズを確保する
    # tofileではなく、memmapを作ってそこにコピーする方が効率的かもしれないが、
    # 確実なのはデータを書き込んでからmmapすること
    
    # メモリマップを作成（書き込み用）
    shape = arr.shape
    dtype = arr.dtype
    
    # データを確保
    #tfile.truncate(arr.nbytes)
    #tfile.flush()
    
    # arrの内容を書き込む
    # np.saveやndarray.tofileはヘッダーがついたりシーク位置がずれる可能性があるため
    # np.memmapを使って直接書き込む
    
    t0 = time.perf_counter()

    mm = np.memmap(tfile, dtype=dtype, mode='w+', shape=shape)
    mm[:] = arr[:]
    mm.flush()
    #mm._mmap.close() # これをしてはいけない。ファイルが閉じられる
    
    t1 = time.perf_counter()
    if arr.nbytes > 1024*1024: # 1MB以上のみログ出力
        logging.info(f"PERF: array_to_memmap write time: {t1-t0:.4f}s. Size: {arr.nbytes/1024/1024:.2f}MB")

    # 読み込みモード（w+なので読み書き可能だが、意図としてはキャッシュ）
    # ファイルオブジェクトが開いている限り、OSが管理する一時ファイルとして存在する
    return mm, tfile

def dict_equal_with_ndarray(dict1: dict, dict2: dict) -> bool:
    """ndarrayを含む辞書を比較"""
    
    # キーが同じかチェック
    if set(dict1.keys()) != set(dict2.keys()):
        return False
    
    for key in dict1:
        val1 = dict1[key]
        val2 = dict2[key]
        
        # ndarrayの場合
        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            if not np.array_equal(val1, val2):
                return False
        
        # 辞書の場合（再帰）
        elif isinstance(val1, dict) and isinstance(val2, dict):
            if not dict_equal_with_ndarray(val1, val2):
                return False
        
        # その他の場合
        elif val1 != val2:
            return False
    
    return True

if __name__ == '__main__':

    img = cv2.imread("your_image.jpg", cv2.IMREAD_UNCHANGED)
    img = img.astype(np.uint16) * 255
    #img = img.astype(np.float32) / 255

    is_success, buffer = cv2.imencode(".jp2", img, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 300])
    pack_buffer, original_len = pack_uint8_to_uint32(buffer)
    list_buffer = pack_buffer.tolist()
    save_data = (list_buffer, original_len)

    with open("your_image.json", 'w') as f:
        json.dump(save_data, f)

    list_buffer, original_len = save_data
    array_buffer = np.array(list_buffer, dtype=np.uint32)
    unpack_buffer = unpack_uint32_to_uint8(array_buffer, original_len)
    img = cv2.imdecode(unpack_buffer, cv2.IMREAD_UNCHANGED)

    cv2.imwrite("your_image.png", img)
