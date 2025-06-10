
from importlib.machinery import BYTECODE_SUFFIXES
import math
import cv2
import numpy as np
from kivy.core.window import Window as KVWindow
from kivy.uix.widget import Widget as KVWidget
from pillow_heif.options import QUALITY
from screeninfo import get_monitors
import json
import base64
import zlib
import numpy as np
from typing import Any, Dict

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

def str_to_orientation(str):
    if str == "Horizontal (normal)":
        orientation = 1
    if str == "Mirror horizontal":
        orientation = 2
    if str == "Rotate 180":
        orientation = 3
    if str  == "Mirror vertical":
        orientation = 4
    if str == "Mirror horizontal and rotate 90 CW":
        orientation = 5
    if str == "Rotate 90 CW":
        orientation = 6
    if str == "Mirror horizontal and rotate 270 CW":
        orientation = 7
    if str == "Rotate 270 CW":
        orientation = 8
    else:
        orientation = 1

    return orientation

def split_orientation(orientation):
    rad, flip = 0, 0
    if orientation == 1:
        rad, flip = 0, 0
        print("Horizontal (normal)")
    elif orientation == 2:
        rad, flip = 0, 1
        print("Mirror horizontal")
    elif orientation == 3:
        rad, flip = math.radians(180), 0
        print("Rotate 180")
    elif orientation == 4:
        rad, flip = 0, 2
        print("Mirror vertical")
    elif orientation == 5:
        rad, flip = math.radians(-90), 1
        print("Mirror horizontal and rotate 90 CW")
    elif orientation == 6:
        rad, flip = math.radians(-90), 0
        print("Rotate 90 CW")
    elif orientation == 7:
        rad, flip = math.radians(-270), 1
        print("Mirror horizontal and rotate 270 CW")
    elif orientation == 8:
        rad, flip = math.radians(-270), 0
        print("Rotate 270 CW")

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

def print_nan_inf(img):
    result = np.isnan(img)
    count = result.sum()
    print("NaN=", count)
    result = np.isinf(img)
    count = result.sum()
    print("inf=", count)


def convert_to_float32(img):
    """
    画像のデータ型をfloat32に変換する関数

    Args:
        img (numpy.ndarray): 変換する画像データ

    Returns:
        numpy.ndarray: float32の画像データ
    """
    if img.dtype == np.uint8:
        img = img.astype(np.float32)/255
    elif img.dtype == np.uint16 or img.dtype == '>u2' or img.dtype == '<u2':
        img = img.astype(np.float32)/65535
    elif img.dtype == np.uint32 or img.dtype == '>u4' or img.dtype == '<u4':
        img = img.astype(np.float32)/4294967295
    elif img.dtype == np.uint64:
        img = img.astype(np.float32)/18446744073709551615
    elif img.dtype == np.int8:
        img = img.astype(np.float32)/127
    elif img.dtype == np.int16:
        img = img.astype(np.float32)/32767
    elif img.dtype == np.int32:
        img = img.astype(np.float32)/2147483647
    elif img.dtype == np.int64:
        img = img.astype(np.float32)/9223372036854775807
    elif img.dtype == np.float16:
        img = img.astype(np.float32)
    elif img.dtype == np.float32:
        pass
    elif img.dtype == np.float64:
        img = img.astype(np.float32)
    else:
        raise ValueError(f"サポートされていないデータ型: {img.dtype}")

    return img

def get_current_dispay():
    # 現在のウィンドウの左上座標
    win_x, win_y = KVWindow.left, KVWindow.top

    # モニタ一覧を取得して、ウィンドウが属しているモニタを探す
    monitors = get_monitors()

    for i, m in enumerate(monitors):
        if m.is_primary == True:
            primary = m
            break

    for i, m in enumerate(monitors):
        if m.y != 0:
            m.y = -m.height if m.y > 0 else primary.height
        if m.x <= win_x < m.x + m.width and m.y <= win_y < m.y + m.height:
            return {"display": i, "width": m.width, "height": m.height, "is_primary": m.is_primary}
    
    return None

def get_entire_widget_tree(root, delay=0.1):
    """全ウィジェット取得（未表示含む）"""
    results = []
    
    def _collect(w):
        if not isinstance(w, KVWidget):
            return
            
        results.append(w)
        
        # 特殊レイアウト対応
        if hasattr(w, 'tab_list'):  # TabbedPanel
            for tab in w.tab_list:
                _collect(tab.content)
                
        if hasattr(w, 'screens'):  # ScreenManager
            for screen in w.screens:
                _collect(screen)
                
        # 通常の子要素
        for child in w.children:
            _collect(child)
    
    # 遅延実行で未初期化要素に対応
    #KVClock.schedule_once(lambda dt: _collect(root), delay)
    _collect(root)

    return results

def traverse_widget(root):
    # すべてのスケールが必要なウィジェットを更新
    if root:
        for child in get_entire_widget_tree(root):
            if hasattr(child, 'ref_width'):
                child.width = dpi_scale_width(child.ref_width)
            if hasattr(child, 'ref_height'):
                child.height = dpi_scale_height(child.ref_height)
            if hasattr(child, 'ref_padding'):
                child.padding = dpi_scale_width(child.ref_padding)
            if hasattr(child, 'ref_spacing'):
                child.spacing = dpi_scale_width(child.ref_spacing)
            if hasattr(child, 'ref_tab_width'):
                child.tab_width = dpi_scale_width(child.ref_tab_width)
            if hasattr(child, 'ref_tab_height'):
                child.tab_height = dpi_scale_height(child.ref_tab_height)

def dpi_scale_width(ref):
    return ref * (KVWindow.dpi / 96)
    #return ref * (KVWindow.width / 1200)

def dpi_scale_height(ref):
    return ref * (KVWindow.dpi / 96)
    #return ref * (KVWindow.height / 800)

def convert_image_to_list(image):
    # 画像を処理できる方に変換
    img = (image * 65535).astype(np.uint16)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 圧縮＆パック
    is_success, buffer = cv2.imencode(".jp2", img, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 1000])
    if is_success is None:
        return None

    pack_buffer, original_len = pack_uint8_to_uint32(buffer)
    list_buffer = pack_buffer #pack_buffer.tolist()
    save_data = (list_buffer, original_len)

    return save_data

def convert_image_from_list(save_data):
    # データを復元
    list_buffer, original_len = save_data
    array_buffer = list_buffer #np.array(list_buffer, dtype=np.uint32)
    unpack_buffer = unpack_uint32_to_uint8(array_buffer, original_len)
    img = cv2.imdecode(unpack_buffer, cv2.IMREAD_UNCHANGED)

    # 画像を処理できる方に変換
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 65535

    return image

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


class CompactNumpyEncoder(json.JSONEncoder):
    """NumPyデータを最小容量で保存するカスタムエンコーダ"""
    
    def default(self, obj: Any) -> Any:
        # NumPy配列の処理
        if isinstance(obj, np.ndarray):
            return self._compress_array(obj)
        
        # NumPyスカラーの処理
        if isinstance(obj, np.generic):
            return obj.item()
            
        return super().default(obj)
    
    def _compress_array(self, array: np.ndarray) -> Dict[str, Any]:
        """配列を圧縮してBase64エンコード"""
        # データをバイト列に変換
        data_bytes = array.tobytes()
        
        # zlibで圧縮 (レベル9で最大圧縮)
        compressed = data_bytes #zlib.compress(data_bytes, level=9)
        
        # Base64エンコード
        encoded = base64.b64encode(compressed).decode('ascii')
        
        return {
            '__numpy_array__': True,
            'dtype': str(array.dtype),
            'shape': array.shape,
            'data': encoded
        }

def compact_numpy_decoder(obj: Dict) -> Any:
    """圧縮されたNumPyデータを復元"""
    if '__numpy_array__' in obj:
        # Base64デコード
        decoded = base64.b64decode(obj['data'])
        
        # zlib解凍
        decompressed = decoded #zlib.decompress(decoded)
        
        # NumPy配列に変換
        array = np.frombuffer(decompressed, dtype=np.dtype(obj['dtype']))
        return array.reshape(obj['shape'])
    
    return obj


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
