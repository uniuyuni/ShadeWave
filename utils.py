
import math
import numpy as np
from kivy.core.window import Window as KVWindow
#from kivy.clock import KVClock
from kivy.uix.widget import Widget as KVWidget
from screeninfo import get_monitors
import pillow_heif
from pillow_heif import register_heif_opener

register_heif_opener()

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

def tone_map(x, threshold=1.0):
    return np.where(x > threshold, np.log1p(x - threshold) + threshold, x)

def soft_clip(x, threshold=1.0):
    return threshold * np.tanh(x / threshold)

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

def dpi_scale_width(ref):
    return ref * (KVWindow.dpi / 96)
    #return ref * (KVWindow.width / 1200)

def dpi_scale_height(ref):
    return ref * (KVWindow.dpi / 96)
    #return ref * (KVWindow.height / 800)


def adjust_to_multiple(image, size=8, mode='constant'):
    # 画像の高さと幅を取得
    h, w = image.shape[:2]
    
    # 8の倍数に切り上げた新しいサイズを計算
    new_h = (h + size-1) // size * size
    new_w = (w + size-1) // size * size
    
    # パディング量を計算
    pad_h = new_h - h
    pad_w = new_w - w
    
    # パディング幅を設定（次元ごとに指定）
    pad_width = [(0, pad_h), (0, pad_w)] + [(0, 0)] * (image.ndim - 2)
    
    # 画像の下側と右側をエッジ値でパディング
    padded_image = np.pad(image, pad_width=pad_width, mode=mode)
    
    return padded_image, (h, w)

def restore_original_size(padded_image, original_size):
    # 元のサイズを取得
    h_orig, w_orig = original_size
    
    # パディングされた部分を切り取って元のサイズに復元
    return padded_image[:h_orig, :w_orig, ...]

def convert_image_to_list(image):
    height, width = image.shape[:2]
    heif_file = pillow_heif.from_bytes("RGB;16", (width, height), (image * 65535).astype(np.uint16).tobytes())
    print(image.shape[:2])
    return (list(heif_file.data), (width, height))

def convert_image_from_list(list, shape):
    heif_file = pillow_heif.from_bytes("RGB;16", shape, bytes(list))
    return np.asarray(heif_file).astype(np.float32) / 65535
