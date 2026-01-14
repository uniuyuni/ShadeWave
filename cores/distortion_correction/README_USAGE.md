# 画像歪み補正ライブラリ - 使用方法

## インストール

```bash
# ディレクトリをPythonパスに追加
import sys
sys.path.insert(0, 'path/to/distortion_correction')
```

または、Kivyアプリケーションのcoresディレクトリに配置：
```
your_app/
  cores/
    distortion_correction/
      __init__.py
      trapezoid_correction_3d.py
      ...
```

## インポート

```python
# Kivyアプリ内
from cores.distortion_correction import (
    correct_trapezoid,
    correct_four_points,
    correct_lens_distortion,
    warp_mesh
)

# スタンドアロン（パッケージとして）
# ディレクトリ名を任意に変更可能
import distortion_correction as dc
result = dc.correct_trapezoid(image, vertical=50)
```

## 主要機能

### 1. 台形補正（3D回転）

カメラを傾けて撮影した画像の補正

```python
from cores.distortion_correction import correct_trapezoid
import numpy as np

# 画像はfloat32、RGB、0.0-1.0の範囲
image = load_image_as_float32("photo.jpg")  # shape: (H, W, 3)

# 建物を下から見上げた場合
corrected = correct_trapezoid(
    image,
    vertical=50,      # -100〜+100
    horizontal=0,     # -100〜+100
    rotation=0,       # -45〜+45度
    offset_x=0,       # -1.0〜1.0
    offset_y=0,       # -1.0〜1.0
    interpolation='bicubic'
)
```

**パラメータ**:
- `vertical > 0`: カメラが上向き（建物を見上げた）
- `horizontal > 0`: カメラが右向き
- `offset_x/y`: 回転中心のオフセット（デフォルト=画像中心）

### 2. 4点補正

任意の4点から矩形への変換

```python
from cores.distortion_correction import correct_four_points

# TCG座標系で4点を指定（左上、右上、右下、左下）
points_tcg = [
    (-0.4, 0.4),   # 左上
    (0.4, 0.3),    # 右上
    (0.4, -0.3),   # 右下
    (-0.4, -0.4)   # 左下
]

corrected = correct_four_points(image, points_tcg)
```

**TCG座標系**:
- 中心が(0, 0)
- X軸: -0.5（左）〜 +0.5（右）
- Y軸: -0.5（下）〜 +0.5（上）

### 3. レンズ歪み補正

```python
from cores.distortion_correction import correct_lens_distortion

# 樽型歪み補正
corrected = correct_lens_distortion(
    image,
    strength=-30,      # -100（樽型）〜 +100（糸巻型）
    scale=1.2,         # 1.0〜1.5（拡大して黒枠を削除）
    interpolation='bicubic',
    grid_size=1        # 0（補正なし）〜3（最高品質、遅い）
)
```

### 4. メッシュワープ

```python
from cores.distortion_correction import warp_mesh

# 10x10のメッシュを作成
mesh_points_tcg = []  # 100個の(x, y)タプル

corrected = warp_mesh(
    image,
    mesh_points_tcg,
    mesh_rows=10,
    mesh_cols=10
)
```

## Widgetの使用

KivyMDアプリケーションでの使用例：

```python
from cores.distortion_correction.widgets import (
    TrapezoidCorrectionWidget,
    FourPointCorrectionWidget,
    LensDistortionWidget
)

# 台形補正Widget
widget = TrapezoidCorrectionWidget()
widget.source_image = image  # numpy array (H, W, 3), float32
widget.vertical = 50
widget.offset_y = 0.1

# 補正結果を取得
corrected = widget.get_corrected_image()
```

## 画像フォーマット

**入力**:
- `numpy.ndarray`
- `dtype=np.float32`
- `shape=(H, W, 3)` (RGB)
- 値の範囲: `0.0 - 1.0`

**変換例**:
```python
import cv2
import numpy as np

# 読み込み（uint8 → float32）
img_bgr = cv2.imread("photo.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_float = img_rgb.astype(np.float32) / 255.0

# 補正
corrected = correct_trapezoid(img_float, vertical=50)

# 保存（float32 → uint8）
output_bgr = cv2.cvtColor((corrected * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
cv2.imwrite("output.jpg", output_bgr)
```

## 注意事項

1. **座標系**: TCG座標系（中心原点、Y軸上向き）を使用
2. **回転中心**: デフォルトは画像中心、offset_x/yで調整可能
3. **補間方法**: 'bilinear'（速い）または'bicubic'（高品質）
4. **境界**: 黒（0, 0, 0）で塗りつぶし

## トラブルシューティング

### TypeError: image.dtype must be float32
→ 画像を`float32`に変換してください

### 画像が画面外に行く
→ `offset_x`/`offset_y`を調整してください

### 補正が弱い/強すぎる
→ パラメータの範囲を確認してください（vertical/horizontal: -100〜+100）
