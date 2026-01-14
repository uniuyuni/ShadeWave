# 画像歪み補正API

KivyMD向けの画像歪み補正ライブラリです。レンズ歪み、台形歪み、自由変形など、様々な歪み補正機能を提供します。

## プロジェクト構造

```
プロジェクトルート/
├── cores/
│   └── distortion_correction/
│       ├── __init__.py
│       ├── coordinate_utils.py
│       ├── lens_distortion.py
│       ├── trapezoid_correction.py
│       ├── warp_correction.py
│       └── distortion_correction.py
└── widgets/
    └── distortion_correction/
        ├── __init__.py
        ├── lens_distortion_widget.py
        ├── trapezoid_correction_widget.py
        ├── four_point_correction_widget.py
        ├── mesh_warp_widget.py
        ├── line_guide_correction_widget.py
        └── point_warp_widget.py
```

**配置方法**: 
- コアモジュール（.py）は `cores/distortion_correction/` に配置
- Widgetファイルは `widgets/distortion_correction/` に配置

## 特徴

- **TCG座標系**: すべてのパラメータは画像中心からのオフセット（TCG座標系）で管理
- **値域保持**: 補間結果を自動クリップせず、元の値域を保持
- **高品質補間**: バイリニア（高速）とバイキュービック（高品質）の選択可能
- **KivyMD統合**: GUI Widgetを提供し、直感的な操作が可能

## インストール

必要なパッケージ:
```bash
pip install numpy opencv-python scipy kivy kivymd --break-system-packages
```

## 座標系について

このライブラリでは3つの座標系を扱います:

### TCG座標系（Transform Center Geometry）
- **原点**: 画像中心(0, 0)
- **X軸**: 右方向が正
- **Y軸**: 下方向が正
- **正規化**: 画像サイズで正規化（幅と高さで独立に正規化）
- **用途**: パラメータ保存、API入出力
- **例**（640x480の画像の場合）:
  - 中心: (0, 0)
  - 右上隅: (0.5, -0.5)
  - 右下隅: (0.5, 0.5)
  - 左上隅: (-0.5, -0.5)
  - 左下隅: (-0.5, 0.5)

### 画像座標系
- **原点**: 画像左上(0, 0)
- **X軸**: 右方向が正
- **Y軸**: 下方向が正
- **単位**: ピクセル
- **用途**: 内部処理

### Kivy座標系
- **原点**: Widget左下(0, 0)
- **X軸**: 右方向が正
- **Y軸**: 上方向が正
- **単位**: ピクセル
- **用途**: GUI表示・操作

## API リファレンス

### 1. レンズ歪み補正

```python
from cores.distortion_correction import correct_lens_distortion, detect_lens_distortion

# 手動で強度を指定
corrected = correct_lens_distortion(
    image,              # numpy.ndarray, float32, shape=(H, W, 3)
    strength=-30.0,     # -100.0〜+100.0（負=樽型補正、正=糸巻き型補正）
    interpolation='bicubic',  # 'bilinear' | 'bicubic'
    grid_size=1         # 1=最高品質、2-4=高速（プレビュー用）
)

# 自動検出を使用
detected_strength = detect_lens_distortion(image)
print(f"検出された歪み強度: {detected_strength}")

corrected_auto = correct_lens_distortion(
    image,
    strength=detected_strength,
    interpolation='bicubic'
)
```

### 2. 台形補正

#### 水平・垂直台形補正
```python
from cores.distortion_correction import correct_trapezoid

corrected = correct_trapezoid(
    image,
    horizontal=-20.0,   # -100.0〜+100.0
    vertical=15.0,      # -100.0〜+100.0
    rotation=2.5,       # -45.0〜+45.0（度）
    interpolation='bicubic'
)
```

#### 4点自由補正
```python
from cores.distortion_correction import correct_four_points

# TCG座標系で4点を指定（正規化座標）
# 例: 640x480の画像で、少しずれた矩形
src_points_tcg = [
    (-0.45, -0.45),   # 左上（中心から左に45%、上に45%）
    (0.42, -0.44),    # 右上
    (0.43, 0.46),     # 右下
    (-0.46, 0.45)     # 左下
]

corrected = correct_four_points(
    image,
    src_points_tcg,
    interpolation='bicubic'
)
```

### 3. メッシュワープ

```python
from cores.distortion_correction import warp_mesh

mesh_size = (4, 4)  # (rows, cols)

# TCG座標系でオフセットを指定（正規化座標）
control_offsets = {
    (1, 1): (0.02, 0.01),    # (row, col): (offset_x, offset_y)
    (2, 2): (-0.015, -0.02)
}

corrected = warp_mesh(
    image,
    mesh_size,
    control_offsets,
    interpolation='bicubic'
)
```

### 4. ラインガイド補正

```python
from cores.distortion_correction import correct_with_lines

# 歪んだ線（TCG座標系、正規化）
target_lines = [
    ((-0.2, 0.1), (0.25, 0.12)),   # 少し傾いた線
    ((-0.15, -0.15), (0.3, -0.14)) # もう1本
]

# 理想の水平線（TCG座標系、正規化）
reference_lines = [
    ((-0.3, 0.11), (0.3, 0.11)),   # 完全に水平
    ((-0.3, -0.145), (0.3, -0.145))
]

corrected = correct_with_lines(
    image,
    reference_lines,
    target_lines,
    influence=85.0,  # 0.0〜100.0
    interpolation='bicubic'
)
```

### 5. ポイントワープ

```python
from cores.distortion_correction import warp_points

# 元の位置（TCG座標系、正規化）
src_points = [
    (-0.1, 0.06),
    (0.15, 0.05),
    (-0.08, -0.15)
]

# 移動先（TCG座標系、正規化）
dst_points = [
    (-0.1, 0.06),    # そのまま
    (0.15, 0.08),    # Y座標を下に移動
    (-0.08, -0.15)   # そのまま
]

corrected = warp_points(
    image,
    src_points,
    dst_points,
    influence=100.0,
    interpolation='bicubic'
)
```

## GUI Widgets

KivyMD用のGUIウィジェットを提供しています。

### 使用例

```python
from widgets import LensDistortionWidget
from kivy.app import App

class MyApp(App):
    def build(self):
        widget = LensDistortionWidget()
        
        # 画像をセット
        import numpy as np
        image = np.random.rand(480, 640, 3).astype(np.float32)
        widget.set_image(image)
        
        return widget

MyApp().run()
```

### 利用可能なWidget

- `LensDistortionWidget`: レンズ歪み補正
- `TrapezoidCorrectionWidget`: 台形補正（水平・垂直）
- `FourPointCorrectionWidget`: 4点自由補正
- `MeshWarpWidget`: メッシュワープ
- `LineGuideCorrectionWidget`: ラインガイド補正
- `PointWarpWidget`: ポイントワープ

### Widgetの共通メソッド

```python
# 画像をセット
widget.set_image(image)

# パラメータを取得（TCG座標系）
params = widget.get_correction_params()

# パラメータを設定（TCG座標系）
widget.set_correction_params(params)

# 補正後の画像を取得（フル解像度）
corrected = widget.get_corrected_image()
```

## パラメータの保存と読み込み

すべてのパラメータはTCG座標系（正規化済み）で保存されます。

```python
import json

# パラメータを保存
params = {
    "type": "four_points",
    "src_points": [
        (-0.45, -0.45),
        (0.42, -0.44),
        (0.43, 0.46),
        (-0.46, 0.45)
    ]
}

with open("params.json", "w") as f:
    json.dump(params, f)

# パラメータを読み込んで適用
with open("params.json", "r") as f:
    params = json.load(f)

corrected = correct_four_points(
    image,
    params["src_points"],
    interpolation='bicubic'
)
```

## 画像の読み込みと保存

APIはfloat32、RGB、値域制限なしの画像を扱います。

```python
import numpy as np
import cv2

def load_image_as_float32(filepath):
    """画像をfloat32, RGB, [0,1]で読み込む"""
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    return image

def save_image_from_float32(image, filepath):
    """float32画像を保存"""
    # 注意: APIは値域をクリップしないので、保存時にクリップする
    image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, image_bgr)
```

## 注意事項

### 値域について
- APIは補間結果を自動的にクリップしません
- 画像の値域は入力時と同じ範囲を保持します
- 保存時に必要に応じてクリップしてください

### 補間方法
- `'bilinear'`: 高速だが品質は中程度
- `'bicubic'`: 低速だが高品質

### パフォーマンス
- プレビュー: 低解像度（1/4サイズ）+ bilinear補間
- 最終出力: フル解像度 + 指定した補間方法

## エラーハンドリング

```python
try:
    corrected = correct_lens_distortion(image, 150.0)  # 範囲外
except ValueError as e:
    print(f"パラメータエラー: {e}")

try:
    corrected = correct_lens_distortion(wrong_image, 0)  # 形式不正
except TypeError as e:
    print(f"画像形式エラー: {e}")
```

## ファイル構成

```
/home/claude/
├── coordinate_utils.py              # 座標変換ユーティリティ
├── lens_distortion.py               # レンズ歪み補正API
├── trapezoid_correction.py          # 台形補正API
├── warp_correction.py               # ワープ補正API
├── distortion_correction.py         # メインAPIモジュール
├── examples.py                      # 使用例
├── widgets/
│   ├── __init__.py
│   ├── lens_distortion_widget.py
│   ├── trapezoid_correction_widget.py
│   ├── four_point_correction_widget.py
│   ├── mesh_warp_widget.py
│   ├── line_guide_correction_widget.py
│   └── point_warp_widget.py
└── README.md
```

## ライセンス

MITライセンス

## 作者

Claude (Anthropic)

## サポート

問題が発生した場合は、以下を確認してください:

1. 画像形式: `dtype=float32`, `shape=(H, W, 3)`
2. パラメータ範囲: 各関数のドキュメントを参照
3. 座標系: すべてのパラメータはTCG座標系で指定

## GUIウィジェットの使用方法

すべてのウィジェットは別スレッドで実行されるロジックと統合できるよう、`@mainthread`デコレータを使用しています。

### 基本的な使い方

```python
from widgets.distortion_correction.lens_distortion_widget import LensDistortionWidget
from kivy.app import App

class MyApp(App):
    def build(self):
        widget = LensDistortionWidget()
        
        # 画像を設定
        import numpy as np
        image = np.random.rand(480, 640, 3).astype(np.float32)
        widget.set_image(image)
        
        # パラメータを設定
        widget.strength = -30.0
        
        # 補正後の画像を取得
        corrected = widget.get_corrected_image()
        
        return widget

if __name__ == '__main__':
    MyApp().run()
```

### 重要な注意事項

1. **マルチスレッド対応**: すべての`update_preview()`メソッドは`@mainthread`デコレータ付きなので、別スレッドから安全に呼び出せます

2. **画像形式**: 
   - 入力: `numpy.ndarray`, `dtype=float32`, `shape=(H, W, 3)`, 値域 `[0, 1]`
   - 内部表示: RGB, float32で直接表示（変換不要）

3. **レイアウト**: ウィジェットは`FloatLayout`ベースで、画像は中央配置・アスペクト比維持で表示されます

## バージョン履歴

- v1.1.0 (2025-01-11): マルチスレッド対応・プロジェクト構造改善
  - `@mainthread`デコレータ追加
  - プロジェクト構造を`cores/`と`widgets/`に分離
  - 画像表示をRGB float32で直接処理
  - レイアウト問題を修正
- v1.0.0 (2025-01-11): 初版リリース
  - レンズ歪み補正
  - 台形補正（水平・垂直、4点自由）
  - メッシュワープ
  - ラインガイド補正
  - ポイントワープ
  - KivyMD GUIウィジェット
