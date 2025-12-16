# Linear RGB to Log Converter with LUT Application

リニアRGB（float32）から各社の最新Logフォーマットに変換し、LUTを適用するPythonツール

## サポートしているLogフォーマット

- **Sony S-Log3** (`slog3`) - Sony Venice 2, FX9, FX6等で使用
- **ARRI LogC4** (`logc4`) - ARRI Alexa 35等で使用（最新規格）
- **Canon Log3** (`clog3`) - Canon C500 Mark II, C70等で使用
- **RED Log3G10** (`redlog3g10`) - RED V-Raptor, Komodo等で使用
- **Panasonic V-Log** (`vlog`) - Panasonic S1H, GH6, LUMIX BS1H等で使用
- **Nikon N-Log** (`nlog`) - Nikon Z9, Z8等で使用
- **Fujifilm F-Log2** (`flog2`) - Fujifilm X-H2S, X-H2等で使用（F-Logより広いダイナミックレンジ）
- **OM SYSTEM OM-Log400** (`omlog400`) - OM-1 Mark II, OM-1等で使用（ISO 400ベース）

## インストール

```bash
pip install numpy imageio
```

オプション（EXRファイルを扱う場合）:
```bash
pip install imageio[pyav]
# または
pip install OpenEXR
```

## 使い方

### 基本的な使い方

```python
import numpy as np
from linear_to_log_lut import process_image

# リニアRGB画像を読み込む（float32、範囲は0以上）
linear_rgb = ...  # shape: (H, W, 3)

# S-Log3に変換
slog3_image = process_image(linear_rgb, 'slog3')

# LogC4に変換してLUTを適用
result = process_image(linear_rgb, 'logc4', lut_path='my_lut.cube')
```

### コマンドライン使用例

```bash
# EXRからS-Log3に変換
python example_usage.py input.exr output.tiff --log slog3

# TIFFからLogC4に変換してLUT適用
python example_usage.py input.tiff output.tiff --log logc4 --lut rec709.cube

# 8-bit PNGとして出力
python example_usage.py input.exr output.png --log clog3 --bit-depth 8
```

## ワークフロー

正しい処理順序：

```
RAWファイル（センサーデータ）
    ↓
リニアRGB（デベイヤー、float32）
    ↓
Log変換（S-Log3/LogC4/等）← このツールが行う
    ↓
LUT適用（カラーグレーディング）← このツールが行う
    ↓
最終出力（Rec.709/Display等）
```

## 重要な注意点

### 入力データについて

- **入力はリニア（線形）RGB**である必要があります
- **float32型**を推奨（範囲：0以上、HDR対応）
- すでにLog変換済みのデータには使用しないでください
- ガンマ補正済み（sRGB等）の画像には使用しないでください

### LUTについて

- **Log用のLUT**を使用してください
- LUTの入力がどのLog形式を想定しているか確認してください
  - 例：S-Log3用LUTならば`--log slog3`を指定
- `.cube`形式のLUTに対応しています

### HDRデータ

各Logフォーマットは1.0以上の値（HDR）も扱えます：
- 適切なエクスポージャーで撮影されたRAWデータ
- 18%グレーが0.18付近になるよう正規化されたデータ

## API リファレンス

### LogConverter クラス

```python
converter = LogConverter()

# 各フォーマットへの変換
slog3 = converter.convert(linear_rgb, 'slog3')
logc4 = converter.convert(linear_rgb, 'logc4')
clog3 = converter.convert(linear_rgb, 'clog3')
redlog3g10 = converter.convert(linear_rgb, 'redlog3g10')
vlog = converter.convert(linear_rgb, 'vlog')
nlog = converter.convert(linear_rgb, 'nlog')
flog2 = converter.convert(linear_rgb, 'flog2')
omlog400 = converter.convert(linear_rgb, 'omlog400')
```

### LUTApplicator クラス

```python
applicator = LUTApplicator()

# LUT読み込み
lut_data, lut_size = applicator.read_cube_lut('path/to/lut.cube')

# LUT適用（trilinear補間）
result = applicator.apply_lut_trilinear(log_image, lut_data, lut_size)
```

### process_image 関数

```python
result = process_image(
    linear_rgb,           # リニアRGB画像（float32）
    log_format='slog3',   # Log形式
    lut_path='lut.cube'   # LUTパス（オプション）
)
```

## Log変換の仕様

各フォーマットの変換式は各社の公式仕様に基づいています：

- **S-Log3**: Sony Technical Paper（ITU-R BT.2100準拠）
- **LogC4**: ARRI LogC4 Specification
- **Canon Log3**: Canon Log3 White Paper
- **RED Log3G10**: RED Digital Cinema specifications
- **V-Log**: Panasonic V-Log/V-Gamut Technical Summary
- **N-Log**: Nikon N-Log specifications for Z9/Z8
- **F-Log2**: Fujifilm F-Log2 Technical Guide（F-Logの後継、より広いダイナミックレンジ）
- **OM-Log400**: OM SYSTEM OM-Log400 specifications（ISO 400最適化）

## ファイル構成

- `linear_to_log_lut.py` - メインライブラリ
- `example_usage.py` - コマンドラインツール
- `test_converter.py` - テストスクリプト
- `README.md` - このファイル

## テスト

```bash
python test_converter.py
```

すべてのLog変換が正しく動作することを確認できます。

## ライセンス

MIT License

## 参考資料

- [Sony S-Log3/S-Gamut3 Technical Summary](https://pro.sony/technology/s-log3-s-gamut3)
- [ARRI LogC4 Specifications](https://www.arri.com/en/learn-help/learn-help-camera-system/image-science/logc4)
- [Canon Log3 White Paper](https://www.usa.canon.com/internet/portal/us/home/learn/education/topics/article/2020/november/understanding-canon-log-3)
- [RED Digital Cinema - Log3G10](https://www.red.com/red-101/log3g10-color-science)
- [Panasonic V-Log](https://pro-av.panasonic.net/en/products/v-log_v-gamut.html)
