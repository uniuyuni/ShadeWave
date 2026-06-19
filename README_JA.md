# Shade Wave

[English](README.md)

![Shade Wave](docs/screenshot%201.png)

Shade Wave は、Python と Kivy で作られた macOS 向けの RAW/RGB フォトエディタです。
リニアライトの画像処理パイプライン、マスクによる部分編集、カラーマネジメントを意識したエクスポート、SAM3 / Depth Pro / SCUNet / インペイント系バックエンドなどの任意AI機能を組み合わせています。

現在は Apple Silicon macOS 上でのローカルデスクトップ利用を主な対象にしています。小さな pip ライブラリとして使うためのプロジェクトではなく、AI/モデル系の機能には大きな外部ダウンロードが必要です。

## クイックスタート

### 必要なもの

| 項目 | 内容 |
| --- | --- |
| OS | macOS、主に Apple Silicon で検証 |
| Python | pixi 経由の Python 3.11 |
| パッケージ管理 | [pixi](https://pixi.sh/) |
| コンパイラ | Xcode Command Line Tools / Apple Clang |
| メタデータツール | ExifTool CLI。例: `brew install exiftool` |
| ネットワーク | セットアップ時に外部リポジトリ、ICCプロファイル、モデル重みの取得が必要 |
| 任意トークン | `facebook/sam3.1` のチェックポイント取得に Hugging Face トークンが必要な場合あり |
| 任意APIキー | Runwareを使うインペイント/オブジェクト消去機能には `RUNWARE_API_KEY` が必要 |

先に pixi をインストールしてから、以下を実行してください。

```bash
git clone https://github.com/uniuyuni/platypus.git
cd platypus
./setup.sh
pixi run python main.py
```

`setup.sh` は pixi 環境の作成、`external/` への外部プロジェクトのclone、ネイティブ拡張のビルド、ICCプロファイルの取得、必要なモデルアセットの取得を行います。

メタデータ、レーティング、エクスポート時のメタデータコピーを使うには、ExifTool CLI が `PATH` から見える必要があります。

ユーザーが編集する設定やプリセットは以下に保存されます。

```text
~/Pictures/Shade Wave
```

## スコープ

Shade Wave でできること:

- 一般的なRGB画像、カメラRAW、OpenEXR画像を開く。
- レイヤー状の非破壊パラメータパイプラインで画像を編集する。
- グローバル補正、ローカルマスク、ジオメトリ補正、フィルム/ルック系効果、ノイズ除去、デヘイズ、シャープ、グレイン、ビネットを適用する。
- ICCプロファイルとメタデータを扱いながら一般的な形式へエクスポートする。
- 必要な外部モデルがある場合にAI支援機能を使う。

現時点で目指していないこと:

- クロスプラットフォームな配布アプリ。
- 軽量なPythonライブラリAPI。
- 安定した公開拡張ポイントを持つ完全なエンドユーザー製品。
- 署名/公証済みのmacOS配布物。

## 使い方

### GUIを起動する

```bash
pixi run python main.py
```

ファイルビューアから画像を開き、編集パネルで効果を調整し、エクスポートダイアログから書き出します。編集内容は、アプリのパラメータ保存フローに従ってソース画像の近く、またはユーザーデータ領域に保存されます。

### macOS `.app` をビルドする

```bash
pixi run build-macos-app
```

生成先:

```text
dist/Shade Wave.app
```

一時的なビルド先や出力先を指定する場合:

```bash
pixi run python scripts/build_macos_app_pyinstaller.py \
  --distpath /tmp/platypus-app-dist \
  --workpath /tmp/platypus-app-build
```

`.app` は現在の環境と大きなランタイムアセットを同梱するため、数GB規模になることがあります。コード署名と公証はこのスクリプトでは行いません。

### テストを実行する

```bash
pixi run python -m unittest discover -s tests -p "test_*.py"
```

一部のテストや実行経路は、`setup.sh` で準備されるネイティブライブラリ、外部リポジトリ、モデルアセットに依存します。

### キーボードショートカット

| ショートカット | 動作 |
| --- | --- |
| `0` | マウス位置を中心にプレビューズームを切り替える。マウスが画像外の場合はプレビュー中央を使う |
| `Space` 長押し | 一時的なプレビュードラッグ/高速表示モード。離すと通常プレビューを再描画 |
| `Cmd/Ctrl + S` | 現在のサイドカーパラメータを保存 |
| `Cmd/Ctrl + C` | 現在のEffect設定をコピー |
| `Cmd/Ctrl + V` | コピーしたEffect設定をペースト |
| `Cmd/Ctrl + F` | プレビューフォーカスモードを切り替え |
| `Cmd/Ctrl + Z` | Undo |
| `Cmd/Ctrl + Shift + Z` | Redo |
| `Delete` / `Backspace` | 歪曲補正のラインガイドエディタで選択中の線を削除 |

テキスト入力にフォーカスがある場合、`0` は無視されます。一部のショートカットは、画像が読み込まれていて対応する編集状態が有効なときだけ意味を持ちます。

## 入出力

### 対応入力

| 種類 | 拡張子 |
| --- | --- |
| RGB / 一般画像 | `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, `.gif`, `.heic`, `.jxl` |
| RAW | `.cr2`, `.cr3`, `.nef`, `.arw`, `.dng`, `.orf`, `.raf`, `.rw2`, `.sr2`, `.pef`, `.raw`, `.3fr`, `.fff` |
| HDR / シーンリニア | `.exr` |

RAW 読み込みには外部バックエンド `libraw_enhanced` を使います。OpenEXR は OpenEXR Python bindings を使い、通常のRGB読み込みとは別経路で扱います。

### 対応エクスポート

| 形式 | 備考 |
| --- | --- |
| JPEG | 品質設定、メタデータ/レーティング対応 |
| TIFF | Deflate圧縮 |
| PNG | エクスポート経路上で品質オプションを扱う |
| JPEG XL | インストール済み画像スタックのJXL対応が必要 |
| HEIF | インストール済み画像スタックのHEIF対応が必要 |
| OpenEXR | シーンリニア書き出し、利用可能な場合はchromaticitiesを書き込み |

エクスポートでは、リサイズ、シャープ、ディザ/量子化、非EXR形式へのICCプロファイル埋め込み、選択されたEXIF/GPSメタデータのコピーができます。

利用可能なICCプロファイルは `icc/` から読み込まれます。セットアップが正常に完了していれば、sRGB、Display P3、Adobe RGB、ProPhoto RGB、ACES、Rec.709、Rec.2020 などを利用できます。

## 機能

- RAW現像: RAWデコード、自動露出、色温度/ティント、レンズ補正連携、色収差補正。
- トーンとカラー: 露出、コントラスト、トーン、レベル、カーブ、HLS、Hue/Saturation/Luminance系カーブ、色分離、入力/ルックLUT。
- ディテールと復元: AIノイズ除去、SCUNet、radiance denoise、デヘイズ、クラリティ、テクスチャ、マイクロコントラスト、アンシャープマスク。
- ジオメトリ: 回転、クロップ、パース/メッシュ系補正、歪曲補正、サブピクセルシフト、レンズブラー、モザイク、スクラッチ/フロストガラス風効果。
- ローカル編集: マスクレイヤー、合成マスク、フリードローマスク、Quick Select系のエッジリファイン、マスクジオメトリ、エクスポート時のヘッドレスマスク再生。
- AI支援マスク: SAM3の矩形/テキストセグメンテーション、Depth Pro深度マスク、顔マスク。
- クリエイティブルック: フィルムシミュレーション、レンズシミュレーション、クロスフィルター、グロー、オートン、グレイン、ビネット、単色オーバーレイ。
- メタデータとレーティング: エクスポート時のメタデータ設定とレーティング反映。
- パッケージ化: 現在の pixi 環境を使った PyInstaller ベースの macOS `.app` 作成。

## アーキテクチャ

コードベースは、プレビュー/エクスポートパイプラインと Effect オブジェクトを中心に構成されています。

```text
main.py                         KivyアプリのエントリポイントとUI制御
effects.py                      Effect定義とパラメータ/Widgetバインディング
pipeline.py                     プレビュー/エクスポートパイプライン
export.py                       ファイル書き出し、色変換、ICC/メタデータ処理
cores/                          画像処理カーネルと再利用アルゴリズム
cores/mask2/                    マスク生成、ヘッドレスマスク、AIマスク実行部
helpers/                        外部AI/ネイティブバックエンド連携
effect_backends/                任意/ネイティブバックエンドのアダプタ
widgets/                        Kivy Widget とエディタUI
external/                       setup.sh が準備する外部プロジェクト
scripts/                        ビルド、パッケージ、環境補助スクリプト
tests/                          パイプライン、UIフロー、ヘルパーの回帰テスト
```

大まかな流れ:

1. `main.py` が画像を `ImageSet` に読み込む。
2. Effectパラメータは `primary_param` に保持され、`effects.py` がWidgetと同期する。
3. `pipeline.py` がEffectレベル順に処理し、プレビューキャッシュとマスク合成を行う。
4. Mask2ツールはUI上でマスクを生成/編集し、エクスポート時はヘッドレスマスクパイプラインで再生する。
5. `export.py` がフルパイプラインを再実行し、選択された出力プロファイルへ色変換してファイルとメタデータを書き出す。

## 外部コンポーネント

`setup.sh` は複数の大きな外部プロジェクトを `external/` 以下にcloneまたは準備します。

| コンポーネント | 用途 |
| --- | --- |
| `libraw_enhanced` | RAWデコードとMetalシェーダーリソース |
| `SAM3` | AIセグメンテーションマスク |
| `ml-depth-pro` | 深度マスク用の深度推定 |
| `SCUNet` | PyTorchノイズ除去モデル重み |
| `SCUNet_CoreML` | Core ML版SCUNetモデルとランタイム |
| `radiance_denoise` | ネイティブノイズ除去バックエンド |
| `demosaicnet_torch` | AIデモザイク補助 |

これらの多くは独自のライセンスやモデル利用条件を持ちます。再配布する場合は各上流リポジトリの条件を確認してください。

## 設定

重要なデフォルト設定は `config.py` にあり、実行時に `~/Pictures/Shade Wave/config.json` へコピー/移行されます。

| キー | デフォルト | 意味 |
| --- | --- | --- |
| `preview_size` | `640` | プレビューテクスチャの最小辺 |
| `raw_auto_exposure` | `true` | RAW自動露出をデフォルトで有効化 |
| `gpu_device` | `mps` | Apple Siliconで優先するAIデバイス |
| `display_color_gamut` | `sRGB` | 表示変換のターゲット |
| `cat` | `cat16` | 色順応変換 |
| `base_resolution_scale` | `[4096, 4096]` | 基本処理解像度の上限 |
| `mesh_rbf_function` | `mls` | メッシュワープ補間方式 |

## 既知の制約

- セットアップは大きめです。SAM3 と Depth Pro のチェックポイントだけで数GBになります。
- SAM3チェックポイントの取得には Hugging Face トークンとモデルアクセス承認が必要な場合があります。
- Runwareを使うインペイント/オブジェクト消去機能には `RUNWARE_API_KEY` が必要です。未設定の場合、その経路は使えず、現状では派手に失敗する可能性があります。
- PyTorch MPS は未対応演算でCPUフォールバックすることがあり、一部AI処理が遅くなる場合があります。
- Core ML tooling は最新PyTorchへの追従が遅れることがあり、このプロジェクトではsetup制約と個別installコマンドで互換性を保っています。
- macOSアプリビルドは未署名/未公証です。
- PyInstaller bundle は Torch/CoreML/ネイティブ依存とモデルリソースを含むため大きくなります。
- Windows と Linux は現時点で第一ターゲットではありません。

## 開発メモ

- `requirements.txt` は `setup.sh` が使う pip 依存の正本として扱います。
- ネイティブ依存やモデルが重い外部プロジェクトは `external/` に置きます。import path を追加する場合は `utils.external_paths` を使います。
- パイプライン、マスク再生、Effectパラメータバインディング、エクスポート挙動、UIフローの回帰テストを重視します。
- ダウンロード済みチェックポイントや生成されたビルド成果物はコミットしないでください。
- リファクタリング指針は `docs/refactoring-guidelines.md` にあります。

よく使うコマンド:

```bash
pixi install
./setup.sh
pixi run python main.py
pixi run python -m unittest discover -s tests -p "test_*.py"
pixi run build-macos-app
```

## バージョンと互換性

現在のアプリケーションバージョンは `define.py` で定義されています。

```text
Shade Wave 2.22.42
```

現在の pixi 環境は `osx-arm64` と Python `>=3.11.13,<3.12` を対象にしています。

## ライセンスとクレジット

このリポジトリには現在、トップレベルのライセンスファイルがありません。プロジェクトライセンスが追加されるまでは、再配布権限は未定義として扱ってください。

Shade Wave は Kivy、KivyMD、NumPy、OpenCV、pyvips、OpenEXR、Pillow、PyTorch、torchvision、coremltools、LibRaw、lensfun/lensfunpy、SAM3、Depth Pro、SCUNet、radiance_denoise など多くのオープンソースライブラリと外部研究プロジェクトに依存しています。バイナリやモデルを配布する前に、各上流のライセンスとモデル利用条件を確認してください。

## リンク

- リポジトリ: https://github.com/uniuyuni/platypus
- リファクタリングメモ: `docs/refactoring-guidelines.md`
- SAM3モデルアクセス: https://huggingface.co/facebook/sam3.1
