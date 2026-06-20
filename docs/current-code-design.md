# Shade Wave 現状コード詳細設計書

作成日: 2026-06-19  
対象バージョン: 現行 `main` 作業ツリーのコード  
目的: 保守、機能追加、リファクタリング、移植、オンボーディングのために、現状コードの構造・実行フロー・不変条件・落とし穴を残す。

## 0. ゴール、読者、スコープ

### ゴール

この文書は「現状のコードを読むための地図」である。関数の写経ではなく、次の判断に必要な設計情報を残す。

- どのファイルから読めば、画像ロード、プレビュー、マスク、エクスポートの流れを追えるか。
- どのデータ構造が共有され、どこで保存され、どこで破棄されるか。
- どの順序で画像処理が適用され、プレビューとエクスポートの差がどこで生まれるか。
- 非同期処理、キャッシュ、座標変換、重いAI結果の保存で壊しやすい不変条件は何か。

### 想定読者

- 将来の開発者本人。
- 共同開発者、外注先、QA担当。
- 既存機能を壊さずに機能追加・移植・リファクタリングをしたい人。

### スコープ

対象は「全体像 + 主要フロー + 主要モジュール」である。全クラス・全関数のAPIリファレンスではない。特に深掘りする領域は以下。

- 起動、設定、ウィンドウ、ワーカー初期化。
- ファイルロード、RAWプレビュー/フルデコード、画像キャッシュ。
- Effectパイプライン、プレビュー、エクスポート。
- Mask2、ヘッドレスマスク、AI推論キャッシュ。
- `.pmck` 永続化、重いpayloadの扱い、レーティング。
- 非同期処理、共有メモリ、メモリ圧迫時のキャッシュ破棄。

## 1. 5分で掴む全体像

### 代表ユースケース

```text
ユーザーが画像を選択
  -> main.MainWidget.on_select
  -> file_cache_system.FileCacheSystem がロードを開始
  -> imageset.ImageSet.preload が形式別ロードタスクを返す
  -> RAWなら preview と full decode の2段階、RGB/EXRなら1段階
  -> MainWidget が callback で imgset / exif / param を受け取る
  -> params.load_json が .pmck を復元
  -> effects.set2widget_all でUIへ反映
  -> MainWidget.start_draw_image_and_crop
  -> MainWidget.draw_image_core
  -> pipeline.process_pipeline
  -> lv0 -> crop/zoom -> lv1/lv2/lv3 -> Mask2 composite -> lv4
  -> 表示用色変換とヒストグラム更新
  -> ユーザーがExport
  -> export.ExportFile.write_to_file
  -> ImageSetをフルロード
  -> Mask2HeadlessPipelineで .pmck マスクを再生
  -> pipeline.export_pipeline
  -> ICC/EXR/メタデータ処理
  -> ファイル生成
```

### 最重要ファイル

| 優先 | ファイル | 読む理由 |
| --- | --- | --- |
| 1 | `main.py` | アプリの司令塔。UI状態、ロードcallback、描画キック、履歴、ショートカット、エクスポート起動を持つ。 |
| 2 | `pipeline.py` | プレビュー/エクスポートの画像処理順序、Effectキャッシュ、Mask2合成、非同期結果の扱いを持つ。 |
| 3 | `effects.py` | Effectの基底クラス、param/widget binding、全Effectの登録順、重い処理の入口を持つ。 |

### 最重要ディレクトリ

| ディレクトリ | 責務 |
| --- | --- |
| `cores/` | Kivy非依存の画像処理コア、色変換、クロップ/変形、マスクアルゴリズム。 |
| `widgets/` | Kivy UI部品。`mask_editor2.py` はMask2 UI実装の中心。 |
| `helpers/` | 外部AI/ネイティブバックエンド接続。SAM3、Depth Pro、SCUNet、Runwareなど。 |

## 2. 静的アーキテクチャ

### ディレクトリ/モジュール責務

| パス | 役割 |
| --- | --- |
| `main.py` | Kivyアプリ本体。`MainWidget` と `MainApp`、ロード/描画/履歴/エクスポート/UI同期。 |
| `main.kv` | メインUIレイアウト。各パネル、ID、Widget接続。 |
| `imageset.py` | ファイル形式別ロード。RAW/RGB/EXRを内部作業色空間のfloat32画像へ変換。 |
| `file_cache_system.py` | ロードキュー、ThreadPool/ProcessPool、先読み、メモリキャッシュ、最終表示キャッシュ。 |
| `pipeline.py` | Effectレベル適用、preview/export差、Mask2合成、AsyncPipelineManager。 |
| `effects.py` | Effect基底、binding、各Effect、登録順。 |
| `export.py` | フルロード、ヘッドレスマスク復元、パイプライン再実行、ICC/EXR/Exif出力。 |
| `params.py` | `.pmck` serialize/deserialize、特殊param削除、座標変換、crop/disp_info管理。 |
| `config.py` | グローバル設定。`multiprocessing.Manager().dict()` を使いワーカーへ共有。 |
| `utils/paths.py` | ユーザーデータディレクトリと旧名からの移行。 |
| `async_worker.py` | 重いEffectを別プロセス/スレッドで処理するワーカー。 |
| `processing_dialog.py` | ブロッキング処理中のmacOS HUD/スタブ。 |
| `memory_manager.py` | RSS/available memory監視、Effect/processorキャッシュ破棄。 |
| `cores/mask2/` | Kivy非依存のMask2ヘッドレス処理、AI推論ランタイム、座標コンテキスト。 |
| `widgets/mask_editor2.py` | Mask2 UI。各マスク型、serialize/deserialize、画面上の操作。 |
| `effect_backends/` | 参照実装/高速実装アダプタ。色変換、tone、cross filterなど。 |
| `external/` | `setup.sh` がcloneする外部プロジェクト。リポジトリ管理対象ではない前提。 |
| `tests/` | 回帰テスト。pipeline、mask、export、UI flow、backend contractを広くカバー。 |

### 主要コンポーネント

| コンポーネント | 所有者 | 主な状態 | 備考 |
| --- | --- | --- | --- |
| GUI | `MainWidget` | `imgset`, `primary_param`, `primary_effects`, `crop_image`, `history` | ほぼ全体の状態を集約するgod object。 |
| File cache | `FileCacheSystem` | `cache`, `preload_registry`, `active_processes`, `final_display_cache` | 画像ロード結果と表示用画像を保持。 |
| Image buffer | `ImageSet` | `file_path`, `img`, `fidelity`, `color_space` | `img` は基本float32、作業色空間はProPhoto RGBリニア。 |
| Effect graph | `effects.create_effects()` | 5段階のdictリスト | 順序はdict挿入順が仕様。 |
| Pipeline config | `EffectConfig` | `mode`, `disp_info`, `resolution_scale`, `upstream_hash`, `layer_status` | pipeline中に各Effectへ渡される実行コンテキスト。 |
| Mask2 UI | `MaskEditor2` | mask list, active mask, ref image, texture size | UI用。Kivy Widgetを持つ。 |
| Mask2 export | `Mask2HeadlessPipeline` | `ctx`, `mask_list` | export用。Kivyに依存しない。 |
| History | `history.History` | `Operation` list | Effect paramとMask2 layer操作のundo/redo。 |
| Async effect manager | `AsyncPipelineManager` | effect result cache | `AsyncWorker` と連携し、最新版タスクだけを採用。 |

### 依存関係の概要

```text
main.py
  -> config.py
  -> file_cache_system.py -> imageset.py -> libraw_enhanced / pyvips / OpenEXR
  -> effects.py -> cores/ / helpers/ / effect_backends/
  -> pipeline.py -> effects.py / cores.core / params.py / Mask2
  -> params.py -> msgpack / Mask2 serialize
  -> export.py -> pipeline.py / Mask2HeadlessPipeline / pyvips / exiftool / OpenEXR

widgets/mask_editor2.py
  -> cores.mask2.inference_runtime -> helpers.sam3_helper / depth_pro_helper / facer_helper
  -> params.py coordinate conversion

setup.sh
  -> pixi.toml / requirements.txt
  -> external/libraw_enhanced, SAM3, depth_pro, SCUNet, SCUNet_CoreML, radiance_denoise
```

重要な依存の向き:

- `pipeline.py` はUI Widgetを直接深く知りすぎないが、`mask_editor2` APIには依存する。
- `export.py` はUIのMaskEditor2ではなく `Mask2HeadlessPipeline` を使う。
- `effects.py` は画像処理コアとUI bindingの両方を持つため、変更の影響範囲が広い。
- `params.py` は永続化形式、座標変換、特殊キー管理を同時に担う。ここは破壊的変更に注意。

## 3. 実行フロー

### 3.1 起動から終了

#### 入力

- `pixi run python main.py`
- frozen appの場合はPyInstallerの `_MEIPASS` 配下へchdirする起動処理が走る。

#### 処理ステップ

1. `MainApp.build()` が `MainWidget` を生成する。
2. `config.init_config(self.main_widget)` がデフォルト設定を `multiprocessing.Manager().dict()` に入れる。
3. `config.load_config()` が `~/Pictures/Shade Wave/config.json` を読み、UIへ適用する。
4. `MainWidget.async_worker.start()` が非同期Effectワーカーを起動する。
5. `KVClock` が async結果poll、メモリ圧迫チェック、Window autosave設定をスケジュールする。
6. `MainWidget.__init__` で描画用スレッド `self.apply_thread` を起動する。

#### 出力

- Kivyウィンドウ。
- ユーザーデータディレクトリ `~/Pictures/Shade Wave`。
- 非同期ワーカー、描画スレッド、FileCacheSystem。

#### 終了

`MainApp.on_stop()` は以下を行う。

1. exportキャンセル要求。
2. export threadの短時間join。
3. export UI状態の終了処理。
4. 現在の `.pmck` 保存。
5. `MainWidget.shutdown()` によるワーカー等の後始末。

#### 失敗パターン

- 設定ファイルがなければ無視してデフォルトを使う。
- frozen appではspawn系子プロセスが再初期化で落ちやすいため、ロードpoolは環境変数/コードでforkへ寄せる箇所がある。
- Kivy初期化と外部MLライブラリimportは相性問題が多い。setup/patch側で抑制しているものがある。

### 3.2 ファイルロード

#### 入力

- Viewerで選択した画像パス。
- ExifTool由来のメタデータ。
- 既存 `.pmck`。

#### 処理ステップ

1. `MainWidget.on_select(card)` が選択パスを受ける。
2. 期待パス `_expected_file_path` を更新し、遅延callbackの取り違えを防ぐ。
3. `FileCacheSystem` が `_load_file_thread` をThreadPoolへ投入する。
4. `_load_file_thread` は `ImageSet.preload()` を呼び、形式別タスクを得る。
5. RAW:
   - `_load_raw_preview` を先に実行し、埋め込みプレビューまたは黒画像をfloat32へ変換する。
   - `_load_raw_full` を別タスクで実行し、LibRaw postprocess結果を共有メモリで返す。
6. RGB:
   - `_load_rgb` がpyvipsで読み込み、ICCを推定し、ProPhoto RGBリニアへ変換する。
7. EXR:
   - `_load_exr` がOpenEXR経由でシーンリニアを読み、chromaticitiesからProPhoto RGBリニアへ変換する。
8. `_task_callback` が結果を `FileCacheSystem.cache[file_path] = (imgset, exif_data, param.copy(), history)` に入れる。
9. MainWidgetのcallbackが `.pmck` を復元し、UIとpipelineへ渡す。

#### 出力

- `ImageSet.img`: 原則 `np.float32`, shape `(H, W, 3)`。
- `ImageSet.fidelity`: RAW previewは `PREVIEW`、RGB/EXR/RAW fullは `FULL`。
- `primary_param`: `original_img_size`, `crop_rect`, `disp_info`, `rgb_or_raw`, `auto_exposure`, `exif_data`, `image_fidelity` など。

#### 失敗パターン

- 未対応拡張子やpreload失敗は `_notify_load_failed()` がFULL_DECODE相当のcallbackを呼び、UIのloadingを解除する。
- RAW埋め込みプレビューがない場合、EXIFサイズに合わせた黒画像を作る。
- Fuji honeycomb等はLibRaw geometry probeでEXIFサイズを補正する。
- ロード結果が遅延して現在選択中のファイルと違う場合、MainWidget側で捨てる必要がある。

### 3.3 プレビュー描画

#### 入力

- `self.imgset.img`
- `self.primary_effects`
- `self.primary_param`
- `self.ids['mask_editor2']`
- preview widget size, zoom state, current tab

#### 処理ステップ

1. UI操作は `start_draw_image()` または `start_draw_image_and_crop()` を呼ぶ。
2. `pipeline_version` を進め、描画スレッドに `draw_event` を送る。
3. 描画スレッド `draw_image()` が `draw_image_core()` を呼ぶ。
4. `draw_image_core()` は `pipeline.process_pipeline()` へ渡す。
5. `process_pipeline()`:
   - `original_img_size` guard。
   - `ensure_initial_crop_rect()`。
   - `disp_info` と `EffectConfig` を構築。
   - `pipeline_lv0()` でロード待ち、AI NR、inpaint、geometry、crop等を処理。
   - crop/zoomを適用してpreviewサイズへ切る。
   - `MaskEditor2.set_ref_image()` でマスクの参照画像を更新。
   - drag中でなければ `pipeline2()` + `pipeline_last()`。
6. `draw_image_core()` が表示色域へ変換し、over/under exposure表示などを適用してblitする。
7. ヒストグラムとMask overlayを更新する。

#### 出力

- Kivy textureへの表示。
- `crop_image` キャッシュ。
- `final_display_cache` への表示完了画像保存。
- `history` とUI状態の同期。

#### 失敗パターン

- `original_img_size` が未定義なら処理しない。
- stale frameは `pipeline_version` と比較し、基本的に破棄する。
- `Space` drag preview中はlv1-lv4をスキップし、通常再描画時に再計算させるため `crop_image` を残さない。
- `debug_nan_inf_check` 有効時は各段でNaN/Infを検査する。

### 3.4 エクスポート

#### 入力

- 元画像パス。
- 現在のparam sidecar。
- export dialog設定: 出力パス、品質、resize、sharpen、ICC、Exif/GPS、dithering。

#### 処理ステップ

1. UIから `ExportFile.write_to_file()` が呼ばれる。
2. `ImageSet.preload()` + `ImageSet.load()` でフル画像を同期ロードする。
3. `params.apply_original_geometry_if_missing()` で最低限のgeometryを確定。
4. `Mask2HeadlessPipeline` を作り、`params.load_json(..., load_heavy=True)` で `.pmck` を読み込む。
5. `pipeline.export_pipeline()`:
   - `EffectMode.EXPORT`。
   - cropはフル解像で先に適用。
   - headless maskをupdate。
   - `pipeline2()` + `pipeline_last()`。
6. `_convert_export_color()`:
   - EXRはシーンリニア保持、ガマットマッピング/transfer/clipなし。
   - 非EXRは出力ICCへ変換、transfer付与、clip。
7. `_prepare_output_array()` でresize/sharpen/dither/quantize。
8. EXRならOpenEXRでwrite。非EXRならpyvipsでwrite。
9. ExifToolで安全なメタデータをコピーし、ratingを書き込む。

#### 出力

- `.JPG`, `.TIFF`, `.PNG`, `.JXL`, `.HEIF`, `.EXR`。
- 非EXRはICC profile blobを埋め込み。
- ratingはメタデータコピーON/OFFに依存せず可能な限り書く。

#### 失敗パターン

- `original_img_size` が確定できない場合は中止。
- cancel_eventが立つと各ステップの区切りでFalseを返す。
- OpenEXR import失敗は `ExportFormatError`。
- ICCファイルが無い場合はwarningして続行。

## 4. データモデルと永続化

### 4.1 ImageSet

| フィールド | 意味 |
| --- | --- |
| `file_path` | 入力画像パス。 |
| `img` | 作業画像。基本 `np.float32`、ProPhoto RGBリニア。 |
| `fidelity` | `ImageFidelity.PREVIEW` または `FULL`。 |
| `color_space` | 読み込み元/現在の色空間メモ。RAW previewはsRGB、通常作業はProPhoto RGB。 |

ライフサイクル:

1. `ImageSet()` を生成。
2. `preload()` がロードタスクを返す。
3. `_load_*` が `img`, `fidelity`, paramを設定。
4. FileCacheSystemがcacheへ入れる。
5. MainWidgetが現在画像として参照する。
6. メモリ圧迫やファイル選択変更でcacheから外れる。

不変条件:

- pipelineへ入る画像は原則float32。
- RGB/RAW full/EXRはProPhoto RGBリニアへ揃える。
- RAW previewは表示可能性優先の近似であり、FULL遷移後にheavy payloadを再マージする。

### 4.2 primary_param

`primary_param` はスキーマレスdictで、UI、pipeline、永続化、historyが共有する中心データである。

主なキー:

| キー | 用途 |
| --- | --- |
| `original_img_size` | 元画像サイズ。pipeline guardの最重要キー。 |
| `crop_rect` | 元画像座標のcrop矩形。 |
| `disp_info` | preview crop/scale情報。Mask座標変換にも使う。 |
| `rotation`, `rotation2`, `flip_mode`, `matrix` | geometry/TCG変換。 |
| `image_fidelity` | preview/full状態。重いpayload保存判断に使う。 |
| `exif_data` | 読み込み時メタデータ。 |
| effect固有キー | `effects.py` の `param_bindings` / `get_param_dict()` が定義。 |

注意:

- dict自体を構造化しすぎると `.pmck` 後方互換を壊しやすい。
- `params.py` の `SPECIAL_PARAM`, `HEAVY_PRIMARY_PARAM_KEYS`, lensfun系キーは保存/削除ルールと直結する。
- historyは値をcopyしてdiffを作る。巨大配列やmutable payloadは扱いに注意。

### 4.3 .pmck

`.pmck` は `msgpack` で保存されるsidecarである。

主な内容:

- `primary_param`
- `mask2`
- RAW専用ratingキー
- 重いpayloadを含む場合は `heavy_saved_at_fidelity`

保存:

- `params.save_json(file_path, param, mask_editor2, raw_sidecar_rating=0)`。
- 実質編集なしなら `.pmck` を削除する。
- RAWでratingだけある場合は最小 `.pmck` を保存する。

読み込み:

- `params.load_json(file_path, param, mask_editor2, load_heavy=True/False)`。
- 破損/読み込み失敗はloggingしてNone。
- 古いtuple/list差は必要箇所でtupleへ戻す。
- 旧 `rating` はparamから除去する。

重いpayload:

- `ai_noise_reduction_result`
- `inpaint_diff_list`
- `patchmatch_inpaint_diff_list`
- `color_match_source_image`
- `heavy_saved_at_fidelity`

設計意図:

- RAW preview段階で重いFULL結果を無条件に復元すると、画像サイズ/座標が合わず破綻する。
- `merge_heavy_from_pmck()` はPREVIEW→FULL遷移時にheavy payloadだけを再マージするための経路。

### 4.4 Mask2

Mask2はUI用とヘッドレス用に分かれる。

| 実装 | 用途 |
| --- | --- |
| `widgets/mask_editor2.py` | Kivy Widgetを持つ通常編集用。 |
| `cores/mask2/headless_pipeline.py` | export/別プロセス用。Kivy非依存。 |

マスク型:

- Composit
- CircularGradient
- Gradient
- Full
- FreeDraw
- Polyline
- Segment
- DepthMap
- Face
- TargetText

ライフサイクル:

1. UIでMaskを作成/編集。
2. `MaskEditor2.serialize()` が `.pmck` 用dictへ変換。
3. 通常プレビューでは `MaskEditor2.get_mask_list()` をpipelineが参照。
4. exportでは `Mask2HeadlessPipeline.deserialize()` がdictからheadless maskを再構築。
5. `pipeline2()` がComposit maskだけを合成対象として扱い、子マスクからmask imageを得る。

不変条件:

- exportで再現できるMaskはheadless実装が必要。
- UI座標とexport座標は `Mask2CoordinateContext` / `params` のTCG変換に依存する。
- mask composite後は `efconfig.upstream_hash` を更新し、lv4 cacheを誤再利用しない。

### 4.5 TCG座標系とMask2座標

TCG座標は、画像表示・Geometry・Crop・Mask2をまたいで使われる中心座標系である。ここを誤解すると、マスクがずれる、メッシュ制御点が画面外に飛ぶ、previewとexportで位置が合わない、という形で壊れる。

#### 座標系の種類

| 名前 | 単位 | 原点/範囲 | 主な用途 |
| --- | --- | --- | --- |
| TCG-normalized | 比率 | 画像中心を `(0, 0)` とする。概ね `[-0.5, +0.5]` 系 | `.pmck` 保存、UI操作点の永続化 |
| TCG-pixel | px | 画像中心を `(0, 0)` とする。`imax = max(original_img_size)/2` を基準に扱う箇所あり | rotation/matrix/mask geometryの演算 |
| full image px | px | 元画像左上を `(0, 0)` | segmentation/depth/faceなど元画像に対する推論 |
| crop/ref image px | px | 現在のcrop画像または参照画像左上を `(0, 0)` | HLS mask、edge refine、preview crop内処理 |
| texture px | px | preview texture左上を `(0, 0)`、Yは描画用にflipされる箇所あり | Mask raster生成、Kivy texture表示 |
| window px | px | Kivy window座標、DPI scale込み | touch/key/mouse位置 |

`params.norm_param()` と `params.denorm_param()` が TCG-normalized と TCG-pixel をつなぐ。`original_img_size` の幅/高さ、または `max(original_img_size)` を使う経路が混在するため、点なのかスカラー半径なのかを意識する必要がある。

#### tcg_info

`params.param_to_tcg_info(primary_param)` は座標変換に必要なスナップショットを作る。

```text
tcg_info = {
  original_img_size,
  disp_info,
  rotation,
  rotation2,
  flip_mode,
  matrix,
}
```

各キーの意味:

| キー | 意味 |
| --- | --- |
| `original_img_size` | 元画像サイズ。TCG正規化/復元の基準。 |
| `disp_info` | 現在の表示cropとscale。`params.set_disp_info()` ではmax sizeで正規化保存され、`get_disp_info()` でpxへ戻る。 |
| `rotation` | ユーザー操作の回転。radianへ変換済み。 |
| `rotation2` | EXIF/内部向き補正由来の追加回転。 |
| `flip_mode` | 反転/向き補正。 |
| `matrix` | 画像GeometryまたはMask Geometryを含む3x3行列。 |

不変条件:

- `original_img_size` が無いとTCG変換は成立しない。
- `disp_info` と `texture_size` は同じpreview状態を指していなければならない。
- `matrix` は画像GeometryとMask Geometryで一時的に差し替えられることがある。

#### window -> TCG

`params.window_to_tcg(cx, cy, widget, texture_size, tcg_info, normalize=True)` の概念順序:

```text
window px
  -> widget local px
  -> DPI scaleを除去
  -> widget内marginを除去
  -> texture Y flip
  -> texture内crop offsetを除去
  -> disp_info[4] でpreview scaleを戻す
  -> disp_info[0:2] でcrop originを足す
  -> imaxを引いて画像中心原点へ
  -> center_rotate_invert(rotation/flip/matrix inverse)
  -> normalize=TrueならTCG-normalizedへ
```

主な利用:

- ユーザーのtouch/mouse位置をMaskやGeometryの保存座標へ変換する。

落とし穴:

- `widget.size`, `widget.pos`, `texture_size`, `disp_info`, `dpi_scale` のどれかが古いと座標がずれる。
- テクスチャ座標ではY方向が反転するため、window/localとの往復で符号ミスが出やすい。

#### TCG -> window / texture / full image

`params.tcg_to_window()` の概念順序:

```text
TCG-normalized
  -> denorm_param でTCG-pixel
  -> center_rotate(rotation/flip/matrix)
  -> imaxを足してfull image系座標へ
  -> disp_info[0:2] でcrop originを引く
  -> disp_info[4] でpreview scaleを掛ける
  -> texture offsetを足す
  -> texture Y flip
  -> widget内marginを足す
  -> DPI scaleを掛ける
  -> widget.to_window(pos)を足す
```

`Mask2CoordinateContext` はheadless export用に、Kivy Widgetなしで同じ概念の変換を提供する。

| メソッド | 戻り先 | 用途 |
| --- | --- | --- |
| `window_to_tcg` | TCG-pixel | UI互換API。headlessではwindowは仮想。 |
| `tcg_to_window` | window px相当 | UI互換API。 |
| `tcg_to_texture` | texture px | マスクラスタ描画。 |
| `tcg_to_full_image` | full image px | Mask mesh MLS構築、画像Geometry込みのフル画像座標。 |
| `tcg_to_crop_image` | crop/ref image px | HLS maskなどcrop画像上で評価する処理。 |
| `tcg_to_original_image` | original image px | SAM3 bbox/text、DepthMap、Faceなど元画像推論へ渡す処理。 |

#### Mask2での保存単位

Mask2の多くの操作点は `.pmck` ではTCG-normalized相当で保存され、deserialize時に `params.denorm_param()` でTCG-pixelへ戻る。

例:

- CircularGradient / Gradient: centerや半径をTCG基準で保存。
- FreeDraw / Polyline: stroke pointをTCG基準で保存し、描画時にtexture pxへ変換。
- Segment / DepthMap / Face / TargetText: 中心やbboxをTCG経由で元画像座標へ戻して推論する。

設計意図:

- preview texture sizeが変わっても、保存されたマスク位置が元画像基準で保たれる。
- crop/zoom表示中でも、保存点は表示textureではなく画像の意味空間へ戻る。
- export時にheadless pipelineが同じ `.pmck` から再現できる。

#### CompositMaskとMask Geometry

CompositMaskは子マスクを束ね、最終的なmask imageを作る。

Mask Geometryが有効な場合:

```text
base = image-only matrix
M_mask = cores.mask2.mask_geometry.build_matrix_tcg(effects_param, original_img_size)
tcg_info["matrix"] = M_mask @ base
children get_mask_image()
tcg_info["matrix"] = saved_matrix
```

重要:

- `tcg_info["matrix"]` はCompositMask描画中に一時差し替えられる。
- `Mask2CoordinateContext._image_only_matrix` は、Mask Geometryを除いた画像Geometryだけの基準行列として保持される。
- Segment/Depth/Face/TargetTextなど `follows_mask_geometry()==False` な推論系マスクは、必要に応じて `_call_with_image_only_matrix()` でMask Geometry抜きのmatrixを使う。

#### texture_size同期の重要性

`texture_size` はpreview textureの実サイズであり、`disp_info[4]` と一組で意味を持つ。

典型的な破綻:

```text
window resize
  -> preview texture_size が 512 -> 768 に変わる
  -> disp_info[4] も 768 / max(original) に更新される
  -> しかし mask mesh editor の texture_size だけ 512 のまま
  -> tcg_to_window が新disp_infoと旧texture_sizeを混ぜて計算
  -> 制御点/マスク位置が数百px単位でずれる
```

この種のバグはタイミング問題に見えやすいが、本質は「同じ表示状態を表す `disp_info` と `texture_size` が同期していない」ことである。

確認すべき同期点:

- `MainWidget.resize()`
- `refresh_preview_overlays()`
- `mask_editor2.set_texture_size(texture_width, texture_height)`
- Geometry editor / Mask mesh editor の `set_texture_size`
- `Mask2CoordinateContext.set_texture_size` in export

#### previewとexportの座標一致条件

previewとexportでMask2位置が一致するための条件:

1. `.pmck` に保存された点がTCG基準である。
2. `primary_param.original_img_size`, `crop_rect`, `rotation`, `rotation2`, `flip_mode`, `matrix` が同じ意味で復元される。
3. previewでは `MaskEditor2.set_primary_param(primary_param, disp_info)`、exportでは `Mask2HeadlessPipeline.set_primary_param(primary_param, disp_info)` が同等の `tcg_info` を作る。
4. headless maskがUI maskと同じ `tcg_to_texture` / `tcg_to_full_image` の意味で描画する。
5. Mask Geometry有効時、`M_mask @ image_only_matrix` の適用順がUI/headlessで一致する。

#### TCGまわりを変更するときのチェック

- 保存形式はTCG-normalizedか、TCG-pixelか、texture pxか。
- `params.norm_param` / `denorm_param` の対象が点か、長さ/半径か。
- `original_img_size` は `(width, height)` で扱っているか。
- NumPy画像shapeは `(height, width, channels)` であることを忘れていないか。
- `disp_info` はpxに戻したものか、正規化保存されたものか。
- `texture_size` は現在のpreview widgetに同期済みか。
- Y flipを二重にかけていないか。
- Mask Geometryを含むmatrixか、image-only matrixか。
- UIとheadless exportの両方で同じ変換を通るか。

## 5. スレッド、非同期、キュー

### GUIスレッド

責務:

- Kivy Widget操作。
- Texture blit。
- UI状態変更。
- processing dialog表示/非表示。

注意:

- Kivy Widgetは基本GUIスレッドで触る。
- `draw_image_core()` は描画スレッドから呼ばれるが、UI反映は `KVClock.schedule_once` を使う箇所がある。

### 描画スレッド

`MainWidget.__init__` で `self.apply_thread = threading.Thread(target=self.draw_image, daemon=False)` を起動する。

責務:

- `draw_event` を待つ。
- 最新のcenter/fast/skip histogram指定で `draw_image_core()` を呼ぶ。

最新版採用:

- `pipeline_version` と `_last_processed_pipeline_version` で古いフレームを破棄する。
- `fast_display` やspace drag時は一時表示を許す場合がある。

### ファイルロード

`FileCacheSystem` はThreadPool/ProcessPoolを使う。

特徴:

- RAW previewは先に表示可能にする。
- RAW full decodeは共有メモリで戻す。
- 大きいFULL画像はcache投入前にmemmap化する場合がある。
- stall warningは `PLATYPUS_LOAD_STALL_WARN_SECONDS`。

キャンセル/取り違え:

- active/preload/cacheから削除するAPIはある。
- callback到着時点で現在期待しているfile pathと違う可能性があるため、MainWidget側の `_expected_file_path` が重要。

### AsyncWorker / AsyncPipelineManager

用途:

- 重いEffectをpreviewパイプラインから分離する。
- 実行中はpreview品質として扱い、完了後に再描画で結果を反映する。

仕様:

- frozen appでは `AsyncWorker.thread_mode=True` でスレッド実行。
- 通常は別プロセスと `multiprocessing.Queue`。
- 入力/出力画像は共有メモリ。
- `latest_tasks[effect_name]` で古いタスクをキャンセル扱いにする。
- `Effect.try_async_execution()` は `param_hash` と `efconfig.upstream_hash` を混ぜた `combined_hash` をキーにする。

不変条件:

- upstreamがPREVIEWなら重い非同期処理を新規開始しない。
- cache keyにはparamだけでなく入力画像由来のupstream hashが必要。
- Worker内には独立した `effects.create_effects()` がある。UI側Effectインスタンスを直接使わない。

### processing_dialog

`wait_processing()` は処理中HUDを表示しながら関数をThreadPoolで実行する。

注意:

- 旧名 `wait_prosessing` は綴り間違いだが、互換エイリアスとして残している。
- macOS + PyObjCが使える場合はネイティブHUD、それ以外はNull実装。

## 6. 画像処理パイプライン

### Effectレベル順

`effects.create_effects()` が5段階のdict listを返す。順序は挿入順であり仕様である。

| Level | 主なEffect | 役割 |
| --- | --- | --- |
| lv0 | loading_wait, ai_noise_reduction, remove_chromatic_aberration, lens_modifier, subpixel_shift, exposure_fusion_debevec, inpaint, patchmatch_inpaint, cross_filter, color_match, geometry, crop | ロード待ち、重い復元、RAW/幾何、crop前処理。 |
| lv1 | distortion, lensblur_filter, scratch, frosted_glass, mosaic, face | crop後の形状/フィルタ系。 |
| lv2 | color_temperature, auto_exposure, LUT, exposure, contrast, tone, level, curves, dehaze, denoise, HLS, film/look, glow, unsharp | 主なトーン/カラー/質感処理。 |
| lv3 | mask2, mask_geometry | マスク表示/マスクジオメトリ関連。 |
| lv4 | grain, vignette | 最終段。Mask composite後の画像に適用。 |

### プレビュー処理順

```text
process_pipeline
  -> guard original_img_size
  -> ensure_initial_crop_rect
  -> build EffectConfig(PREVIEW)
  -> pipeline_lv0
  -> crop/zoom/deferred geometry transform
  -> MaskEditor2 ref image update
  -> if not dragging:
       pipeline2
         -> primary lv1
         -> primary lv2
         -> primary lv3
         -> for each composit mask:
              mask lv1
              mask lv2
              mask image generation
              apply_mask_draw_effects
       pipeline_last(lv4)
     else:
       use cropped image only
```

### エクスポート処理順

```text
export_pipeline
  -> guard original_img_size
  -> build EffectConfig(EXPORT)
  -> pipeline_lv0
  -> crop full-resolution image first
  -> Mask2HeadlessPipeline set ref/update
  -> pipeline2
  -> pipeline_last(lv4)
  -> export.py color conversion / quantization / file write
```

### 色空間と値域

- 作業空間は基本 ProPhoto RGB リニア。
- RGB入力はICC profileを読み、sRGB等からProPhoto RGBへ変換する。
- RAW fullはLibRawからProPhoto RGB、gamma `(1, 1)`、32bit floatで得る。
- EXRはシーンリニアとして読み、chromaticitiesからProPhoto RGBへ変換する。HDR/負値を保持する意図がある。
- 表示時は `display_color_gamut` と `cat` に従って表示色域へ変換し、表示用にclipする。
- 非EXR exportは出力ICCへ変換、transfer付与、clip、8/16bit量子化。
- EXR exportはシーンリニア保持で、ガマットマッピング/transfer/clipをしない。

### プレビューとエクスポートで差が出る原因

- RAW previewは埋め込みJPEGベースでFULLと画質/色/座標が異なる。
- previewはtexture size/crop/zoomに合わせて縮小される。
- `Space` drag中は下流Effectをスキップする。
- AI async effectは未完了時にPREVIEW状態を返す。
- exportはheadless maskを使うため、UI専用状態に依存するMaskは再現できない。
- EXRと非EXRでclip/gamut/transferの扱いが違う。

## 7. 設定、プリセット、デフォルト

### 設定優先順位

```text
config.py のデフォルト
  < ~/Pictures/Shade Wave/config.json
  < 実行中のUI操作で config.set_config
  < 画像ごとの .pmck / primary_param
  < Effectごとの param default fallback
```

### 設定保存

- `config.set_config()` は `_apply_config()` 後に `save_config()`。
- `init_config()` はユーザーデータにconfigが無い場合に保存する。
- `utils.paths.ensure_user_data_dir()` は旧 `~/Pictures/Platypus` を `Shade Wave` へ移行する。

### UI初期値

- Effect UIの初期値は `effects.py` の `param_bindings` / `get_param_dict()`。
- `set2widget_all()` がparamをUIへ反映する。
- `delete_default_param_all()` が保存前にデフォルト値を落とす。

### プリセット

- `utils/preset_utils.py` と `~/Pictures/Shade Wave/presets`。
- export presetは `~/Pictures/Shade Wave/export_presets.json`。
- Film Process はプリセット JSON ではなく、Effect param の構造パラメータから生成する。

## 8. 外部依存と環境差

### セットアップ責務

`setup.sh` は以下を行う。

- pixi環境作成。
- `external/` 以下のclone。
- SAM3をpin commitへcheckoutし、macOS patchを適用。
- `requirements.txt` をpip依存の正本としてinstall。
- `depth_pro`, `mediapipe`, `libraw_enhanced` は個別条件でinstall。
- SCUNet CoreMLモデル生成。
- LibRaw build。
- radiance_denoise native build。
- ICC profile検証/取得。
- SCUNet/SAM3/Depth Pro model download。

### 外部コンポーネント

| コンポーネント | 差が出る点 |
| --- | --- |
| `libraw_enhanced` | Apple Clang、LibRaw build、Metal shader symlink、RAW形式差。 |
| `SAM3` | Hugging Face token、MPS/CPU fallback、macOS patch、checkpoint互換。 |
| `Depth Pro` | Apple公式外部プロジェクト、checkpoint、MPS/CPU。 |
| `SCUNet_CoreML` | coremltoolsとPyTorch互換、`.mlpackage` 生成。 |
| `radiance_denoise` | native extension build、Metal availability。 |
| `mediapipe` | NumPy 2維持のため `--no-deps` install。 |
| `Runware` | `RUNWARE_API_KEY` が無いとRunware系inpaint/object eraserは使えない。 |
| `exiftool` | metadata読取/書込。未導入・異常metadataで差が出る。 |

### GPU/CPU

- デフォルト `gpu_device` は `mps`。
- PyTorch MPSは未対応演算でCPU fallbackする。
- SAM3, Depth Pro, SCUNet系は環境差が大きい。
- frozen appではmultiprocessing start methodとKivy再初期化に注意。

### メモリ

環境変数:

- `PLATYPUS_MEMORY_DEBUG`
- `PLATYPUS_MEMORY_AVAILABLE_MIN_MB`
- `PLATYPUS_MEMORY_RSS_LIMIT_MB`
- `PLATYPUS_LOAD_STALL_WARN_SECONDS`

設計:

- 大きいFULL画像はmemmap化する場合がある。
- メモリ圧迫時はEffect diff、async processor cache、final display cacheを破棄する。
- `.pmck` heavy payloadは画像切替時に明示的に解放する経路がある。

## 9. 境界条件と落とし穴

### original_img_size未定義

再現:

1. ロード失敗やparam破損で `original_img_size` が入らない。
2. preview/exportを走らせる。

仕様:

- `process_pipeline()` と `export_pipeline()` はguardして処理しない。
- exportはログを出してFalse/None相当で中止する。

### RAW previewとFULLの差

再現:

1. RAWを開く。
2. 埋め込みpreview段階で編集/Mask表示。
3. FULL_DECODE後に画像サイズ/色/座標が変わる。

仕様:

- `ImageFidelity.PREVIEW` と `FULL` をparamに入れる。
- heavy payloadはFULLで保存/復元する。
- Mask2はRAW preview中に専用gateがある。

### crop/disp_info/TCG座標

落とし穴:

- crop, zoom, rotation, flip, matrix, mask geometryが同じ座標系を共有している。
- `disp_info` はpreview更新で変わる。
- Mask2 overlayとexport headlessの座標一致は `params.py` と `Mask2CoordinateContext` に依存する。

壊れやすい変更:

- `crop_rect` の型や意味変更。
- `disp_info` の要素順変更。
- preview texture sizeを画像座標として扱う変更。

### mask composite cache

落とし穴:

- Mask compositeはlv3後、lv4前に画像を変える。
- 子マスクやmask draw effectsが変わってもlv4 cacheが再利用されると見た目が古くなる。

仕様:

- `pipeline2()` はmask composite後に `lv1reset=True` と `efconfig.upstream_hash` 更新を行う。

### 非同期Effectの古い結果

落とし穴:

- paramだけでcacheすると、入力画像や上流Effect変更後に古いAI結果を使う。

仕様:

- `Effect.try_async_execution()` は `param_hash` と `efconfig.upstream_hash` を混ぜる。
- `latest_tasks` でeffect単位の旧タスクをcancel扱いにする。

### EXR/非EXRの出力差

仕様:

- EXRはHDR/負値/シーンリニアを維持するためclipしない。
- JPEG/TIFF/PNG/JXL/HEIFは出力profile transferを適用し、`[0,1]` clip後に量子化する。

### Runware API key

再現:

1. `RUNWARE_API_KEY` 未設定。
2. Runware backed inpaint/object eraserを使う。

仕様/現状:

- helperは未設定warningを出す。
- UI経路ではクラッシュまたは派手な失敗になる可能性がある。READMEにも既知制約として記載。

### PyInstaller/frozen app

落とし穴:

- Kivy, multiprocessing, PyTorch, CoreML, external package pathが絡む。
- `main.py` はfrozen時に `_MEIPASS` へchdirする。
- `scripts/build_macos_app_pyinstaller.py` はexternal path、checkpoints、ICC、SCUNet CoreML assetsを同梱する。

## 10. モジュール別仕様テンプレ適用

### main.py

概要:

- UIとアプリ状態の中心。ファイル選択、param反映、history、pipeline起動、export起動、ショートカット、メモリ管理を持つ。

入力:

- Kivy UIイベント、FileCacheSystem callback、AsyncWorker結果、キーボード。

出力:

- Texture表示、sidecar保存、export thread起動、UI状態更新。

主要クラス/関数:

- `MainWidget.__init__`: 状態初期化、Effect生成、AsyncWorker/描画スレッド開始。
- `on_select`: ファイル選択開始。
- `draw_image_core`: pipeline実行と表示更新。
- `start_draw_image`: pipeline_version更新と描画イベント送出。
- `save_current_sidecar`: `.pmck` 保存。
- `on_key_down`: `0`, Space, Cmd/Ctrl shortcuts。
- `MainApp.build/on_stop`: 設定ロード、worker start、終了処理。

エラー処理:

- 遅延callbackを期待パスで破棄。
- export cancel。
- memory pressure checkは例外を握ってログ。

テスト観点:

- ファイル選択中の別ファイルcallback破棄。
- RAW preview -> full遷移。
- Space drag/stale frame。
- sidecar保存タイミング。

### imageset.py

概要:

- 入力画像を作業用float32画像へ変換する。

入力:

- file path, exif_data, param。

出力:

- `ImageSet.img`, `param`, `LoadStage`。

主要関数:

- `preload`: 拡張子でロードタスクを返す。
- `_load_raw_preview`: 埋め込みpreviewをsRGB linear化し、元RAWサイズへresize。
- `_load_raw_full`: LibRaw full decodeを共有メモリで返す。
- `_load_rgb`: pyvips + ICC -> ProPhoto RGB。
- `_load_exr`: OpenEXR -> ProPhoto RGB linear。

落とし穴:

- RAW previewとfullの色/サイズ差。
- EXIF Orientationは読み込み時にgeometryへ反映し、削除する。
- Fuji geometry probe。

### pipeline.py

概要:

- Effect適用順序とpreview/export差の仕様本体。

入力:

- image, primary_effects, primary_param, mask_editor2, EffectConfig。

出力:

- preview/export処理後のfloat画像。

主要関数:

- `process_pipeline`
- `export_pipeline`
- `pipeline2`
- `pipeline_lv0/lv1/lv2/lv3`
- `pipeline_last`
- `AsyncPipelineManager`

不変条件:

- upstream変更時は下流Effectを `reeffect()`。
- async cache keyはupstream hash込み。
- mask composite後にlv4を必ず無効化。

### effects.py

概要:

- Effectクラス階層、param binding、Effect順序の定義。

入力:

- param dict, widget, image, EffectConfig。

出力:

- `diff`, `hash`, param更新、widget反映。

主要要素:

- `Effect.param_bindings`
- `set2widget`, `set2param`
- `try_async_execution`
- `create_effects`

落とし穴:

- `make_diff` と `apply_diff` の役割はEffectごとに違う。
- `diff is not None` が再計算/下流reset判定に使われる。
- binding共通化は便利だが、特殊UI副作用は `after_set2widget/after_set2param` へ逃がす。

### params.py

概要:

- param永続化、座標変換、特殊キー処理。

入力:

- primary_param, MaskEditor2/HeadlessPipeline, file path。

出力:

- `.pmck`, param復元、TCG/window座標変換。

重要:

- `.pmck` 後方互換。
- heavy payloadはFULL基準。
- lensfun user/default/capability/effectiveの分離。

### widgets/mask_editor2.py と cores/mask2

概要:

- UIマスク編集とexport再生。

入力:

- UI操作、image/crop/ref image、serialized mask dict。

出力:

- mask image、serialized `.pmck` mask2、composite result。

重要:

- UIクラスとheadlessクラスの機能差はexport再現性に直結する。
- AI mask推論は `cores.mask2.inference_runtime` がキャッシュを持つ。

### export.py

概要:

- フル解像ロードからファイル出力までを同期実行する。

入力:

- source path, export settings, exif_data, sidecar。

出力:

- 画像ファイル、ICC、metadata、rating。

落とし穴:

- EXRと非EXRで色変換仕様が違う。
- Mask2はheadless再生。
- cancel_eventを各段で確認する。

### file_cache_system.py

概要:

- ファイルロード/先読み/キャッシュ/メモリ管理。

入力:

- file path, metadata, callback。

出力:

- cache entry, callback実行。

落とし穴:

- shared memory cleanup。
- memmap化した配列の寿命。
- active/preload/cacheの不整合。

## 11. 解析・デバッグの実務手順

### まず見るログ/入口

- 起動: `MainApp.build`, `MainWidget.__init__`
- ファイルロード: `FileCacheSystem._load_file_thread`, `ImageSet._load_*`, `_task_callback`
- プレビュー: `MainWidget.start_draw_image`, `draw_image_core`, `pipeline.process_pipeline`
- エクスポート: `ExportFile.write_to_file`, `pipeline.export_pipeline`
- マスク: `MaskEditor2.serialize/deserialize/get_mask_image`, `Mask2HeadlessPipeline.deserialize`

### バグが出やすい境界

1. RAW preview -> FULL。
2. preview -> export。
3. UI Mask2 -> headless Mask2。
4. crop/rotation/zoom/mask coordinate。
5. async running -> result applied。
6. Effect cache invalidation。
7. `.pmck` heavy payloadの保存/復元。

### 最小再現を作る時の観点

- 入力形式: RGB / RAW / EXR。
- `ImageFidelity`: preview / full。
- crop/rotation/flip/matrixの有無。
- Mask2種類。
- async effectのrunning中か完了後か。
- export形式: EXR / 非EXR。
- GPU device: mps / cpu。

## 12. 変更時のチェックリスト

### Effectを追加/変更する

- `create_effects()` のどのlevelに入れるか決める。
- `param_bindings` とdefaultを定義する。
- `.pmck` に保存されるkey名を後方互換込みで決める。
- `make_diff/apply_diff` の返り値と `diff is not None` の意味を確認する。
- preview/export/Mask layerで同じ動作が必要か確認する。
- 重い処理ならasync化、upstream hash、FULL/preview保存ルールを確認する。

### Maskを追加/変更する

- UI実装だけでなくheadless実装を追加する。
- serialize/deserialize形式を決める。
- TCG/window/full image座標のどれで保存するか明記する。
- exportで同じmask imageになるテストを追加する。

### paramを変更する

- default削除で古い `.pmck` がどう読まれるか確認する。
- `delete_default_param_all`, `SPECIAL_PARAM`, heavy payloadに影響するか確認する。
- history diffで巨大配列をcopyしないか確認する。

### pipeline順を変える

- Effect cache invalidation。
- Mask composite位置。
- preview/export差。
- downstream `lvXreset`。
- `upstream_hash`。

### external依存を変える

- `setup.sh`, `requirements.txt`, `pixi.toml`, PyInstaller scriptを同時に確認する。
- frozen app同梱物、model path、editable installの扱いを確認する。
- NumPy 2互換と `--no-deps` installの意図を壊さない。

## 13. 既知の未整理点

現状仕様として把握しておくべき未整理点:

- `MainWidget` と `effects.py` は依然として非常に大きく、責務が集中している。
- `primary_param` はスキーマレスdictで、型安全性は低い。
- 一部helper/API key未設定時のUI経路は防御が薄い可能性がある。
- `async_worker.py` には子プロセス内のprint診断が残っている。
- `processing_dialog.wait_prosessing` は旧綴りの互換エイリアスとして残っている。
- PyInstaller/frozen環境は通常実行と挙動差が出やすい。

この文書は現状の安定化と今後の分割のための足場であり、変更後は必ず実コードに合わせて更新する。
