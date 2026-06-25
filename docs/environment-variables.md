# Environment Variables

Shade Wave / Platypus は、通常設定に出さない実験機能、デバッグ出力、backend 切替を環境変数で制御している。
この一覧は「アプリ実行時に意味があるもの」を中心にまとめる。`tests/` や benchmark script 専用のものは末尾に分ける。

真偽値は特記がなければ `1`, `true`, `yes`, `on` が有効、`0`, `false`, `no`, `off` が無効として扱われる。

## AI / Mask2

| 変数 | 既定 | 用途 |
| --- | --- | --- |
| `PLATYPUS_AI_DISPLAY_INPUT` | 有効 | AI系入力を表示用sRGBへ変換する。`0` で変換せず入力を渡す。現在SAM3 bbox/textは個別に未変換入力を使う。 |
| `PLATYPUS_AI_COMPLETED_CACHE_MAX_MB` | `512` | AI-NR完了結果の短期再利用キャッシュ上限。上限超過時は古い完了結果から破棄し、単体で上限を超える結果はキャッシュせず処理側へ直接返す。`0` でキャッシュ無効。 |
| `PLATYPUS_AI_JOB_NICE` | `10` | AI Job worker process のnice増分。大きいほどCPU優先度を下げる。`0` で無効。thread modeでは親プロセス全体に影響するため適用しない。 |
| `PLATYPUS_AI_SIDECAR_MERGE_MAX_PENDING` | `2` | AI-NR完了後の `.pmck` background merge キュー上限。上限超過時はUI/OOM保護を優先して該当mergeを破棄する。 |
| `PLATYPUS_SAM3_BBOX_CLIP` | 有効 | SAM3 bbox結果を指定bbox外でゼロにする。 |
| `PLATYPUS_SAM3_ROI_INPUT` | 有効 | SAM3 bbox推論を画像全体ではなくbbox周辺ROIで行う。 |
| `PLATYPUS_SAM3_ROI_SCALE` | `1.5` | SAM3 bbox用ROIをbboxの縦横何倍に広げるか。 |
| `SCUNET_APPLESILICON_ROOT` | 未指定 | Apple Silicon版SCUNet helper の外部root。 |

## Quick Select / Edge Refine

| 変数 | 既定 | 用途 |
| --- | --- | --- |
| `PLATYPUS_DEBUG_EDGE_REFINE` | 無効 | Edge refine / Draw Quick Select のデバッグdumpやログを有効にする。 |
| `PLATYPUS_DEBUG_EDGE_REFINE_DIR` | `/tmp/platypus_edge_refine` など | Edge refine debug dump の出力先。 |
| `PLATYPUS_DEBUG_EDGE_REFINE_LIMIT` | `80` | Edge refine debug dump の最大数。`-1` で無制限。 |
| `PLATYPUS_DRAW_QS_LEGACY` | 無効 | Draw Quick Select の旧経路へ戻す。 |
| `PLATYPUS_DRAW_QS_FULL_VIEW` | 有効 | Quick Select のfull-view refine経路を使う。`0` で無効。 |
| `PLATYPUS_DRAW_REFINE_MAX_PIXELS` | `1200000` | Draw refine の最大処理ピクセル数。 |
| `QS_DUMP_INPUT` | 未指定 | Quick Select 入力dump先。 |
| `QS_DUMP_INPUT_LIMIT` | `500` | Quick Select 入力dump最大数。 |
| `QS_DRAW_V2` / `QS_DRAW_V3` / `QS_DRAW_V4` | build/runtime依存 | Quick Select solver 世代の切替。 |
| `QS_DRAW_SOLVER` | 自動 | metrics系で solver を明示する。 |
| `QS_EDGE_PERCEPTUAL` | 有効 | Edge計算をperceptual寄りにする。`0` で無効。 |
| `QS_FULLVIEW_VALIDITY` | 有効 | full-view validity 判定を使う。 |
| `QS_RIM_EDGE_T` | `0.08` | v3 rim edge threshold。 |
| `QS_V2_CANONICAL_SCALE` | 有効 | v2 canonical scale を使う。`0` で無効。 |
| `QS_V2_STABLE_EDGE` | 無効 | v2 stable edge 経路。 |
| `QS_V2_STRENGTH_MODE` | 自動 | v2 strength mode の実験切替。 |
| `QS_DRAW_V2_OFFSET` | 無効 | v2 offset strength mode の互換フラグ。 |
| `QS_V4_EDGE_SNAP` | runtimeでは通常有効 | v4 edge snap。 |
| `QS_V4_SNAP_ALPHA` | 実装既定 | v4 snap alpha。 |
| `QS_V4_TRACE_DISTPRIOR` | 無効 | v4 distance prior tracing。 |

## Preview / Interaction

| 変数 | 既定 | 用途 |
| --- | --- | --- |
| `PLATYPUS_GE_PREVIEW_RENDER` | `full` | Geタブのプレビュー描画方式。既定はfull-preview描画。`fast` などで通常のfast preview寄りに戻す。 |
| `PLATYPUS_DRAG_PREVIEW_RENDER` | `fast` | 通常ドラッグプレビューの描画方式。`full`, `quality`, `complete` などでfull描画寄りにする。 |
| `PLATYPUS_GE_PREVIEW_DRAIN_ALL` | 有効 | Ge full-preview時、溜まった描画versionを順に処理する。`0` で最新寄りだけ処理する。 |
| `PLATYPUS_DRAG_PREVIEW_DRAIN_ALL` | 無効 | 通常ドラッグプレビューで溜まった描画versionを順に処理する実験用。 |
| `PLATYPUS_GE_PREVIEW_ALLOW_STALE` | 有効 | Ge full-preview時、現在versionより古い完成フレームのblitを許可する。`DRAIN_ALL=1` と組み合わせた最速体感用。 |
| `PLATYPUS_DRAG_PREVIEW_ALLOW_STALE` | 有効 | 通常drag/fast previewで古い完成フレームのblitを許可する。 |
| `PLATYPUS_GE_PREVIEW_INTERPOLATION` | `linear` | Ge編集中のdeferred geometry補間。`linear`, `area`, `nearest`。`pyramid_linear` は現在 `linear` にfallbackする。 |
| `PLATYPUS_PREVIEW_OVERLAY_AFTER_BLIT` | 無効 | preview overlay同期を画像blit後のresize経路へ寄せる実験用。 |
| `PLATYPUS_PREVIEW_OVERLAY_FRAME_DELAY` | 無効 | CropEditor枠線更新を次Kivy tickへ遅延する実験用。 |

## Effect / Image Backends

| 変数 | 既定 | 用途 |
| --- | --- | --- |
| `PLATYPUS_IMAGE_TRANSFORM_BACKEND` | `auto` | 画像変換backend。`metal`, `gpu`, `reference`, `opencv`, `off` など。 |
| `PLATYPUS_IMAGE_TRANSFORM_AREA_MODE` | `exact` | Metal area補間の扱い。`exact`, `linear`, `reference`。 |
| `PLATYPUS_IMAGE_TRANSFORM_METAL_STRICT` | 無効 | Metal画像変換失敗時にfallbackせず例外を投げる。 |
| `PLATYPUS_CROSS_FILTER_BACKEND` | `auto` | CrossFilter backend。`metal`, `cpu`, `reference` など。 |
| `PLATYPUS_CROSS_FILTER_METAL_STRICT` | 無効 | CrossFilter Metal失敗時にfallbackせず例外を投げる。 |
| `PLATYPUS_LOW_FREQUENCY_TRANSFER_BACKEND` | `auto` | Low Frequency Transfer backend。`metal`, `cpu`, `exact`, `reference` など。 |
| `PLATYPUS_LOW_FREQUENCY_TRANSFER_DOWNSAMPLE` | `off` | Low Frequency Transfer の低解像度処理。`auto`, `on`, 数値倍率など。 |
| `PLATYPUS_LOW_FREQUENCY_TRANSFER_LUMA_STRENGTH` | `1.0` | Low Frequency Transfer の輝度移植強度。 |
| `PLATYPUS_LOW_FREQUENCY_TRANSFER_METAL_STRICT` | 無効 | Low Frequency Transfer Metal失敗時にfallbackせず例外を投げる。 |
| `PLATYPUS_LOW_FREQUENCY_TRANSFER_STRICT` | 無効 | Low Frequency Transfer native失敗時にfallbackせず例外を投げる。 |
| `PLATYPUS_SUBPIXEL_SHIFT_BACKEND` | `auto` | Subpixel shift backend。`reference` などで戻せる。 |
| `PLATYPUS_SUBPIXEL_SHIFT_STRICT` | 無効 | Subpixel shift native失敗時にfallbackせず例外を投げる。 |
| `PLATYPUS_TONE_BACKEND` | `auto` | Tone backend。`reference` などで戻せる。 |
| `PLATYPUS_TONE_STRICT` | 無効 | Tone native失敗時にfallbackせず例外を投げる。 |
| `PLATYPUS_COLOUR_FUNCTIONS_BACKEND` | `auto` | colour functions backend。`reference` などで戻せる。 |
| `PLATYPUS_COLOUR_FUNCTIONS_STRICT` | 無効 | colour functions native失敗時にfallbackせず例外を投げる。 |
| `PLATYPUS_COLOR_SEPARATION_BACKEND` | `auto` | Color Separation backend。`reference`, `python`, `off` などでreference実装へ戻せる。 |
| `PLATYPUS_COLOR_SEPARATION_STRICT` | 無効 | Color Separation native失敗時にfallbackせず例外を投げる。 |
| `PLATYPUS_FILM_GRAIN_BACKEND` | `auto` | Film Grain backend。`reference`, `python`, `off` などでreference実装へ戻せる。 |
| `PLATYPUS_FILM_GRAIN_STRICT` | 無効 | Film Grain native失敗時にfallbackせず例外を投げる。 |

## Startup

| 変数 | 既定 | 用途 |
| --- | --- | --- |
| `PLATYPUS_SPLASH_SCREEN` | 有効 | 起動時に `assets/Shade Wave.png` のスプラッシュスクリーンを表示する。`0` で無効。 |

## Debug / Logging

| 変数 | 既定 | 用途 |
| --- | --- | --- |
| `PLATYPUS_DEBUG_MASK_GEOMETRY` | 無効 | Mask geometry の座標・状態ログ。 |
| `PLATYPUS_DEBUG_MASK_ZOOM_SYNC` | 無効 | Mask/zoom同期ログ。 |
| `PLATYPUS_DEBUG_MESH_WARP` | 無効 | Mesh warp / distortion correction のdebug log。 |
| `PLATYPUS_DEBUG_PIPELINE_STATS` | 無効 | pipeline のstage/effect timing統計。 |
| `PLATYPUS_DEBUG_PIPELINE_STATS_VERBOSE` | 無効 | pipeline timing の詳細出力。 |
| `PLATYPUS_DEBUG_PIPELINE_STATS_EFFECTS` | 未指定 | timing対象effect名のカンマ区切りfilter。 |
| `PLATYPUS_DEBUG_PIPELINE_STATS_LABELS` | 未指定 | timing対象labelのカンマ区切りfilter。 |
| `PLATYPUS_PIPELINE_TIMING` | 無効 | pipeline timing summary。 |
| `PLATYPUS_PIPELINE_TIMING_EVERY` | `1` | timing summary の出力間隔。 |
| `PLATYPUS_DEBUG_NAN_INF` | 無効 | NaN/Inf検出debug dump。 |
| `PLATYPUS_RESIZE_DEBUG` | 無効 | window/texture resize debug log。 |
| `PLATYPUS_PERF_TRACE` | 無効 | `utils.perf_trace` のtrace出力。 |

## Load / Memory

| 変数 | 既定 | 用途 |
| --- | --- | --- |
| `PLATYPUS_LOAD_STALL_WARN_SECONDS` | `15` | loader stall warning の秒数。 |
| `PLATYPUS_LOAD_POOL_START_METHOD` | 自動 | load process pool のstart method。macOS appではfork寄せのために使う。 |
| `PLATYPUS_FORCE_THREAD_LOAD_POOL` | 無効 | process poolではなくthread poolを強制する。 |
| `PLATYPUS_FINAL_DISPLAY_CACHE_MAX` | `8` | final display image cache の最大枚数。 |
| `PLATYPUS_MEMORY_DEBUG` | 無効 | memory manager のdebug log。 |
| `PLATYPUS_MEMORY_AVAILABLE_MIN_MB` | `1024` | 空きメモリ下限MB。 |
| `PLATYPUS_MEMORY_RSS_LIMIT_MB` | `0` | RSS上限MB。`0` は無効。 |

## External Services

| 変数 | 既定 | 用途 |
| --- | --- | --- |
| `RUNWARE_API_KEY` | 未指定 | Runware object eraser / inpaint 系API key。 |
| `RUNWARE_API_URL` | `https://api.runware.ai/v1` | Runware API endpoint。 |
| `RUNWARE_REQUEST_TIMEOUT` | `120` | Runware request timeout秒。 |
| `RUNWARE_OBJECT_ERASER_MODEL` | `runware:300@1` | Object eraser model id。 |
| `RUNWARE_OBJECT_ERASER_STEPS` | `4` | Object eraser steps。 |
| `RUNWARE_OBJECT_ERASER_CFG` | `1` | Object eraser CFG scale。 |
| `RUNWARE_OBJECT_ERASER_BLEND_DILATE` | `8` | Runware結果blend maskのdilate px。 |
| `RUNWARE_OBJECT_ERASER_BLEND_BLUR` | `5` | Runware結果blend blur px。 |
| `DASHSCOPE_API_KEY` | 未指定 | Qwen/DashScope image helper API key。 |
| `DASHSCOPE_REQUEST_TIMEOUT` | `60` | DashScope request timeout秒。 |
| `PLATYPUS_NANO_BANANA_MODEL` | `gemini-3.1-flash-image` | Nano Banana helper の通常model。 |
| `PLATYPUS_NANO_BANANA_EDIT_MODEL` | `imagen-3.0-capability-001` | Nano Banana edit model。 |
| `PLATYPUS_NANO_BANANA_FALLBACK_MODELS` | `gemini-2.5-flash-image` | fallback modelのカンマ区切り。 |

## Build / Runtime Compatibility

| 変数 | 既定 | 用途 |
| --- | --- | --- |
| `KIVY_NO_ARGS` | 起動時に設定 | Kivy がCLI引数を消費しないようにする。 |
| `KMP_WARNINGS` / `OMP_DISPLAY_ENV` / `LIBOMP_VERBOSE` | 起動時に抑制値 | OpenMP/libompのnoise抑制。 |
| `KMP_DUPLICATE_LIB_OK` | build/test/runtime依存 | libomp重複ロード対策。通常は安易に有効化しない。 |
| `PYTORCH_ENABLE_MPS_FALLBACK` | macOS app runtimeで `1` | PyTorch MPS未対応opをCPU fallbackする。 |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | macOS app runtimeで `0.0` | MPS memory watermark調整。 |
| `PYTORCH_MPS_FAST_MATH` | macOS app runtimeで `0` | MPS fast mathの明示設定。 |
| `ENABLE_PJRT_COMPATIBILITY` | macOS app runtimeで `1` | bundled runtime互換用。 |
| `KIVY_HOME` | build scriptで設定 | PyInstaller build用Kivy home。 |
| `DYLD_LIBRARY_PATH` | runtime hookで追加 | bundled library探索path。 |

## Test / Benchmark Helpers

| 変数 | 用途 |
| --- | --- |
| `PLATYPUS_REAL_PIPELINE_PREVIEW_SIZE` | headless real pipeline test のpreview size。 |
| `PLATYPUS_SELECTION_STRESS_MAX_INTERVAL` | file selection stress test の最大間隔。 |
| `PLATYPUS_SELECTION_STRESS_REAL_SLEEP` | file selection stress test のsleep有無。 |
| `PIXI_PROJECT_ROOT` | perf/debug dumpやtestでproject rootを明示する。 |
| `CONDA_PREFIX` | native extension build/check script用。 |
| `COLORTERM` | terminal color capability判定。 |
