# Effect Backend Design

## 目的

Platypus の高レベルな `Effect` クラスは、UI、パラメータ、キャッシュ、非同期実行、履歴連携を受け持つ。一方で画像計算本体は、Python / C++ / Metal / OpenCL / 将来の別ランタイムへ段階的に移せるよう、安定した薄いバックエンド境界へ切り出す。

この設計のゴールは次の3つ。

- 1ファイル1エフェクト単位で、単体移植できる。
- Python からは単純な関数呼び出しで使え、未ビルド環境では既存実装へ戻れる。
- 将来 Python 側が別言語に置き換わっても、画像バッファと小さなパラメータ構造だけで同じ実装を呼べる。

## レイヤー

```text
effects.py / pipeline.py
  UI state, params, cache, async policy, history policy
        |
        v
effect_backends/<effect>_adapter.py
  stable adapter API, backend selection, dtype/contiguity normalization,
  Python fallback, backend_status()
        |
        v
backend implementation
  Python fallback / C++ pybind11 / Metal / OpenCL / future runtime
```

`effects.py` は実装技術を知らない。`effect_backends.vignette_adapter.apply_vignette()` のような安定APIだけを呼ぶ。

## ファイル命名ルール

複数フォルダに `adapter.py` や `reference.py` のような同名ファイルを量産しない。検索性とタブ上の見分けやすさを優先し、ファイル名には必ず effect 名を入れる。

基本形は次のようにする。

```text
effect_backends/
  vignette_adapter.py       # Pythonからの安定入口
  vignette_reference.py     # 旧Python/Numba実装、fallback、数値基準
  vignette_capi.h           # Swift/Python/将来UI共通のC ABI
  vignette_cpu.c            # Python非依存のCPU backend
  vignette_pybind.cpp       # Python bindingのみ。計算ロジックを持たない
```

将来ファイル数が増えてフォルダ分けする場合も、同じ命名を維持する。

```text
effect_backends/
  vignette/
    vignette_adapter.py
    vignette_reference.py
    vignette_capi.h
    vignette_cpu.c
    vignette_pybind.cpp
```

禁止する形。

```text
effect_backends/
  vignette/adapter.py
  geometry/adapter.py
  cross_filter/adapter.py
```

## 実装境界のルール

`pybind11` は Python から呼ぶための binding であり、backend 本体ではない。計算ロジックは `vignette_cpu.c` などの Python 非依存ファイルへ置く。

```text
vignette_adapter.py
  -> Python-facing API, backend selection, fallback

vignette_pybind.cpp
  -> NumPy配列とC ABI構造体の変換だけ
  -> GIL release
  -> vignette_apply_v1() を呼ぶ

vignette_cpu.c
  -> Python/pybind11をincludeしない
  -> C ABIを実装する
```

`vignette_cpu.c` は CPU backend の明示名であり、実装言語を永遠にCへ固定する意図ではない。外に見える境界はC ABIに揃える。必要なら内部実装はC++/Objective-C++/Metalへ逃がしてよいが、PythonやSwiftから見える入口は小さなC ABIに保つ。

## 現在のサンプル

Vignette を最初のサンプルとして追加した。

```text
effect_backends/
  __init__.py
  setup.py
  vignette_adapter.py      # Python adapter
  vignette_reference.py    # Python/Numba fallback
  vignette_capi.h          # C ABI
  vignette_cpu.c           # CPU backend
  vignette_pybind.cpp      # pybind11 bridge
tests/
  test_vignette_backend.py
```

`VignetteEffect.make_diff()` は、従来の `cores.core.apply_vignette()` ではなく `effect_backends.vignette_adapter.apply_vignette()` を呼ぶ。compiled backend が import できる場合は `vignette_pybind.cpp` 経由で `vignette_cpu.c` を呼び、できない場合は `vignette_reference.py` の既存 Numba 実装へフォールバックする。

この命名をVignetteの実装で採用し、2つ目以降のeffectも同じ規則に揃える。

CrossFilter も同じ規則へ移行する。

```text
effect_backends/
  cross_filter_adapter.py
  cross_filter_reference.py
  cross_filter_capi.h
  cross_filter_cpu.c
  cross_filter_pybind.cpp
  cross_filter_metal.mm
tests/
  test_cross_filter_backend.py
  test_cross_filter_metal_backend.py
```

`cores/cross_filter.py` は互換shimにする。旧単体実装は `cross_filter_reference.py` に移し、Python fallback と数値・挙動確認の基準にする。

CrossFilter の `cross_filter_cpu.c` backend は、Python 非依存実装とC ABI境界のサンプルとして用意する。初期の素朴実装はOpenCV参照実装より遅かったが、指数減衰カーネルを行方向の再帰フィルタへ置き換えたため、現時点ではCPU backendを既定にする。参照実装へ戻したい場合は `PLATYPUS_CROSS_FILTER_BACKEND=reference` を指定する。

GPUへの第一歩として `cross_filter_metal.mm` も用意する。Metal backendは利用可能なら既定で使う。これはreference完全一致を目指すものではなく、mini buffer上で光条方向を直接サンプルするGPU近似backendとして扱う。問題がある場合は `PLATYPUS_CROSS_FILTER_BACKEND=cpu` または `PLATYPUS_CROSS_FILTER_BACKEND=reference` で戻せる。

CrossFilter backend の実測には `scripts/bench_cross_filter_backends.py` を使う。Codexのコマンド実行環境では Metal extension はビルドできても `MTLCreateSystemDefaultDevice()` が `nil` を返す場合があるため、Metal backend は実際のアプリ実行環境でも確認する。

## 既存 `core.py` 関数の扱い

旧実装を `cores/core.py` に残し続けると、effect の所有権が分散する。移植対象になった effect は、Python reference 実装も含めて `effect_backends` 側へ寄せる。

Vignette の場合は次の方針にする。

```text
effect_backends/vignette_reference.py
  -> 旧 cores.core.apply_vignette の実体を移す
  -> fallback と数値基準を担当

cores/core.py
  -> apply_vignette は互換shimとして残してよい
  -> 内部で effect_backends.vignette_adapter.apply_vignette を呼ぶ
  -> 新しい計算ロジックは追加しない
```

新規 effect backend を追加する時も同じ方針にする。`core.py` は大きな画像処理実装の置き場から、古い呼び出し口の互換層へ徐々に役割を縮小する。

## Effect Backend API ルール

各 effect module は次の形を基本にする。

```python
def apply_<effect>(image: np.ndarray, ..., *, options=None) -> np.ndarray:
    ...

def backend_status() -> BackendStatus:
    ...
```

守るべきルール。

- 入力画像は原則 `float32`, `H x W x 3`, pipeline 現在色空間のまま渡す。
- adapter は必要最小限の dtype/contiguous 変換だけ行う。
- backend は Kivy widget、global config、`param` dict、UI 状態を直接読まない。
- 返り値は入力と同じ shape の新しい `float32` 配列を基本にする。
- in-place 処理や scratch/cache は、API上で明示するまで隠し仕様にしない。
- Python boundary は effect につき1回にする。pixel単位、行単位、callback単位で Python に戻らない。
- backend は可能なら GIL を解放する。
- 既存実装との数値差は effect ごとに test tolerance を決める。

## Backend Selection

現段階の優先順は単純にする。

```text
compiled backend available -> use it
otherwise -> existing Python/Numba fallback
```

将来、複数 backend を持つ場合は adapter に集約する。

```text
effect_backends/vignette_adapter.py
  _vignette_metal
  _vignette_cpu
  vignette_reference.apply_vignette
```

環境変数や config による強制指定は adapter 層だけで扱う。`effects.py` 側に `if metal_available` のような分岐を増やさない。

## 将来の ABI

pybind11 は現在の Python から呼ぶための adapter であり、最終ABIではない。長期的には、実装本体は Python 非依存の薄いABIへ寄せる。Swift/AppKit/Metal UIへ移行しても、同じC ABIを呼べる状態を保つ。

```c
typedef struct {
    float* data;
    int width;
    int height;
    int channels;
    int stride_bytes;
} VignetteImageF32;

typedef struct {
    float intensity;
    float radius_percent;
    float gradient_softness;
    float disp_info[5];
    float crop_rect[4];
    float offset[2];
} VignetteParams;

int vignette_apply_v1(
    const VignetteImageF32* input,
    VignetteImageF32* output,
    const VignetteParams* params
);
```

Python/pybind11、将来のRust/Swift/C++ UI、GPU command encoder は、このABIまたは同等の typed buffer contract を呼ぶだけにする。

大元アプリをPython以外へ移すこと自体は、この設計の前提にしない。まずはPython UIを維持し、重い処理と表示変換だけをbackend化する。UI基盤、macOS統合、ColorSync、Metal texture管理が主な制約になった時点で、Swift等への移行を検討する。

## パフォーマンス方針

単純なC移植より、次を優先する。

- full-frame pass をまとめる。
- 中間配列を減らす。
- map/mask/kernel を backend 側で cache できる形にする。
- geometry/remap/convolution のような処理は CPU native だけでなく GPU backend を候補にする。
- adapter で大きな copy を作らない。copy が必要な場合は理由を明確にする。
- CPU native backend の並列化は、可能なら OpenMP を優先する。`pthread` は既存実装との互換や、ビルド環境で OpenMP を使えない場合の fallback として扱う。新規 backend では `#pragma omp parallel for` で row/tile 単位の並列化を行い、pybind 層では引き続き GIL を解放する。

## Core Image Transform Backend

`core.crop_image()` や `core.rotation()` は通常のeffectではないが、同じbackend境界で扱う。名前は処理意図を優先し、`fit_crop_to_canvas()` と `transform_to_canvas()` を安定APIにする。C ABI名にもアプリ名prefixは付けず、同じ意図の名前を使う。

```text
effect_backends/
  image_transform_adapter.py
  image_transform_reference.py
  image_transform_capi.h
  image_transform_metal.mm
```

`fit_crop_to_canvas()` は、cropした矩形を指定サイズへリサイズし、ゼロ埋めcanvasへ配置する。現在の `core.crop_image()` の非ズームpathは `INTER_AREA` + constant zero pad、ズームpathは `zoom_ratio >= 1.0` で `nearest`、それ以外で `area` を使う。この挙動をreference contractにする。

Metal backend が利用可能な場合、`fit_crop_to_canvas()` は `nearest` / `linear` / `area` を `_image_transform_metal` へ流す。`area` は既定でsource coverage平均のMetal areaを使い、縮小表示の品質を保つ。`PLATYPUS_IMAGE_TRANSFORM_AREA_MODE=linear` で旧preview向けlinear近似、`reference` でOpenCV referenceへ戻せる。`PLATYPUS_IMAGE_TRANSFORM_BACKEND=reference` は常にOpenCV referenceを使う。

Metal backend は NumPy 配列を `newBufferWithBytesNoCopy` で直接読む/書く。CPU ndarray へ戻す現在のpipelineでは、このno-copy化が重要になる。8192x5120 -> 1600x1000 の純粋なcrop/resizeでは、Metal exact areaが約41ms、OpenCV referenceが約37msだった。純粋なcropだけではOpenCVが僅かに速い場合があるが、変形とcropを融合したpathでは中間画像を避けられるため大きく効く。

`transform_to_canvas()` は affine / perspective / rotation / warp を扱う予約APIとする。最初から全remap処理を統合せず、`core.rotation()` のpreview pathから段階的に載せる。

`core.rotation()` は `transform_to_canvas()` へ接続する。Metal backend は当面 `float32 RGB` の `linear` preview transform を担当し、`bicubic` export、2D mask、未対応dtypeはOpenCV referenceへ戻す。OpenCV完全一致ではなく、編集中の操作感を優先するpreview backendとして扱う。

40MP級では `rotation -> full square image -> crop/resize` の中間画像が支配的になる。8192x5120 を回転すると 8192x8192 の float32 RGB、約805MBの中間配列になる。これを避けるため `transform_crop_to_canvas()` を追加し、変換後キャンバス上のcrop矩形を最終preview canvasへ直接描く。

`GeometryEffect + CropEffect` のpreview fast pathとして、Metal利用可能、`float32 RGB` の場合にこの融合pathを使う。zoom中も `crop_image` と同じ zoom crop 計算を使って `transform_crop_to_canvas()` へ流す。zoom時は既存指定どおりcrop拡大段を `nearest` にし、ドットが見える操作感を残す。mesh、export、2D mask、Metal不可環境では従来pathへ戻す。lens distortion は `lens_distortion_strength` のみ、かつ `lens_distortion_scale == 0` のpreviewで融合対象にする。lens scaleはゼロ埋めとgeometry borderの組み合わせを保守的に扱うため、現時点では従来pathに残す。8192x5120 -> 1600x1000 previewでは、回転full square生成後にcropする2 pass Metal areaが約154ms、融合Metal areaが約72msだった。zoom 2.0 ではcrop拡大段をnearestにし、2 passが約117ms、融合Metalが約17msだった。lens strength 30 の 8192x5120 -> 1600x1000 previewでは、旧二段pathが約234ms、融合Metalが約23msだった。

trapezoid、four-point、reference lines は同じ deferred homography として扱う。CPU側では3x3行列だけを計算し、画像本体は `transform_crop_to_canvas()` で最終previewへ直接描く。8192x5120 で rotation + trapezoid + crop を行う条件では、2 pass Metalが約168ms、融合Metalが約27msだった。reference lines は既存の `calculate_lines_homography()` を使い、2本以上で行列が作れる場合のみdeferred pathへ合成する。

mesh warp は `mesh_rbf_function=mls` を前提に、previewではMLS coarse mapだけをCPUで生成して `transform_crop_to_canvas()` に渡す。Metal側は最終canvasの各ピクセルで crop逆変換、MLS map sample、homography逆変換、lens sampleを1回のkernelで行う。exportとMetal不可環境は従来の `warp_mesh()` pathへ戻す。preview用coarse gridは64px stepとし、既存の32px step/full remap品質はexport側に残す。8192x5120 -> 1600x1000 previewでは、MLS map生成が約69ms、融合Metal描画が約23ms、旧二段pathが約431msだった。

Vignette は sample として扱いやすいが、すでに Numba で速い。実測では 900x1200 の合成画像で、既存 Numba が約 4-5ms、C++ backend が約 2-3ms 程度だった。速度面の本命というより、移植動線の基準実装として価値がある。

## colour_functions について

`colour_functions_reference.py` の表示変換は重い。特に `main.draw_image_core()` の pipeline 後段で毎フレーム通るため、pipeline timing の effect 集計には出にくいが体感には効く。

既存ドキュメント `docs/gpu-display-color-management.md` には、通常表示変換 `color_ms` が約 88-120ms、fast display が約 20ms 前後まで下がる記録がある。手元の 900x1200 合成画像でも、matrix 変換自体は数msだが、negative gamut compression と display encode が大きい。

重要なのは、ここは通常の effect ではなく display transform であること。設計上は effect chain に混ぜず、同じ adapter/backend 方針を使った別ノードとして扱う。旧 `cores/colour_functions.py` は残さず、参照実装は `effect_backends/colour_functions_reference.py`、実運用の入口は `effect_backends/colour_functions_adapter.py` にする。

```text
main.draw_image_core()
  -> effect_backends/colour_functions_adapter.py
       _colour_functions_cpu fused backend
       future GPU shader backend
       effect_backends.colour_functions_reference fallback
```

CPU native 化するなら、matrix、negative gamut compression、transfer encoding を1パスへ融合する。`colour_functions_adapter.py` は従来の `colour_functions` APIを再エクスポートし、表示変換系だけをnative優先で上書きする。より大きな方向性としては、preview 表示だけ GPU shader または 3D LUT shader へ移すのが本命。

## 3D LUT Backend

外部 `.cube` の 3D LUT trilinear 補間を backend 化した。`image_transform` と同型で、**Metal + reference のみ**
（CPU C backend は持たない）。

```text
effect_backends/
  lut_adapter.py     # Python adapter（backend選択・fallback・backend_status）
  lut_reference.py   # NumPy 参照実装＝数値基準＆fallback（apply_lut3d）
  lut_capi.h         # 将来ABI用 C ABI 構造体
  lut_metal.mm       # Metal compute kernel + pybind（_lut_metal）
tests/
  test_lut_backend.py
scripts/
  bench_lut_backends.py
```

- `cores/lut_functions.py` の `LUT3D.apply` は `lut_adapter.apply_lut3d(image, table, domain, size)` への
  互換 shim。`overrange`/`lut_intensity` ブレンド/linear→log/1D LUT は Python 側（`cubelut`/`LUTEffect`）に残し、
  Metal 化したのは 3D trilinear のコアのみ。
- 選択: `PLATYPUS_LUT_BACKEND`(`auto`/`metal`/`reference`)、`PLATYPUS_LUT_METAL_STRICT`。Metal 不可環境では
  自動で `lut_reference` にフォールバック。
- **BGR インデックス規約**: 入力 RGB grid 座標 `g=(gR,gG,gB)=norm*(size-1)` に対し `table[a,b,c]`
  （flat=`((a*size+b)*size+c)*3`）を `a=floor(gB), b=floor(gG), c=floor(gR)`、補間重み
  axis0=frac(gB)/axis1=frac(gG)/axis2=frac(gR) で参照する。reference と kernel はこの規約を一致させる
  （tolerance < 1e-4、フル解像度で約45x）。

> 補足: 現行 backend は vignette / cross_filter / image_transform / colour_functions / tone /
> color_separation / film_grain / subpixel_shift / low_frequency_transfer / lut。上の「現在のサンプル」節は
> 初期 2 例（vignette / cross_filter）の記述で、実体はこの一覧まで拡張済み。

## 移植順の考え方

1. Vignette で adapter/backend/test/build の型を固める。
2. `display_transform` を同じ思想で分離し、毎フレーム後段コストを削る。
3. Geometry / LensModifier / CrossFilter のような map・remap・convolution 系は、cache と pass fusion を前提に backend 化する。
4. Basic color stack は effect 個別ではなく、融合 backend として検討する。

この順にすると、Python の現行UIを保ちながら、将来のフルネイティブ化でも呼び出し境界が散らからない。
