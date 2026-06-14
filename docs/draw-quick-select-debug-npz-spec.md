# Draw Quick Select Debug NPZ 仕様

この文書は Draw Quick Select / Edge Refine の再現用 debug 入力ファイル
`qs_input_*.npz` の仕様をまとめる。

## 目的

`qs_input_*.npz` は、本番 UI で `draw_quick_select.compute_draw_support()` に渡された入力を、
後から同じ解像度・同じ色空間・同じ stroke 条件で replay するためのファイルである。

主な用途:

- 実画像で起きた edge snap の失敗をローカル unit test / script で再現する
- zoom / crop / full-view 経路の違いを切り分ける
- `radius`、`strength`、stroke geometry、`pixel_scale` の影響を固定入力で比較する

## 生成方法

保存処理は [cores/mask2/draw_quick_select.py](/Users/uniuyuni/PythonProjects/platypus/cores/mask2/draw_quick_select.py) の
`_maybe_dump_input()` にある。

有効化方法:

```bash
QS_DUMP_INPUT=edge_refine_debug
```

または、既存の edge refine debug に相乗りする。

```bash
PLATYPUS_DEBUG_EDGE_REFINE=1
PLATYPUS_DEBUG_EDGE_REFINE_DIR=edge_refine_debug
```

出力名は通常 `qs_input_000.npz`, `qs_input_001.npz`, ... になる。
保存上限は `QS_DUMP_INPUT_LIMIT` で指定できる。既定は `500`。

## ファイル形式

形式は NumPy の compressed NPZ で、`np.savez_compressed()` により保存される。

読み込みには stroke object 配列が含まれるため `allow_pickle=True` が必要。

```python
data = np.load(path, allow_pickle=True)
```

注意: `allow_pickle=True` は信頼できる debug ファイルにだけ使う。

## トップレベルフィールド

| key | dtype | shape | 必須 | 意味 |
| --- | --- | --- | --- | --- |
| `guide` | `float32` | `(H, W, 3)` | yes | solver に渡された RGB guide 画像。通常は 0..1 範囲。 |
| `mask` | `float32` | `(H, W)` | yes | solver に渡された FreeDraw mask。0..1 の濃度 mask。 |
| `seed_mask` | `bool` または空配列 | `(H, W)` または `(0,)` | yes | 呼び出し時の seed mask。`None` の場合は空配列で保存される。 |
| `radius` | `float32` scalar | `()` | yes | UI の Quick Radius 値。Draw QS では brush 半径への offset として扱われる。 |
| `strength` | `float32` scalar | `()` | yes | UI の Edge Lock 値。新規 V2 dump では auto からの offset、`strength_mode` が無い旧 dump では solver 内部の 0..100 値として扱う。 |
| `edge_bias` | `float32` scalar | `()` | no | Edge Bias offset。0 は auto のまま、+ は選択側へ寄せる、- は控えめにする。無い旧 dump では `0.0`。 |
| `pixel_scale` | `float32` scalar | `()` | yes | dump 時の draw pixel scale。full-view / scaled replay の補正に使う。 |
| `strokes` | `object` | `(N,)` | yes | stroke dict の object 配列。各要素の仕様は下記。 |
| `strength_mode` | string scalar | `()` | no | `"internal"` または `"offset"`。無い場合は `"internal"` として読む。 |
| `edge_lock_auto` | `float32` scalar | `()` | no | V2 が推定した auto EdgeLock。 |
| `edge_lock_effective` | `float32` scalar | `()` | no | V2 が実際に solver / matte に渡した内部 EdgeLock。 |
| `edge_lock_offset` | `float32` scalar | `()` | no | offset mode 時の UI offset。0 = auto、+ = strict、- = loose。 |

## `strokes` 要素

`strokes` は object 配列で、各要素は次の dict。

| key | dtype / type | shape | 意味 |
| --- | --- | --- | --- |
| `points` | `float32` array | `(P, 2)` | stroke 点列。各点は `(x, y)`。 |
| `size` | `float` | scalar | brush size。直径相当。 |
| `soft` | `float` | scalar | brush softness。0..100。 |
| `is_erasing` | `bool` | scalar | erase stroke なら `True`。 |

座標系:

- `points[:, 0]` が x、`points[:, 1]` が y。
- 原点は画像左上。
- 単位は `guide` / `mask` と同じ pixel 座標。
- `pixel_scale != 1.0` の場合でも、dump 済みの `guide` / `mask` / `points` は solver 入力時点の同一座標系に揃っている。

## `seed_mask` の扱い

保存時に `seed_mask is None` の場合、`seed_mask=np.array([])` として保存される。
replay 時は次のように復元する。

```python
seed_mask = data["seed_mask"] if data["seed_mask"].size else None
```

`None` の場合は、必要に応じて `edge_refine.make_confident_seed(mask)` で再生成する。

## replay 最小例

```python
from types import SimpleNamespace
import numpy as np

from cores.mask2 import draw_quick_select_v3, edge_refine

data = np.load("edge_refine_debug/qs_input_easy.npz", allow_pickle=True)

strokes = []
for raw in data["strokes"]:
    if isinstance(raw, np.ndarray) and raw.shape == ():
        raw = raw.item()
    points = np.asarray(raw.get("points", []), dtype=np.float32)
    strokes.append(SimpleNamespace(
        points=[(float(x), float(y)) for x, y in points[:, :2]],
        size=float(raw.get("size", 1.0)),
        soft=float(raw.get("soft", 100.0)),
        is_erasing=bool(raw.get("is_erasing", False)),
    ))

mask = data["mask"].astype(np.float32)
seed_mask = data["seed_mask"] if data["seed_mask"].size else edge_refine.make_confident_seed(mask)

result = draw_quick_select_v3.compute_draw_support(
    data["guide"].astype(np.float32),
    mask,
    float(data["radius"]),
    float(data["strength"]),
    seed_mask=seed_mask,
    draw_strokes=strokes,
    pixel_scale=float(data["pixel_scale"]) if "pixel_scale" in data.files else 1.0,
)

support = result.support
debug_planes = dict(result.debug_planes)
```

## 現在の代表サンプル

`edge_refine_debug/` に置かれている代表サンプルの実測値。

| file | guide shape | strokes | radius | strength | pixel_scale | 備考 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `qs_input_easy.npz` | `(657, 657, 3)` | 2 | `0.0` | `60.0` | `1.0` | 大きい dab が2つ。 |
| `qs_input_erase.npz` | `(657, 657, 3)` | 2 | `0.0` | `100.0` | `1.0` | add stroke + erase stroke。 |
| `qs_input_lowcontrast.npz` | `(657, 657, 3)` | 1 | `0.0` | `100.0` | `1.0` | 低コントラスト雪/空系。 |
| `qs_input_tree.npz` | `(657, 657, 3)` | 1 | `1.0` | `60.0` | `1.0` | 木・細かい枝。 |
| `qs_input_animal.npz` | `(657, 657, 3)` | 1 | `0.170031` | `0.0` | `1.0` | 動物輪郭。 |
| `qs_input_flower.npz` | `(657, 603, 3)` | 1 | `0.0` | `0.0` | `0.085016` | scaled replay 入力。 |
| `qs_input_roof.npz` | `(657, 657, 3)` | 1 | `1.0` | `-40.38` | `1.0` | 屋根。負 strength offset。 |
| `qs_input_snow_edge.npz` | `(657, 657, 3)` | 1 | `1.0` | `100.0` | `1.0` | 雪の大きい境界。 |

## 後方互換と注意点

- 古い debug ファイルには `pixel_scale` が無い可能性がある。replay 時は `1.0` を既定にする。
- `strokes` は object 配列なので、`np.load(..., allow_pickle=False)` では読めない。
- ファイル名は通常 `.npz` だが、手動でリネームされたファイルでは拡張子が欠けていても zip/npz として読める場合がある。
- `strength_mode` が無い旧 dump の `strength` は内部 Edge Lock として読む。新規 V2 dump では `strength_mode="offset"` として、0 = auto、+ = strict、- = loose と読む。
- `guide` は「元画像全体」ではなく、solver に渡された crop / scaled region の画像である。

## 回帰ハーネスと capture ループ

`qs_input_*.npz` は目視確認だけでなく、**ラベル不要のコーパス回帰テスト**の入力として使う。

- メトリック計算: [cores/mask2/draw_qs_metrics.py](/Users/uniuyuni/PythonProjects/platypus/cores/mask2/draw_qs_metrics.py)
  （`load_dump` / `solve` / `metrics_for_dump` / `compare` / `pair_metrics`）。
- CLI: [scripts/draw_qs_corpus.py](/Users/uniuyuni/PythonProjects/platypus/scripts/draw_qs_corpus.py)。
- golden baseline: `tests/fixtures/draw_qs_baseline.json`（dump 名→stable メトリック）。
  この JSON のキーが「明示コーパス一覧」を兼ね、dump が黙って欠落しないようにする。
- テスト: `tests/test_edge_refine.py` の corpus baseline クラス（`pixi run python -m unittest
  discover -s tests -p test_edge_refine.py` で実行）。

主なメトリック（dump 単体から算出・ラベル不要）:

| metric | 意味 |
| --- | --- |
| `support_hint_ratio` | support/hint 面積比。崩壊/爆発の検出。 |
| `edge_boundary_frac` | support 境界が強エッジ上に乗る割合（吸着できたか）。 |
| `outside_px` / `outside_no_edge_px` | 描画足跡外への成長 / 無エッジ領域での膨張。 |
| `outside_overgrowth_dist` | hint から最も離れた support 画素の距離。 |
| `far_blob_px` | seed core に非連結な support（暴走 grab、≒0 であるべき）。 |
| `deterministic` / `idempotence_iou` | solve の安定性と不動点性。 |

### 失敗ケースを 1 コマンドで固定する

新しく「出鱈目」になる場所を見つけたら、次の手順でコーパスに登録して回帰検出下に置く。

```bash
# 1. dump を有効化してアプリ実行（または PLATYPUS_DEBUG_EDGE_REFINE=1 に相乗り）
QS_DUMP_INPUT=edge_refine_debug pixi run python main.py
# 2. 失敗する stroke を再現 → edge_refine_debug/qs_input_NNN.npz が出る
# 3. 説明的な名前で登録（最新の qs_input_*.npz を取り込み baseline 追記）
pixi run python scripts/draw_qs_corpus.py add <name>
```

### edited expected PNG を確認する

```bash
pixi run python scripts/draw_qs_corpus.py label-report --names <name> --solver v3 --label-dir edge_refine_debug/label_exports
pixi run python scripts/draw_qs_corpus.py label-diff --names <name> --solver v3 --label-dir edge_refine_debug/label_exports --out edge_refine_debug/label_eval
```

`label-diff` は緑=一致、赤=余分、青=不足、黄=予測境界、白=expected 境界の確認画像を出力する。

### つまみの影響を測る（sweep）

```bash
pixi run python scripts/draw_qs_corpus.py sweep QS_COLOR_W 0.4 3.0
```

既定値に対し各値でメトリックがどの dump で動くかを表示する。動かない knob は削除
CANDIDATE、2 dump 以上動かす knob は CORE。`QS_*` は全て `_envf` 経由のためコード改変なしに sweep できる。

## debug plane との関係

NPZ には solver 入力だけが保存される。`result.debug_planes` は replay 後に生成される。

代表的な debug plane:

| plane | 意味 |
| --- | --- |
| `image_edge` | guide から計算した raw edge strength。 |
| `context_edge` | color context などを反映した solver 用 edge strength。 |
| `edge_cost` | min-cut の n-link cost 可視化。白いほど切りにくい。 |
| `color_score` | FG/BG color term を 0..1 表示に変換したもの。0.5 が中立。 |
| `seed_fg` | hard foreground seed。 |
| `seed_bg` | hard background seed。 |
| `prior` | geometric prior を 0..1 表示に変換したもの。0.5 が中立。 |
| `cut_boundary` | min-cut support 境界。 |
| `edge_restore` | selected-side edge rim restore で追加された領域。 |
| `neutral_edge_bias` | Edge Bias auto/offset による neutral edge rim restore 領域。 |
| `edge_bridge` | edge seam bridge で追加された領域。 |
| `interior_fill` | hint 内 hole fill で追加された領域。 |
| `same_side_gap_fill` | V2 の同側ギャップ補完で追加された領域。 |
| `v3_boundary_bias_delta` | V3 Boundary Bias の px シフトで追加/削除された領域。 |
| `v3_erase_support` | V3-native erase で最終的に消えた領域（全 erase footprint のうち support に残らなかった部分）。 |
| `edge_lock_effective` | replay 時の内部 EdgeLock。 |
| `edge_lock_auto` | auto 推定された EdgeLock。 |
| `edge_lock_offset` | auto からの UI offset。 |
| `edge_lock_mode_offset` | offset mode なら 1、internal mode なら 0。 |
| `edge_bias_auto` | unit feature から推定された Edge Bias auto 値。 |
| `edge_bias_effective` | auto と UI offset を合成した Edge Bias 値。 |
| `edge_bias_offset` | UI から渡された Edge Bias offset。 |
| `edge_policy_ridge_threshold` | EdgeLock から解決された ridge thinning threshold。 |
| `edge_policy_restore_threshold` | selected-side rim restore の edge threshold。 |
| `edge_policy_side_threshold` | brush 内部を seed-side / opposite-side に分ける edge threshold。 |
| `edge_policy_outside_keep_threshold` | hint 外 support を残すための edge threshold。 |
| `boundary_bias_px` | Edge Bias の px offset。EdgeLock や色 membership とは分離する。 |
