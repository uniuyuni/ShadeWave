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
| `strength` | `float32` scalar | `()` | yes | UI の EdgeLock / strength 値。現在の実装では auto strength への offset として解釈される。 |
| `pixel_scale` | `float32` scalar | `()` | yes | dump 時の draw pixel scale。full-view / scaled replay の補正に使う。 |
| `strokes` | `object` | `(N,)` | yes | stroke dict の object 配列。各要素の仕様は下記。 |

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

from cores.mask2 import draw_quick_select, edge_refine

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

result = draw_quick_select.compute_draw_support(
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
- `strength` は保存時の UI 値であり、`DrawSupportResult.edge_lock` に入る内部 effective EdgeLock とは一致しない場合がある。
- `guide` は「元画像全体」ではなく、solver に渡された crop / scaled region の画像である。

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
| `edge_bridge` | edge seam bridge で追加された領域。 |
| `interior_fill` | hint 内 hole fill で追加された領域。 |
| `edge_lock_effective` | replay 時の内部 EdgeLock。 |
| `edge_lock_auto` | auto 推定された EdgeLock。 |

