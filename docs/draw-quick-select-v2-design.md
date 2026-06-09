# Draw Quick Select V2 Design

作成日: 2026-06-09

目的は、現在の Draw Quick Select を小修正で延命することではなく、UI / debug npz /
matte 合成は活かしつつ、境界決定の中核を作り直すこと。

最重要課題は次の 2 つ。

1. サンプルが増えるたびに別のケースが壊れることを止める。
2. 表示倍率 / crop / full-view の違いでマスク形状が変わることを止める。

追加で重視すること。

3. 雪側 / 空側のように、同じ edge を反対側から選んだ 2 つのマスクが重なった時、
   合成後の alpha が seam で途切れず綺麗に繋がること。
4. EdgeLock は固定既定値ではなく auto を中心に、プラス/マイナスで繊細にも大胆にも
   調整できること。
5. 実操作中に待たされない速度で動くこと。

この文書は、そのための V2 実装方針と不変条件を固定する。

----

## 1. V2 の基本姿勢

V2 は「画像全体を理解して領域選択する」機能ではなく、ユーザーが描いた stroke
から **近傍の境界を決める** 機能とする。

したがって solver がやるべきことは次の順番。

1. stroke / mask からユーザー意図の seed と探索範囲を作る。
2. 探索範囲内の候補 edge を評価する。
3. seed 側を保ったまま、もっとも妥当な edge に境界を置く。
4. edge が無い場所では元の brush 境界に戻る。

「面積が増えた/減った」ではなく「境界が妥当な edge に乗ったか」を主品質指標にする。

----

## 2. 絶対に守る不変条件

### 2.1 表示倍率不変

同じ画像座標上の stroke は、次の経路で同じ support を返す必要がある。

- 通常表示
- 拡大表示
- crop preview
- full-view / export

許容差は、最終 support を同じ画像座標へ戻した時の境界 1px 程度。

実装上のルール:

- solver 入力は必ず image/source 座標系へ正規化する。
- preview texture px を solver の意味単位にしない。
- `pixel_scale` は「既に crop/scale された dump の再現情報」であり、edge 強度を場当たり的に補正する値にしない。
- solver は canonical resolution で実行する。canonical へ downscale/upscale するときは、stroke size、radius、mask、guide を同じ scale で変換する。
- binary support の upscale/downscale は最近傍、edge/guide は area/linear を使い分ける。

検証:

- 各 npz を 1x / 2x / 0.5x へリサイズして replay し、元サイズへ戻した support IoU を測る。
- 失敗時は metrics だけでなく contact sheet を保存する。

### 2.2 Stroke 意図不変

過去 stroke の意味が、後から描いた stroke によって勝手に反転してはいけない。

実装上のルール:

- stroke は単なる「最終 mask を作った履歴」ではなく、seed intent として扱う。
- 1 stroke ごとに seed 側を決める。
- 複数 stroke を合成するとき、別 stroke の seed 側を壊さない。
- add stroke と erase stroke は同じ solver profile に入れない。

特に `easy` のような複数 dab では、stroke ごとの seed 側と最終 mask 全体の接続性の両方が必要。
brush 円だけで solve すると探索範囲が狭すぎる。一方、全 stroke を一つの seed として混ぜると
雪側/空側の意図が混線する。

### 2.3 Edge 優先、ただし featureless では元境界

V2 は「エッジを無視して mask を保持する」方向に逃げない。
ただし、edge が無い場所で半径いっぱい膨らむのも禁止。

実装上のルール:

- cut の安い場所は edge ridge 上に限定する。
- edge confidence が低い場所では、元 brush 境界の prior が勝つ。
- texture edge と object boundary edge を区別するため、edge 単独ではなく color separability / seed side consistency を見る。
- edge が強くても seed 側を分断する内側 texture edge は候補から落とす。

### 2.4 Add と Erase を混ぜない

右クリック消しゴムは add の逆操作ではあるが、同じパラメータで解くと調整不能になる。

実装上のルール:

- Add solver と Erase solver は別 profile。
- Add は「seed 側を edge まで広げる/削る」。
- Erase は「削る stroke が示した側を edge で切って subtraction mask を作る」。
- 同じ EdgeLock UI 値でも、内部 auto profile は add / erase で別。
- V2 の最初の実装では add-only を完成させる。erase は後段で入れる。

### 2.5 反対側マスクの seam を閉じる

雪側 mask と空側 mask のように、同じ画像 edge を反対側から選ぶことがある。
このとき、それぞれ単独では正しく見えても、2 つを足すと edge 上に 1px の穴や
alpha gap が残ると実用上失敗。

実装上のルール:

- 同じ edge の両側から作られた support は、matte 合成後に seam 上で alpha が足りる必要がある。
- pair 検証では `refined_a + refined_b >= threshold` を shared edge band 上で確認する。
- 片側だけを過度に細くする edge alpha は避ける。必要なら pair-aware な 1px seam fill を入れる。
- pair-aware 処理は mask を太らせる処理ではなく、edge ridge 上の alpha gap を閉じる処理に限定する。

検証:

- `snow_side` / `sky_side` のような pair を corpus に登録する。
- `pair_union_gap_px` と contact sheet で、両側の alpha 合計を可視化する。

### 2.6 速度を設計制約にする

V2 は綺麗でも遅ければ使えない。
特にブラシ描画中はリアルタイム更新されるため、速度は後回しにしない。

実装上のルール:

- stroke 中の preview は ROI / dirty region に限定する。
- edge field は guide / scale / ROI 単位で cache する。
- 同じ stroke drag 中に毎回 full solve しない。必要なら低解像度 preview と確定 solve を分ける。
- graph cut のノード数に上限を置く。超えた場合は tiled solve / downscale / fallback を明示する。
- debug plane 生成は通常時 off。npz/debug 有効時のみ詳細 plane を作る。

目標速度:

- 通常 preview: 100ms 台を目標。
- 重い巨大ブラシ / full-view: 500ms 未満を目標。
- それを超える場合は UI を固めず、段階的 preview を返す。

----

## 3. V2 Pipeline

### 3.1 入出力

V2 の public API は現行と同じにする。

```python
compute_draw_support_v2(
    guide,
    mask,
    radius,
    strength,
    seed_mask=None,
    draw_strokes=None,
    pixel_scale=1.0,
) -> DrawSupportResult
```

既存 `edge_refine.refine_mask_edge_aware()` からは切り替えだけで使える形にする。

### 3.2 Stage A: Normalize

入力を solver 内部表現へ正規化する。

- `guide`: float32 RGB, 0..1, image/source 座標
- `mask`: float32 0..1, image/source 座標
- `strokes`: points / size を image/source 座標へ統一
- `brush_radius`: stroke size / 2
- `ui_radius`: brush radius に対する探索範囲 offset

探索半径:

```text
inner_reach = max(brush_radius + min(ui_radius, 0), min_band)
outer_reach = max(brush_radius + max(ui_radius, 0), min_band)
```

UI radius は「ブラシサイズそのもの」ではなく、探索範囲を小さく/大きくする offset。

### 3.3 Stage B: Canonical Solve Space

表示倍率不変のため、solve は canonical pixel pitch で行う。

方針:

- 探索帯の幅が大きい場合は downscale する。
- 小さい場合は native で解く。
- 目標は、探索帯幅がだいたい一定 px になること。

例:

```text
canonical_band_px = 96 or 128
work_scale = min(1.0, canonical_band_px / actual_band_px)
```

ただし `work_scale` の選択は全 stroke / component で安定させる。
stroke ごとに scale が変わると合成境界がズレる。

### 3.4 Stage C: Stroke Intention Units

solve 単位は「brush mask」ではなく「stroke が触った connected hint component + stroke seed」。

1. 最終 mask から connected component を作る。
2. 各 add stroke が触れている component を列挙する。
3. component ごとに、その stroke の centerline を FG seed とする。
4. 同じ component に複数 stroke が触れる場合は、seed side を stroke ごとに保持する。

この段階ではまだ support を OR しない。
まず stroke unit ごとの candidate boundary を出す。

### 3.5 Stage D: Edge Field

edge map は単純な Sobel/Canny の正規化ではなく、次の複合 confidence とする。

```text
edge_confidence =
    ridge_strength
  * scale_stability
  * color_separation_gate
  * seed_side_consistency
```

要素:

- `ridge_strength`: edge ridge の強さ。blurred edge をそのまま使わず ridge 化する。
- `scale_stability`: 1x / half-res でも存在する edge を優先し、細かい texture edge を弱める。
- `color_separation_gate`: edge の両側で LAB/RGB が seed foreground / local background と分かれるか。
- `seed_side_consistency`: edge を越えた先が stroke seed 側と矛盾しないか。

雪/雲/枝のようなケースでは、raw edge の強さより `color_separation_gate` と
`scale_stability` を重視する。

### 3.6 Stage E: Boundary Solver

候補は 2D graph cut / min-cut。

ノード:

- 探索帯内の pixels

n-link:

- edge ridge 上ほど切りやすい
- featureless では短く滑らかな cut を好む

t-link:

- stroke centerline / confident interior は FG
- outer band rim は BG
- erase unit では erase stroke side を BG
- 元 mask 境界 prior は弱く入れる

重要:

- 元 mask を強く保持しすぎると edge を無視する。
- edge を強くしすぎると texture に穴が空く。
- したがって prior と edge の重みは固定値ではなく、unit features から auto 推定する。

### 3.7 Stage F: Compose Stroke Units

Add-only:

- 各 unit の support を seed-side preserving union で合成する。
- 同じ connected component 内で unit が矛盾する場合、edge confidence の高い boundary を優先する。
- featureless 領域では元 mask を維持する。

Erase:

- Add support を先に作る。
- Erase stroke から subtraction support を別 profile で作る。
- `final = add_support & ~erase_support`。
- erase の edge lock は add と共有しない。

----

## 4. EdgeLock UI の設計

現行の `0..100` と `0=auto offset` が混ざると replay が壊れる。
V2 では保存値の意味を明示する。

推奨:

- backend internal: `edge_lock_internal` 0..100
  - 0 = strict
  - 100 = loose
- UI V2: `edge_lock_offset`
  - 0 = auto
  - + = stricter
  - - = looser
- debug npz: `strength_mode`
  - `"internal"` or `"offset"`

移行中は既存 npz を `"internal"` とみなす。

### 4.1 UI の感触

EdgeLock は「60 が既定」のような絶対値スライダーにしない。
ユーザーが触る値は auto からの offset にする。

```text
EdgeLock Offset = 0     auto
EdgeLock Offset > 0     stricter / 強い edge だけ見る
EdgeLock Offset < 0     looser / 弱い edge も見る
```

内部では次のように扱う。

```text
internal_lock = clamp(auto_lock - ui_offset, 0, 100)
```

ここで internal の意味は従来通り。

```text
0   = strict
100 = loose
```

UI の調整幅は、細かくも大胆にも動かせる必要がある。

- slider range: `-100 .. +100`
- step: `0.1` 以下
- option: Shift/修飾キーや入力欄で大きく動かせる
- debug plane: `edge_lock_auto`, `edge_lock_effective`, `edge_lock_offset`

既存 npz との互換:

- `strength_mode` が無い npz は旧 internal 値として読む。
- 新規 dump は offset と auto/effective の両方を保存する。

----

## 5. 回帰設計

V2 は数値だけで判定しない。
毎回 contact sheet を作る。

### 5.1 必須サンプル分類

| group | 目的 |
| --- | --- |
| easy | 巨大 brush / 複数 dab / 明瞭 edge |
| snow_edge | 雪側/空側の seam と reach |
| tree | 枝と空、細い edge |
| lowcontrast | 低コントラスト境界 |
| erase | erase は add と別 profile で扱えるか |
| radius | radius を増やしても featureless に膨らまないか |
| zoom | 表示倍率で形が変わらないか |

### 5.2 Metrics

既存 metrics に加えて V2 では次を追加する。

- `zoom_iou_2x`
- `zoom_iou_0_5x`
- `edge_contact_on_cut`
- `featureless_growth_px`
- `seed_side_leak_px`
- `texture_hole_px`
- `pair_union_gap_px`
- `runtime_ms_preview`
- `runtime_ms_full`
- `graph_nodes`
- `graph_edges`

### 5.3 Contact Sheet

各サンプルで次を横並びにする。

1. normalized guide
2. raw mask
3. support
4. kept/removed/grown diff
5. edge confidence
6. cut boundary over guide
7. zoom 1x vs 2x diff

この contact sheet が V2 の主要な合否判断。

----

## 6. 実装順序

### Phase 1: Harness first

- V2 replay script を作る。
- 現行実装と V2 の output を同じ contact sheet に並べる。
- zoom invariance test を先に入れる。
- pair mask seam test を先に入れる。
- runtime / graph size を必ず記録する。

この段階では V2 が空実装でもよい。

### Phase 2: Add-only MVP

- erase は無効。
- stroke touched connected component + centerline seed。
- scale-normalized edge field。
- min-cut で support を返す。

目標:

- `easy` で edge を無視しない。
- `snow_edge` で雪側が足りなさすぎない。
- `tree` で枝 edge を最低限拾う。
- zoom 1x / 2x が一致する。

### Phase 3: Auto Edge Model

- unit features から internal edge lock を推定する。
- UI offset は最後に適用する。
- `strength_mode` を npz に保存する。

### Phase 4: Erase

- erase profile を別に実装する。
- add support と erase support の引き算にする。
- 同じ EdgeLock UI 値でも内部推定は add / erase 別。

### Phase 5: Switch over

- `QS_DRAW_V2=1` で切り替え。
- 全サンプルの contact sheet を確認。
- V2 が上回ったら default を V2 にする。

----

## 7. 今やらないこと

- 既存 V1 にさらに erase / auto / zoom 補正を継ぎ足さない。
- support ratio だけを改善目標にしない。
- 1 サンプル専用の magic threshold を入れない。
- UI を先に変えない。
- erase を add-only が安定する前に戻さない。

----

## 8. 成功条件

V2 が成功したと言える条件:

1. 代表 npz で contact sheet が目視で破綻しない。
2. 1x / 2x / 0.5x replay の support がほぼ一致する。
3. `easy` のような明瞭 edge で edge を無視しない。
4. `snow_edge` の雪側/空側が pair で繋がる。
5. radius を大きくしても featureless に膨らまない。
6. erase を入れても add-only の品質が落ちない。
7. EdgeLock Offset で strict / loose の差が目視でも数値でも分かる。
8. preview が実操作に耐える速度で動く。
