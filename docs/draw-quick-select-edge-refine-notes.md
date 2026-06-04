# Draw Quick Select / Edge Snap 試行錯誤と今後の設計メモ

作成日: 2026-06-04

Draw 系マスクに「Photoshop のクイック選択に近い、エッジへ吸着する補正」を入れるための試行錯誤まとめ。
結論として、機能自体は有効だが、現在試している **Random Walker 主体の実装は本用途に合っていない**。
次は **最終マスクの周辺帯に対する 2D graph cut / min-cut 型の境界決定** に切り替える方針が妥当。

----

## 1. 目的

ユーザーが FreeDraw / PolyLine などでざっくり描いたマスクを、近傍の画像エッジへ自然に吸着させたい。

代表的な想定:

- 森の写真で空を TextTarget / Segment / Draw したとき、枝や木の隙間のエッジまで補完したい
- Draw で境界付近をなぞったとき、線の内側または近傍の強い境界へマスクが寄ってほしい
- エッジを超えて広がらない
- エッジが無い場所では勝手に半径いっぱい膨らまない
- 拡大表示 / crop 表示でも通常表示と同じ結果になる
- 消しゴムで消した部分は維持される

----

## 2. 対象範囲

### 適用する

- FreeDrawMask
- PolyLine 系 Draw マスク
- AI 系マスクの補完処理

### 適用しない

- CircularGradientMask
- Linear Gradient / Line 系
- FullMask
- パラメトリックマスク全般

理由:

パラメトリックマスクは「どこを基準にエッジへ寄せるか」が曖昧。
中心点基準にすると意図と逆方向へ広がりやすく、AI セグメント並みの領域理解が必要になる。
Draw 系はユーザーがストロークで意図を示しているため、補正対象として最も自然。

----

## 3. これまで試した方式

### 3.1 Guided Filter / 既存 refine 方向

当初は、現在のマスクを guided filter / エッジ判定で広げる方向を試した。

観察:

- AI マスク補完には比較的向く
- Draw では「太った線」になるだけで、エッジに張り付かない
- 元マスクの内側にあるエッジへ寄せる動作が弱い
- エッジを超えない保証が弱く、似た色の領域へ漏れる

結論:

AI マスクの補完には残せるが、Draw のクイック選択動作の主役にはしない。

### 3.2 GrabCut / Quick Select 寄せ

Draw の線を seed にして GrabCut 的な領域選択を試した。

観察:

- 画像によっては領域らしく広がる
- ただし Draw の「線を境界に寄せる」意図とは違い、領域推定になりすぎる
- 背景/前景 seed が曖昧だと反転、全選択、無変化が起きる
- 半径や強度の意味がユーザーに伝わりにくい

結論:

Draw の境界補正としては制御しづらい。

### 3.3 法線方向探索 / stroke boundary snap

ストローク中心線から法線方向にサンプリングし、強いエッジへ寄せる案を検討した。

利点:

- 直線的な境界ではわかりやすい
- 元ブラシの片側だけを境界へ寄せる設計がしやすい
- 始点/終点の cap も扱いやすい

問題:

- 凹境界で法線が交差する
- 入り組んだ境界で近傍の別エッジへ吸われる
- ストロークがエッジを跨いだ場合、内側/外側の判定が不安定
- 消しゴムで線が分断された最終マスクに対して扱いが複雑

結論:

単独の主役にはしない。
ただし、Graph Cut の seed / 初期境界 / デバッグ可視化には使える。

### 3.4 Random Walker

現在の試行では、Draw の最終マスクを hint とし、stroke center を foreground seed、消しゴムや外周を background seed として Random Walker を解いた。

実装上の主な追加:

- `cores/mask2/edge_refine.py`
- Draw 系のみ Random Walker path
- `bg_seed`, `probability` の debug panel
- zoom/crop のセルフテスト
- 消しゴム seed の反映

観察:

- 小さい合成テストでは通る
- 実画像では「ほぼ何も起きない」ケースが多い
- 拡大表示では通常表示と違う結果になることがある
- `target_edge` が短すぎると、補正先が無くなる
- `target_edge` を広げると、木や雪の細かいエッジまで拾いすぎる
- `probability` が元マスクとほぼ同じになり、境界選択として機能しないケースがある
- 縮小画像では吸着せず、拡大画像ではマダラになる

代表的な debug 画像:

- `edge_refine_debug/0013_FreeDrawMask_edge_snap_1780388123940_mosaic.png`
  - `image_edge` は出ているが `target_edge` が短い
  - `support/refined` はほぼ元マスク
- `edge_refine_debug/0017_FreeDrawMask_edge_snap_1780388158359_mosaic.png`
  - `target_edge` は出ている
  - しかし `probability/support/refined` がほぼ元マスクと同じ
  - 拡大表示時に通常表示と別物になる

結論:

Random Walker は「seed からの確率拡散」であり、今回ほしい「帯領域内でエッジに沿った閉じた境界を選ぶ」用途とはズレている。
特に Draw の最終マスクはすでに形を持っているため、Random Walker で再推定すると、元マスク保持とエッジ吸着のバランスが不安定になる。

----

## 4. 現時点の根本課題

### 4.1 必要なのは確率場ではなく境界決定

Draw 補正で本当に決めたいのは「どの領域が FG っぽいか」ではなく、
**元マスク周辺のどこで切るか**。

Random Walker は前者に寄る。
必要なのは後者。

### 4.2 エッジは複雑に絡む

雪、木、雲、枝のような画像では、エッジ候補が大量に出る。

単純に edge map を強くすると:

- 木の細かいエッジを拾う
- マスク内に穴が空く
- マダラになる

弱くすると:

- 本命の雲境界まで拾えない
- 何も起きない

### 4.3 拡大表示と通常表示の一致

crop 表示では texture 上の線幅、探索半径、背景 seed、候補領域が変わりやすい。

必要な条件:

- すべて original/image 座標基準で意味が一致すること
- texture px 固定の `+6px` のような値を避けること
- crop 端を強い BG として扱いすぎないこと
- full render と crop render の結果が同じ領域になること

### 4.4 消しゴム

Draw 系は右クリック / eraser により、線の一部または全部を削除できる。

したがって補正は stroke history だけでなく、**最終マスク画像** を基準にする必要がある。
ただし stroke history は foreground / background seed の補助には使える。

----

## 5. 次の設計方針

Random Walker を主役から外し、2D の探索帯に対する graph cut / min-cut 型の境界決定へ切り替える。

### 5.1 基本方針

1. 最終 Draw マスクを `M` とする
2. `M` を半径分 dilate / erode して探索帯 `Band` を作る
3. `Band` 内で foreground / background を分ける cut を求める
4. 画像エッジが強い場所ほど cut が通りやすい
5. 元マスク境界は「弱いデフォルト境界」として扱う
6. エッジが無い場所では元マスク境界に戻る
7. 消しゴム領域は BG seed として維持

### 5.2 入力

| 名前 | 内容 |
|---|---|
| `image_rgb` | edge cost の元画像 |
| `mask` | Draw 後の最終マスク |
| `draw_strokes` | add / erase stroke 履歴 |
| `radius` | 探索帯の幅 |
| `strength` | エッジ優先度 |

### 5.3 seed 設計

Foreground seed:

- 最終マスク `M` の芯
- add stroke の中心線
- `M` のうち境界から十分内側の領域

Background seed:

- `dilate(M, radius)` の外側
- erase stroke のブラシ領域
- `M` から明確に離れた外周

重要:

- crop 端そのものを強い BG seed にしすぎない
- seed は texture 表示ではなく image/crop 一貫座標で作る

### 5.4 graph の考え方

各 pixel を node とし、隣接 pixel 間に n-link、FG/BG seed との接続に t-link を置く。

n-link:

- 画像エッジが強いほど切れやすい
- 画像が滑らかなほど切れにくい
- 元マスク境界にも弱い切れやすさを与える

t-link:

- FG seed は foreground 側へ固定
- BG seed は background 側へ固定
- seed 以外は弱い prior

### 5.5 cost の直感

エッジが無い場所:

- 元マスク境界に近いところで切る
- 半径いっぱいには膨らまない

強いエッジが近い場所:

- 元マスク境界より画像エッジを優先して切る
- エッジ上の pixel もマスクに含める方向へ調整する

ノイズエッジが多い場所:

- 小さい孤立エッジは閉じた cut として安定しにくいため選ばれにくい
- 必要なら edge map を connected component / hysteresis で整理する

----

## 6. 実装計画

### Phase 1: 現 Random Walker 実装を実験扱いにする

- `edge_refine.py` 内の Random Walker path は残してもよいが、Draw の主経路から外す
- debug panel は流用する
- `probability` panel は graph cut では `cost` / `cut` panel に置き換える

### Phase 2: Graph Cut backend を追加

候補:

- `PyMaxflow` が使えるなら最も直接的
- 依存を増やしたくない場合は OpenCV `grabCut` を mask 初期化で再利用する案もあるが、今回の目的にはやや不安
- SciPy sparse で s-t cut を自前実装するのは重いので避けたい

暫定 API:

```python
def _draw_graph_cut_support(
        guide,
        mask,
        radius,
        strength,
        seed_mask=None,
        draw_strokes=None):
    ...
    return seed, candidate, support, extra_debug_planes
```

### Phase 3: debug panel

最低限出すもの:

- `guide`
- `mask`
- `seed_fg`
- `seed_bg`
- `candidate_band`
- `edge_cost`
- `mask_boundary_prior`
- `cut_boundary`
- `support`
- `mask_change`
- `overlay`

### Phase 4: セルフテスト

合成テスト:

- 単純な縦境界
- 雲/空の境界
- 凹境界
- 細かい木/枝ノイズ
- 消しゴムで分断
- 全消し
- uniform image で膨らまない
- radius 小/大の差
- zoom crop と full crop の一致

実画像寄せテスト:

- `edge_refine_debug` の雪/雲画像を fixture 化する
- debug mosaic の panel からではなく、元画像相当の小さい synthetic scene を用意する
- 期待値は pixel 完全一致ではなく、領域指標で判定する

----

## 7. 受け入れ基準

最低限:

- Draw でエッジ近辺をなぞると、マスク境界が見える程度に変化する
- エッジが無い場所では元マスクから大きく膨らまない
- 強い雲/空境界に沿う
- 木や雪の細かい内部エッジでマダラにならない
- 消しゴムで消した部分が復活しない
- 拡大表示と通常表示で同じ範囲になる
- `pixi run python -m unittest discover -s tests -p test_edge_refine.py` が通る

望ましい:

- 半径の意味が「探索帯の幅」として直感的
- 強度の意味が「画像エッジをどれだけ優先するか」として直感的
- debug mosaic を見れば、どの seed / cost / cut が原因か追える

----

## 8. 現状コードへの扱い

現状の `cores/mask2/edge_refine.py` は untracked で、Random Walker 実験が多く含まれている。

推奨:

1. 現在のファイルを別名で退避するか、git で実験コミットにする
2. Draw 用 Random Walker path は主経路から外す
3. AI 系補完に使える部品だけ残す
4. Graph Cut path を新規に実装する

Random Walker 実験から流用できるもの:

- debug dump の枠組み
- `image_edge`
- `hint_boundary`
- `seed_fg` / `seed_bg` 作成の一部
- zoom crop テスト
- 消しゴムテスト

捨てるべきもの:

- `probability` を最終 mask 決定に直接使う設計
- `target_edge` を後段で継ぎ足す設計
- texture px 固定の距離調整

----

## 9. 判断メモ

今回の失敗は無駄ではない。

わかったこと:

- Draw 補正は「領域推定」ではなく「境界決定」
- 法線 1D 探索は凹境界で弱い
- Random Walker は元マスク保持とエッジ吸着の両立が難しい
- debug panel は必須
- zoom/crop 一致は最初からテストに入れるべき

次の一手:

**2D 探索帯 + graph cut / min-cut で閉じた境界を決める。**

ここに切り替えるのが、現時点で一番筋が良い。

----

## 10. 2026-06-04 実装結果

Draw Quick Select の通常経路を Random Walker から外し、`_draw_grabcut_band_support` を新しい主経路にした。

採用した構成:

- 最終 Draw マスクから `hint` を作る
- `radius` を探索帯 `band` の幅として使う
- 元マスクの芯 / add stroke 中心線を hard FG seed にする
- eraser stroke と band 外周を hard BG seed にする
- OpenCV `grabCut` は band 内の補助的な領域推定として使う
- 実際の境界決定は、target edge が属する image edge 成分を壁にした band 内連結選択で行う
- edge barrier 側で、選択側に隣接する薄い境界面の pixel を復元する
- 汎用の post-solve target edge 復元は小さい edge 断片だけに限定する
- 雲/雪のような大量エッジでは追加復元を抑え、外側漏れを防ぐ
- 長い stroke に対して target edge が短い断片しか出ない場合は、その edge は切断に使わず `raw_target_edge` として debug にだけ残す
- stroke がエッジをうっかり跨いだ場合は、target edge が連続した境界として信用できる時だけ、跨いだ FG seed を BG seed へ回して外側を落とす。雪/木の細かい断片 edge では seed を切らず、stroke 中心線まわりだけを細く保護して内部穴あきを避ける。マスク内側全体を保護すると、太いブラシで本命エッジの遠い側を削れなくなる
- target edge は stroke 中心線の近傍だけでなく、マスク境界面に近い連続 edge も残す。太いブラシでは本命境界が中心線より外周寄りにあるため、中心線距離だけで絞ると雲/木の境界を落として内部テクスチャだけが残る
- target edge が seed を覆う場合は、free 領域の色を FG seed と比較して選択側を決める
- zoom crop で巨大ブラシが crop 端に clipped blob / wide stroke として見える場合は、外側 grow を禁止しつつ stroke 中心線近傍へ制限して、拡大時だけ巨大な塗り領域になる挙動を避ける
- FreeDraw の zoom/crop 表示では、crop-local の clipped stroke を直接 refine せず、フル画像座標系の余白付き小領域で Draw Quick Select を再計算してから現在表示へ切り出す。これにより通常表示・拡大表示・headless/export の基準を寄せる
- この小領域は表示 crop だけでなく、表示 crop に触れている stroke の全体 bbox も含める。拡大表示で stroke の一部だけが切られて巨大な filled component になるのを避ける
- 上記の余白付き領域が大きすぎる場合も従来の crop-local 経路へは戻さず、領域画像・stroke 座標・brush サイズ・半径・出力 crop 座標を同じ scale で内部縮小して処理する。Quick Select 選択時に元画像全体を同期処理して UI が戻らなくなるのを避けつつ、拡大時だけ座標系が変わる問題を防ぐ
- FreeDraw の full-coordinate ROI は、実画像配列の左上座標ではなく TCG と同じ正方形パディング座標で作る。ROI から実画像を切り出す時だけパディングを差し引く
- UI の通常 full 表示では、Full 経路でも current crop/preview 画像を source にして local 経路と同じ解像度で解く。export では crop と original が同じなので full 解像度になる
- `FreeDrawMaskFull` の debug mosaic は内部計算領域を表示するため、拡大表示そのものにはならない。現在表示へ切り出した後の確認用に `FreeDrawMaskFullCrop` debug mosaic も出す

OpenCV `grabCut` 単独では不十分だった点:

- 同色のまま近傍エッジまで伸ばす圧力が弱い
- hard BG shell に同色背景が混ざると、PR_FG が背景へ縮む
- Draw の「境界へ寄せる」意図は color GMM だけでは表現しにくい

そのため、今回の実装では `grabCut` を backend 名に含めつつ、主決定は **Band + edge barrier + connected side selection** に寄せている。
将来 PyMaxflow などを導入する場合は、この barrier / seed / band 設計をそのまま n-link/t-link の本物の min-cut へ移せる。

追加・維持した debug panel:

- `image_edge`
- `hint_boundary`
- `raw_target_edge`
- `target_edge`
- `hard_fg`
- `hard_bg`
- `band`
- `probable_fg`
- `probable_bg`
- `grabcut_result`

通過確認:

```bash
pixi run python -m unittest discover -s tests -p test_edge_refine.py
```

結果:

```text
Ran 39 tests
OK
```
