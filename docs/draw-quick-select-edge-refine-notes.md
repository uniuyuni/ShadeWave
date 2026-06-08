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

----

## 11. 2026-06-05 現状スナップショット

### 11.1 現在のユーザー観察

既存の雪山/雲境界の一部ではかなり良い結果が出ている。
一方で、同じ画像内でも今まで試していなかった場所を FreeDraw で描くと、結果が出鱈目になるケースがある。

特に目立つ症状:

- `radius` が低い時は、細部は甘いが大枠は比較的良い
- `radius` を上げると急に破綻する
- 破綻時は、範囲大幅超過、二重線、内側/外側の混在が同時に出る
- 以前よく見ていた雲境界の U 字 stroke だけでは問題を代表できていない
- デバッグ画像が出ないことがあり、原因追跡が難しくなる

この観察から、現状は「既知の一点を通すチューニング」になりかけており、Quick Select としての一般性はまだ足りない。

### 11.2 現在入っている主な変更

対象ファイル:

- `cores/mask2/edge_refine.py`
- `tests/test_edge_refine.py`
- `effects.py`
- `main.kv`
- `widgets/mask_editor2.py`

主な内容:

- Draw Quick Select の結果に自然な edge matte を入れる処理を追加
- `geometry_snap` を debug panel に追加
- 大きい `radius` で support が異常膨張した場合、幾何スナップ結果を guard として使う処理を追加
- 大きい `radius` の時だけ、`target_edge` を `geometry_snap` 境界近傍へ制限する処理を追加
- `mask2_edge_refine_radius` のデフォルトを `60` から `0` に変更
- FreeDraw の描画開始が preview 領域外でも反応するバグを修正
- debug dump の既定上限を `20` から `80` に変更し、上限到達時に warning を出すように変更

直近の確認:

```bash
pixi run python -m unittest discover -s tests -p test_edge_refine.py
```

結果:

```text
Ran 43 tests
OK
```

ただし、この OK は「現在のテストケースを満たす」という意味であり、実使用全体の品質保証にはまだ不足している。

### 11.3 実画像 fixture での直近数値

`tests/fixtures/edge_refine_snow_600.png` と既存 U 字 stroke に対する計測では、直近の guard により大 radius の巨大化は抑えられている。

```text
radius=0  support=5729  refsum=4762.6  lower=0.0
radius=50 support=8188  refsum=7017.5  lower=0.0
radius=83 support=8205  refsum=7042.3  lower=0.0
radius=93 support=8205  refsum=7042.3  lower=0.0
```

この fixture では `radius=83/93` で破綻しなくなった。
しかし、ユーザーが別の場所を描いた時に出鱈目になるため、この fixture は十分な代表性を持っていない。

### 11.4 現在の強い疑い

`radius` が大きくなると、目的のエッジだけでなく、近くの別エッジも `target_edge` / `barrier` 候補に入る。
その結果、以下が起きる可能性が高い。

- 本命ではない texture edge が barrier になる
- barrier が二重または分断される
- connected side selection が意図しない側を選ぶ
- `support` が遠くの連結領域まで伸びる
- 幾何スナップ guard が効く場所と効かない場所の差が大きくなる

つまり、破綻の主因は edge map そのものよりも、**edge 候補の選別と side selection** にある可能性が高い。

### 11.5 `radius` の設計上の問題

現状の `radius` は複数の意味を同時に持っている。

- ROI をどこまで広げるか
- `candidate` / `band` をどこまで広げるか
- `target_edge` をどこまで探すか
- BG shell をどこに置くか
- geometry snap / guard を発火させるか

このため、ユーザーが「少し遠いエッジも探したい」と思って `radius` を上げると、実際には問題設定そのものが変わる。

改善方針:

- `radius` を「ストローク境界から許す補正幅」に寄せる
- ROI padding は別パラメータまたは内部固定値にする
- `candidate` の最大面積や最大距離には brush 幅ベースの安全上限を置く
- `target_edge` 探索距離と support growth 距離を分ける

### 11.6 現在のテスト不足

いまのテストは、既知の U 字 stroke / 雲境界にかなり寄っている。
そのため、次のようなケースを十分に覆えていない。

- 同じ画像内の別位置
- stroke が本命エッジに対して内側すぎる / 外側すぎるケース
- 近くに別の強いエッジがあるケース
- 直線、S 字、急カーブ、短い stroke
- ブラシが太く、中心線と本命境界がかなり離れるケース
- 本命境界が弱く、別エッジが強いケース
- 拡大表示と export/headless の一致

特に、今回の「別の場所を描くと出鱈目」は、テストが代表点に偏っていることを示している。

### 11.7 次に作るべきデバッグ情報

現在の mosaic だけでは、どの edge component が選ばれたかが見えにくい。
次の debug panel / log を追加したい。

- `target_edge_labels`: `target_edge` の connected component を色分け
- `barrier_edge`: 実際に barrier として使った edge
- `selected_side`: connected side selection で選ばれた free 領域
- `geometry_snap`: 既に追加済み。target/support と比較する
- `support_overgrowth`: geometry/hint から遠く伸びた部分
- 数値ログ:
  - hint area
  - candidate area
  - target edge pixels
  - target edge component count
  - support area
  - support/hint ratio
  - geometry/support ratio

これがないと、「edge 検出が悪い」のか「選択側が悪い」のかが毎回目視推測になる。

### 11.8 次の進め方

短期:

1. ユーザーが出鱈目になった位置を再現できる fixture / stroke として固定する
2. そのケースの debug mosaic を必ず出す
3. `target_edge` と `barrier_edge` を分けて可視化する
4. `radius` を上げた時に、どの edge component が増えるかを見る
5. 既知の U 字ケースを壊さず、別位置ケースを改善する

中期:

1. `radius` の役割を分離する
2. `target_edge` を「近い edge」ではなく「stroke 境界として一貫した edge」として選ぶ
3. connected side selection を、単純な連結ではなく graph cut / min-cut へ移す
4. natural matte は support 決定後の別段階として維持する

保留:

- いまの guard は応急処置としては有効だが、Quick Select の一般解ではない
- 別位置で出鱈目になるなら、guard の閾値調整だけで追いかけるのは危険
- 次は「失敗ケースをテスト化してから直す」順番にする

### 11.9 判断

現時点で一番重要なのは、さらにパラメータを微調整することではない。

優先すべきこと:

1. 失敗ケースを固定する
2. edge 候補の選別過程を見える化する
3. `radius` が問題設定を変えすぎる構造を直す

現在のコードは既知 fixture ではかなり良く見えるが、一般化できていない。
したがって、次の作業は「品質改善」ではなく「再現性と観測性の改善」として扱うべき。

----

## 12. 2026-06-05 作り直し実装（Band + 2D min-cut）

第 5 / 第 9 節の方針どおり、Draw Quick Select を「領域推定」から
**「探索帯 Band に対する 2D min-cut で閉じた境界を 1 本決める」**へ全面的に作り直した。
新モジュール `cores/mask2/draw_quick_select.py`（エントリ `compute_draw_support`）を新設し、
`edge_refine.py` の `STRATEGY_DRAW` 経路をそこへ即切替した。
旧 `_draw_grabcut_band_support`（grabCut + target_edge + connected-side）系は
環境変数 `PLATYPUS_DRAW_QS_LEGACY=1` で 1 サイクルだけ残してある。

### 12.1 採用した構成

- **ソルバ**: `scipy.sparse.csgraph.maximum_flow`（Dinic）で真の min-cut。
  新しい compiled 依存は足していない（PyMaxflow 不採用）。
  連結成分ごと・padded ROI 内・**Band 画素のみをノード**にするので速い
  （600×400 fixture で radius 0–200 を通して 110–230ms）。
- **ノード**: Band 画素 ∪ {source, sink}。inner（hard FG）/ outer（hard BG）は
  ノードにせず、隣接 Band 画素の t-link（∞容量）へ畳み込む。
- **n-link**（隣接 Band 画素）: `w = CAP*(g + LAMBDA)`、
  `g = exp(-(edge/SIGMA)^2)`（平滑≈1=切りにくい、強エッジ≈0=安く切れる=吸着）。
  `SIGMA` は strength から決定（Edge Lock が高いほど小さく、弱いエッジでも吸着）。
  `LAMBDA` floor が「エッジが無ければ元境界で切る」default-cut を与える。
- **t-link**: hard FG/BG は ∞。それ以外は後述の `score`（幾何 prior + color）から軟容量。
- **min-cut 側復元**: 残差グラフ `csr - flow` の正容量辺を
  `breadth_first_order` で source から BFS、到達 Band 画素 = FG。

### 12.2 設計上の要点（旧実装との違い）

1. **hard FG = スケルトン（medial axis, `skimage.morphology.skeletonize`）＋ add-stroke 中心線。**
   - 細いブラシ帯ではエッジが body を貫いてもクリップできる（centerline は細い）。
   - ソリッド領域では medial 線が連続するため、内部を横切る強エッジでも cut が
     その線を切断できず、**描いた領域の内部が保持**される（保持と吸着の両立）。
   - これが第 4.1 節「保持と吸着の両立が難しい」への回答。
2. **color データ項（Boykov–Jolly 流）を追加。**
   - 純幾何 min-cut は境界を短くしようとする収縮バイアスを持ち、
     一様画像でブラシが芯まで縮む / 凹境界が内側へショートカットする。
   - そこで「seed 色に近い→FG、背景シェル色に近い→BG」の単項を加え、
     色で body を保持し、色違いの spill（雲エッジを越えて暗部へはみ出した部分）を切る。
   - **外側（マスク外）では color を BG 方向にしか効かせない。**
     これにより grabCut 系の「同色を遠くまで巻き込む爆発」を構造的に封じた
     （第 10 節「同色のまま伸ばす圧力」「PR_FG が背景へ縮む」への対処）。
   - 外向きの拡張はエッジ（n-link）が駆動し、color では駆動させない。
3. **`radius` の役割を分離・再定義（第 11.5 節の課題）。**
   - `radius` = **探索帯 Band の半幅 = ストローク/マスク境界からエッジを探して
     境界を動かせる最大距離**（＝ユーザー認識の「エッジ探索範囲の拡張幅」）。
   - ROI padding は `radius` から内部派生、target_edge 探索・BG shell・guard 等の
     **三役兼任を撤廃**。`MIN_BAND=2` で radius=0 でもブラシ内 snap を残す。
   - 「膨張しない」は radius のクランプではなく、**幾何 prior の floor（外側）＋
     default-cut** で担保する。

### 12.3 解決した破綻モード（観察結果）

- **木/雪/雲の細かいテクスチャを拾わない**: 孤立エッジは閉じた安い cut を作れず棄却
  （`far_texture` 系で 0 を確認）。第 4.2 節の「マダラ」を解消。
- **大 radius で爆発しない**: 600×400 fixture で radius 0–200 を通して
  support/hint 比 = 0.76–0.86 に収束。第 11.1 節「radius を上げると急に破綻」を解消。
- **エッジ越え漏れ・二重線なし。** ソリッドマスクが内部エッジを跨いでも保持。
- **消しゴム維持 / 全消しで空。**
- **拡大表示・通常表示・export 一致**（モジュールは座標系非依存。
  既存の `extended_params.render_freedraw_edge_refine_full_view` の余白付き ROI を流用）。

### 12.4 テスト

- 新規 `tests/test_edge_refine.py::DrawQuickSelectMinCutTest`（11 シーン、面積比/領域判定）:
  直線エッジ / S字 / 凹(U) / 木ノイズテクスチャ / 近接2エッジ / 太ブラシ遠エッジ /
  ストローク外側 / 消しゴム分断 / 全消し / 一様無膨張 / zoom-crop と full の一致。
- 旧 Draw テストのうち、旧 grabcut/target_edge の挙動に**過適合していたもの**
  （垂直エッジ端への ribbon 強制成長 / 一様画像のピクセル完全一致 / 同色ソフト境界の
  spill 強制クリップ / 24px 外側への積極リーチ）は、**doc 記載の本来の意図**へ緩和した。
  エッジ越え禁止・遠方漏れ禁止・ストローク保持・無爆発・テクスチャ非選択といった
  本質ガードは維持。
- `pixi run python -m unittest discover -s tests -p test_edge_refine.py` → **54 tests OK**。

### 12.5 観測性・チューニング・退避

- **debug mosaic**（`PLATYPUS_DEBUG_EDGE_REFINE=1`）に追加した plane:
  `image_edge` / `edge_cost`(=g) / `color_score` / `seed_fg` / `seed_bg` /
  `prior` / `cut_boundary`。
- **数値ログ**: `[DRAW_QS] hint= band= edge_px_in_band= support= ratio= comps= max_flow= radius=`（INFO）。
- **無コード調整つまみ**（環境変数、実画像チューニング用、`draw_quick_select.py` 冒頭の定数）:
  `QS_FLOOR_IN`(0.25) / `QS_FLOOR_OUT`(0.22, 外向き膨張抑制) /
  `QS_BETA`(0.28, prior 重み) / `QS_COLOR_W`(1.1, color 重み)。
- **退避**: `PLATYPUS_DRAW_QS_LEGACY=1` で旧経路へ即戻し。

### 12.6 既知のトレードオフ・残課題

- **遠い（例: 半径いっぱいの）エッジへの積極的な外向きリーチは意図的に抑えている。**
  外向き prior（anti-inflation）と両立しないため、爆発を避ける側に倒した。
  「中に描いて遠いエッジまで取りに行く」用途は弱め（第 11 節の爆発回避を優先）。
- **同色かつソフト境界の spill は積極クリップしない。**
  色でもエッジでも区別できない場面（背景テクスチャが前景と同色）では、
  明確な根拠が無い限り描いたマスクを保持する保守側に倒している。
- **min-cut のメトリケーションで凸の端が ~1px 丸まる**（一様画像で僅かに縮む）。実害は小。
- **保留（旧コード撤去, plan Phase 4）**: legacy フラグで 1 サイクル様子見後、
  `_draw_grabcut_band_support` ＋約 30 helper ＋ Random Walker 残骸（~1500 行）を削除。

### 12.7 判断

第 9 / 11 節の「次の一手 = 2D 探索帯 + min-cut で閉じた境界を決める」を実装し、
既知 fixture 依存のチューニングから、**幾何（min-cut）＋色（データ項）＋エッジ（n-link）**
という一般性のある定式化へ移行できた。今後の改善は、爆発を再導入しない範囲で
`QS_*` つまみと debug mosaic を見ながら実画像で詰めるフェーズに入る。

----

## 13. 2026-06-06 外向きリーチ修正（実画像テストのフィードバック反映）

実機テストで 3 点の指摘:

1. `radius` がほとんど効かず、**ブラシがエッジに届いていない所でエッジを捉えられない**。
2. 雪側に中心点がある時は良いが、**空側に中心点がある時にエッジを拾わない**（左右非対称）。
3. メニューで Quick Select を選んだ直後／拡大直後に計算が走らないように見える時がある。

### 13.1 原因

- 1・2 の主因は、第 12.2 節で爆発回避のために入れた
  **「外側 color は BG 方向にしか効かせない」クランプ＋外向き prior の単調増加**。
  これが「同色ギャップを越えてエッジまで届く外向きの駆動力」を殺していた。
  ブラシがエッジに接していない／中心が逆側にある時、外へ伸びられず捉えられなかった。
  （非対称に見えたのは、接している側は内側クリップで動くが、離れた側は外向きリーチ依存のため。）

### 13.2 修正

- color によるリーチ（同色領域の塗り）は grabCut 的爆発を招くため**不採用のまま**。
  外向きリーチは **幾何（エッジ = 安い n-link）で駆動**する方式へ。
- **外向き prior を「帯のほぼ全域で軽く平坦・rim 近傍だけ急峻」に変更**:
  `dist_out/R_out < REACH_FRAC` の間は軽い floor（`QS_FLOOR_OUT`）、
  そこから rim で 1 まで立ち上げる。
  → radius 内の任意のエッジへ境界が移動して吸着でき（= radius が効く）、
    エッジが無ければ rim ランプが radius でハードに頭打ち＝featureless は膨らまない。
- 結果（計測）: ブラシがエッジから離れていても radius を上げるとエッジへ到達。
  空側中心も対称に到達。雪 fixture は radius 0–200 で比 0.79–1.12 に収まり**非爆発**。

### 13.3 トレードオフと新つまみ

- ソフト境界の直下が前景と同色のテクスチャだと、リーチがそのテクスチャ側へ
  ~8–20px 行き過ぎることがある（色でもエッジでも区別が付かない領域）。深部漏れは無し。
- つまみ追加: `QS_FLOOR_OUT`(0.12, **下げると遠くまで届く / 上げると控えめ**)、
  `QS_REACH_FRAC`(0.85, 軽い floor を保つ帯の割合)。
- 過適合だった旧テスト（photo/snow の外側成長境界）は「非爆発・深部漏れ無し」へ意図緩和。

### 13.4 指摘 3（計算が走らない）の所見

- マスクのキャッシュ鍵には mode/radius/strength と変換 matrix が含まれ、
  Quick Select 選択・拡大のいずれでも cache は invalidate される（鍵の問題ではない）。
- 拡大時の full-view ROI 経路は **original 画像を source** にするため crop 生成待ちに依存しない。
- 最有力は **`mask2_edge_refine_radius` の既定 0**。radius=0 は内側クリップのみ（外向きリーチ 0）で
  見た目の変化が小さく「走っていない」ように見える。**radius を上げると即リーチが効く**。
  → 既定値を小さめの非 0（例 20–30）へ戻すと選択直後から手応えが出る（要判断、UX 既定値のため未変更）。
  実際に動いているかは `PLATYPUS_DEBUG_EDGE_REFINE=1` の `[DRAW_QS]` ログで確認可能。

### 13.5 外向きリーチの「エッジ手前で止まる」精度問題（追加修正）

実機追試:「リーチで検索範囲は広がるが、**エッジの少し手前で止まり**、ブラシ内クリップ時と精度が全然違う」。

- 原因: `_draw_snap_edge_strength` のエッジは数 px ボケており、安く切れる帯が広い。
  内側クリップは内側 prior（FG）が境界をエッジの外側へ押すので精度が出るが、
  外向きリーチは外側 prior（BG）が境界をボケ帯の**内側**へ引くため、ピーク手前 ~3px で止まる。
  （計測: 真のエッジ x=120 に対し x=117=es0.3 で停止。）
- 修正: n-link 用エッジを **skeletonize でリッジ（1px）へ細線化**してから g を計算
  （`_thin_edge_to_ridge`）。安く切れる場所がエッジピークだけになり、境界がピークへ吸着。
  計測: x=117→**x=118–119**（真のエッジ x=120 に対し 1–2px）。
- ただし細線化を強くしすぎると (a) 遠ギャップへ届きにくくなる、
  (b) テクスチャの細エッジも鋭い標的になり誤リーチが増える。
  → `QS_EDGE_FALLOFF`(0.8) でリッジ幅を調整。0.8 で精度（1–2px）と非誤リーチを両立。
- リーチ距離は `QS_FLOOR_OUT`(0.12) で制御。小ギャップ（〜20px）は既定で精密到達。
  さらに遠くを取りたい場合は `QS_FLOOR_OUT` を下げる（誤リーチ/凹膨らみとのトレードオフ）。

### 13.6 残: 空側中心の取りこぼし（実画像）

合成シーンでは空側中心も対称に到達・吸着するようになったが、ユーザーの実画像では
まだ取りこぼす場合があるとのこと。color/エッジどちらが効いていないかは画像依存のため、
当該箇所の `PLATYPUS_DEBUG_EDGE_REFINE=1` mosaic（`edge_cost`/`color_score`/`seed_fg`/
`cut_boundary`）での切り分けが必要。次の調査対象。

### 13.7 実画像 `tests/guide_full.png` での確定診断と color 信頼度修正

ユーザー提供の mosaic ＋ 元画像（白い雪雲／青空／樹氷）で再現・計測した結果:

- **`color_score` が死んでいた**: 雪景色は低コントラストで、雲と空の FG/BG 中央値の
  LAB 距離が ~6 と、`_color_score` の旧ハード閾値 `< 6.0` に引っかかり color=0 になっていた。
  → **ハード閾値を信頼度スケールに置換**（`conf = clip((sep-MIN)/SCALE,0,1)`、`QS_COLOR_MIN_SEP`=1.5,
    `QS_COLOR_SEP_SCALE`=6.0）。低コントラストでも弱い color 信号が残り、雲/空の輝度段差へ
    境界を寄せる助けになる。合成テスト（高コントラスト）は conf≈1 で不変。
- **「空側で 5px 手前」は計測アーティファクトだった**: 真のエッジ強度リッジ基準で測り直すと
  **空側は毎回リッジに ±0px で着地**（es 0.9–0.97）。むしろ snow 側が稀に直下の樹氷エッジへ
  +7/8px 行き過ぎる（busy reach）。本セッションの細線化＋color 信頼度で空側の吸着は良好。
- **回帰固定**: `tests/guide_full.png` を使った実画像テスト 2 本を追加
  （`test_real_image_sky_side_snaps_to_cloud_edge` / `..._cloud_side_...`）。雲/空どちら側中心でも
  ブラシが跨いだ反対側を切り、自分側を残すことを検証。`Ran 56 tests OK`。

残: ユーザー環境でまだ空側取りこぼしが出る場合は、(a) 旧版で試している、
(b) 雲縁が特に diffuse な箇所、のいずれか。最新で当該座標を指定してもらえれば
その点をピンポイントで再現・調整する。

### 13.8 本番入力の確定診断（QS_DUMP_INPUT）と radius 予測可能性の修正

`compute_draw_support` の入力（guide/mask/radius/strength/stroke）を `.npz` で保存する
フックを追加（`QS_DUMP_INPUT=<dir>`、または `PLATYPUS_DEBUG_EDGE_REFINE=1` に相乗り）。
ユーザーから本番の `qs_input` を取得し、**本番解像度・本番色空間・本番 radius**で再現できた。

確定した「長いと失敗」の正体:

- 本番入力は **radius=1, strength=0（ほぼ既定値）**。長いストロークが**雲塊から ~130px 下の
  開けた青空**（直下に樹氷の天辺）に描かれていた。
- 旧設計（hard FG = 中心線のみ）では、**radius に関係なくブラシ body 全体が可変**だったため、
  ブラシが偶然重なった樹氷エッジへクリップして**底に欠け（ノッチ）**が出た。短いストロークは
  body が小さく欠けが目立たない → 「短い OK / 長い 失敗」。
- 雲塊は遠すぎ（~130px > radius）て届かない＝「エッジを拾わない」とも一致。

修正: **内向きクリップを radius で頭打ち**（`inner_fg = comp & (dist_in > R_out)` を hard FG に
追加。doc 5.1 の `band = dilate(M,r) \ erode(M,r)` に回帰）。
中心線 core はブラシより細い部分を可変に保つフォールバック。結果:

- 本番 radius=1: removed 545→**31px**（＝ほぼ描いたまま、ノッチ消失）。
- radius を上げると in/out 両側へ比例してスナップ（r=20→693, r=60→1480 clip）。
- **radius = 境界を動かせる距離**という直感どおりの予測可能な挙動に。

これに伴い「radius≈0 でブラシ内を積極クリップ」を前提にしていた旧テスト 3 本を、
スピル幅相当の radius で検証するよう更新（snow→radius14, fixture→radius24）。全 56 OK。

**UX 上の含意**: この時点では `mask2_edge_refine_radius` の既定 **0** が「ほぼ何もしない」
になっていた。第 14 節で brush 半径を基準 radius にしたため、既定 0 でもブラシ範囲内の
スナップが効く意味へ変更した。

----

## 14. 2026-06-06 brush 半径を Draw Quick Select の基準 radius に変更

ユーザー再現の `edge_refine_debug/qs_input_000.npz` / `qs_input_001.npz` を確認したところ、
どちらも `radius=1, strength=0` で、ブラシ径はそれぞれ約 46px / 193px だった。
旧実装では `QS_BRUSH_AS_RADIUS=0` が既定のため、ブラシ半径 23px / 97px に対して
探索 band が常に `MIN_BAND=2px` となり、拡大時ほど「大きいブラシなのにエッジを見ない」
状態になっていた。

### 14.1 変更

- Draw Quick Select の内側 clip 幅を **`brush half-width + UI radius offset`** に変更。
  UI の `Quick Radius=0` は「ブラシ内部だけを探索」、正値は外側にも広げる、負値は
  内側探索を狭める。
- 外側 grow 幅は **`max(UI radius offset, 0)`** に分離。`Quick Radius=0` では
  ブラシ外へ単純に膨らまない。
- Draw 経路だけは `edge_refine.refine_mask_edge_aware` の `max(1, radius)` clamp を通さず、
  offset としての 0 / 負値を維持する。
- UI slider の最小値を `-200` に変更。
- 太ブラシの内部に強いエッジがある場合、stroke centerline から強エッジを越えずに到達できる
  seed-side を計算し、反対側 inside には BG prior を与えるようにした。
  これにより、ブラシが雪/空境界を跨いでも「中心線側」を残し、反対側を切れる。
- 同じ Draw レイヤーに複数 stroke がある場合、解く単位を「描画後 mask の connected component」
  ではなく **stroke 由来の unit** に変更。各 unit は自分の stroke size から internal radius を
  解決し、自分の centerline core と local color model を使う。離れた1本目/2本目、または
  サイズの違う複数 stroke が、同じレイヤー内で基準を混ぜないための修正。
- stroke 入力時の BG color shell は、全周 median が snow/sky の両側を混ぜて FG と近くなる場合が
  あるため、FG から最も離れた方向/象限の shell median を使う。これにより、ユーザー再現の
  `qs_input_001.npz` で 2 本目の `color_score` が 0 固定になる問題を避ける。
- EdgeLock のレンジを拡大し、Draw Quick Select では **0=強い ridge だけを見る strict**、
  **100=弱い/diffuse ridge も候補にする loose** として扱う。旧レンジでは 0 の時点で弱いエッジが
  そこそこ安く、slider を動かしても見た目が変わりにくかった。
- refined/matte の馴染みは EdgeLock から切り離し、全体に細め・強めへ変更。前回の
  「明るい雪/雲側の alpha リムを厚く残す」補正は、雪側に厚く残る症状を悪化させるため撤回。
  `edge_width=2.25` を基準に、半透明リムを薄く短くする。最エッジ alpha の追加低下は
  **暗い selected side（空側）だけ** に限定し、明るい selected side（雪側）はこの cap を
  掛けない。`qs_input_009.npz` では edge rim の mean/p50 が約 `0.432/0.432` まで下がり、
  `qs_input_007.npz` の雪側は約 `0.583/0.605` のまま残す。
- 単一 stroke 再現 `qs_input_023.npz` で、radius を上げても雪側の見た目が変わらない原因を確認。
  `candidate` は増えるが、外側 positive color を FG へ引っ張らない安全策により support がほぼ
  動かない。そこで min-cut 後に **selected-side edge band を小さく復元**する処理を追加。
  条件は candidate 内、support 隣接、強い image edge 上、かつ color_score が選択色寄り。
  初回は 1px rim だったが、雪の diffuse edge では不足したため最大 4 step まで復元する。
  通常はノイズ画像での膨張を避けるため `QS_EDGE_RESTORE_COLOR_MIN=0.05` とし、色側の確信がある
  edge に限定。さらに、selected side が背景 shell より明るい stroke だけは
  `QS_BRIGHT_EDGE_RESTORE_COLOR_MIN=-0.70` まで緩める。これは混色になった雪/空の最エッジを
  雪側 support に含めるためで、暗い selected side（空側）では緩めない。
  `QS_EDGE_RESTORE_EDGE_NEAR=2` で diffuse edge の近傍も許可する。`qs_input_007.npz` では
  restore が約 2748px になり、雪側の seam へ届く。`qs_input_009.npz` では restore は約 1px に
  留まり、空側のほぼ完成している境界を太らせない。
- 片側だけの見た目では不足を見逃しやすいため、`scripts/check_draw_qs_pair_union.py` を追加。
  空側/雪側の debug npz を replay し、refined mask を足した alpha が shared seam 上で
  `0.5` 未満になる gap ratio を測る。`qs_input_007.npz + qs_input_009.npz` は
  `gap_ratio=0.0144`, `alpha_components=1` で、旧実装の `0.177` から改善。
- 追加観察: 「EdgeLock が厳しくなっただけに見える。しかも拡大時だけ」。
  最新 debug では同じ single stroke が current-view `657x657`（brush 54.75）と full-view
  `1132x1060`（brush 190.78）で別々に解かれていた。full-view high-res solve では EdgeLock が
  support ratio を `0.690 -> 0.534` まで大きく変え、通常 current-view は `0.703` 付近でほぼ不変。
  つまり問題は edge_restore ではなく、拡大時だけ通る full-view solve のスケール非不変性。
  暫定対応として `render_freedraw_edge_refine_full_view` を既定 OFF に戻し、拡大時も current-view
  経路で Quick Select を解く。検証用には `PLATYPUS_DRAW_QS_FULL_VIEW=1` で再有効化できる。
  full-view を使う場合のため、`draw_pixel_scale` による edge sensitivity の補正 hook も追加。
- `mask2_edge_refine_strength` の既定値を `60` に変更。`0` はかなり strict になったため、
  新規 Draw Quick Select では弱い雪/空境界も拾う中間値から始める。
- 描画中 preview は、touch down 時点の確定済みマスクを凍結し、現在 stroke は raw preview として
  合成する。touch up で全体を Quick Select 確定再計算するため、move ごとに過去領域が
  min-cut で吸われたり膨らんだりする挙動を避ける。

### 14.2 確認

- `qs_input` 系: `Quick Radius=0` では `grow_radius=0`、candidate はブラシ外へ膨張しない。
  内側 clip radius は各 stroke の brush half-width になる。
- 追加テスト:
  - `test_brush_size_is_draw_quick_select_base_radius`
  - `test_draw_quick_select_resolves_radius_per_stroke`
  - `test_thick_brush_uses_seed_side_when_edge_is_inside_brush`
  - `test_edge_lock_expands_weak_edge_sensitivity`
  - `test_selected_edge_rim_restores_only_selected_color_side`
  - `test_natural_edge_matte_stays_narrow`
- ユーザー再現:
  - `qs_input_001.npz`: 2 本目 component の support/hint が **0.893 → 0.665** になり、1 本目と
    同程度に edge 側で削れる。
  - `qs_input_002.npz`: 3 component がそれぞれ 0.671 / 0.665 / 0.573。
  - `qs_input_033.npz`: matte を細くした後、soft pixel は EdgeLock 0 で **9171**、既定値 60 で
    **10615**。EdgeLock は support 面積を **158573 → 152885 → 139703**（0/60/100）と
    大きく変える。
  - `qs_input_023.npz`: radius 0 / EdgeLock 60 / single stroke。`edge_restore=172px`。
- `pixi run python -m unittest discover -s tests -p test_edge_refine.py` → **62 tests OK**。
