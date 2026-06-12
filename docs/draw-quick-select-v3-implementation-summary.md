# Draw Quick Select V3 Implementation Summary

この資料は、Draw Quick Select V3 を作る時に重視した設計判断と、今後の実装・調整で守るべきポイントをまとめる。

## 1. V3 の目的

最終目標は「人間が見た時にエッジとして読める場所で自然に切る」こと。

ただし、1ストロークで常に完璧に切ることよりも、次の性質を優先する。

- ストロークを追加しても、既存ストロークの判断基準が勝手に変わらない。
- 雪側/空側のような反対側から作ったマスクが、境界で自然につながる。
- Radius、EdgeLock、Edge Bias の意味が分離していて、調整結果を予測できる。
- 拡大率や表示位置が変わっても、同じ入力なら同じ形になる。
- エッジがない場所ではブラシ形状が勝手に膨らまない。
- 実験 npz と expected PNG で、目視だけではなく再現可能に検証できる。

## 2. V1/V2 で起きた主な問題

### 2.1 ストローク全体の再設計

複数ストロークを同じレイヤーに描けるため、全体を一括で見ると、新しいストロークによって既存ストロークの色基準・エッジ基準が変わる。

その結果、ユーザーから見ると「描いた場所だけでなく、前に描いた領域まで動く」ように見える。

V3 では、add stroke は原則として stroke 単位で解く。

### 2.2 UI パラメータの意味が混ざる

以前は EdgeLock が次の複数の意味を同時に持ちがちだった。

- どの強さの edge を boundary と認めるか
- brush 内の seed 側/反対側を分ける barrier をどう見るか
- edge rim をどれだけ復元するか
- gap fill をどれだけ許すか
- alpha の柔らかさ

これではスライダーを動かしても、ある部分では厳しくなり、別の後処理では緩くなり、体感が読めない。

V3 では「EdgeLock はエッジ感度」「Edge Bias は境界位置/alpha」「Radius は探索範囲」という分離を基本にする。

### 2.3 Radius が探索範囲以上の意味を持つ

Radius を大きくした時に、エッジがない場所でも brush stroke が膨らむ問題があった。

V3 では Radius を「solver が境界を探してよい範囲」として扱う。Radius を大きくしても、accepted edge や同側 fill の根拠がない限り、featureless 領域へ膨らませない。

### 2.4 拡大率で結果が変わる

表示倍率が変わると、edge map の太さや pixel scale が変わり、同じストロークでも別の結果になる問題があった。

V3 では `pixel_scale` を debug 入力に含め、edge strength の正規化と replay で検証できるようにする。

## 3. コア設計

## 3.1 Add stroke は独立に解く

V3 の add path は、各 stroke を個別に rasterize し、V2 の add-only solver を stroke ごとに呼ぶ。

実装箇所:

- `cores/mask2/draw_quick_select_v3.py`
- `compute_draw_support()`
- `_single_stroke_mask()`

基本フロー:

1. `draw_strokes` を normalize する。
2. erase stroke が混ざる場合は安全側として V2 fallback を使う。
3. add stroke だけなら、stroke ごとに mask/seed を作る。
4. stroke ごとに `_v2._compute_add_only_support()` を呼ぶ。
5. stroke local な support/candidate/debug plane を得る。
6. V3 の post process を適用する。
7. add stroke の support/alpha を union/max 合成する。

重要な不変条件:

- 2本目の stroke が、1本目の colour model や EdgeLock auto を変えてはいけない。
- stroke local の候補領域外に support を増やしてはいけない。
- add support の合成は union、alpha は max を基本にする。

## 3.2 Erase はまだ fallback 優先

V3 の主目的は add stroke の安定化。erase stroke は add と同じ基準で完全に解くには別 profile が必要。

現在は erase が混ざる場合、V2 fallback を使い、V3 alpha/postprocess を重ねる。

将来 erase を V3 化する場合も、add と erase の auto profile は分けるべき。

## 4. UI パラメータの意味

## 4.1 Brush Size

Brush size は、その stroke の内部 radius の基準。

- stroke の中心/内部 seed を作る。
- brush 内で edge が crossing している場合、その edge で切れる。
- ただし brush size だけで外側に support を作ってはいけない。

## 4.2 Quick Radius

Quick Radius は brush radius に対する探索範囲 offset。

- `0`: brush size 由来の内部探索を基本にする。
- `+`: 外側方向にも edge を探しに行ける。
- `-`: 探索範囲を狭くする。

実装上は `draw_quick_select._resolve_scales()` が次を決める。

- `band_half_width`: inward clip radius
- `grow_radius`: outward grow radius
- `roi_pad`: solve ROI padding
- `stroke_half_width`: stroke size 由来の基準幅

注意点:

- Radius を上げることは、support を膨らませる命令ではない。
- エッジのない外側 grow は `_limit_smooth_outside_growth()` で落とす。
- featureless growth を防ぐテストを必ず見る。

## 4.3 EdgeLock

UI の EdgeLock は offset 型。

- `0`: auto 推定
- `+`: stricter。強い edge だけを見たい
- `-`: looser。弱い/diffuse edge も見たい

内部では 0..100 の effective sensitivity に解決する。

- 低い値: 強い crisp ridge だけを boundary とする。
- 高い値: 弱い/diffuse edge も boundary 候補にする。

実装箇所:

- `draw_quick_select_v2._resolve_edge_lock()`
- `draw_quick_select_v2._estimate_auto_edge_lock()`
- `draw_quick_select_v2._apply_edge_lock_offset()`
- `draw_quick_select._edge_policy()`

重要な設計:

- UI offset を auto から直接 `auto - offset` で動かすと、auto が極端な case で破綻しやすい。
- `_apply_edge_lock_offset()` で auto 近傍の動きを damping し、+/- の方向性を保ちつつ急に別挙動へ飛ばないようにする。
- EdgeLock は「どの edge を認めるか」に使う。境界の選択側 offset や alpha softness は Edge Bias 側に寄せる。

## 4.4 Edge Bias

Edge Bias は、すでに accepted edge として認めた境界付近で、どちら側の pixel を support/alpha に含めるかを調整する。

- `+`: 選択側へ少し含める/alpha を柔らかくする方向
- `-`: edge rim を控えめにする方向

重要:

- Edge Bias は edge sensitivity を変えてはいけない。
- Edge Bias は colour membership を緩めるためのものではない。
- positive bias が brush-shaped growth にならないよう、候補と edge threshold で制限する。

実装箇所:

- `draw_quick_select_v3._apply_boundary_bias()`
- `draw_quick_select_v3._support_alpha_from_edge_softness()`
- debug plane `boundary_bias_px`

## 5. Edge policy の分離

V3/V2 add path は `draw_quick_select._edge_policy()` を通して、EdgeLock から各 threshold を解決する。

主な policy 値:

- `ridge_threshold`: raw edge を 1px ridge 化する閾値
- `ridge_falloff_sigma`: ridge 周辺の falloff
- `cut_sigma`: graph n-link cost の edge response
- `side_threshold`: brush 内の seed-side/opposite-side split
- `outside_keep_threshold`: 外側 support を残す edge 条件
- `restore_threshold`: selected-side rim restore 条件
- `boundary_bias_px`: Edge Bias の px offset

設計上の注意:

- policy plane は debug contract。名前を変える時は docs と test も更新する。
- `side_threshold` は EdgeLock と完全同期させすぎると、スライダー操作で seed-side 分割まで変わり、体感が不安定になる。
- side split は「brush 内で明らかに反対側になった領域を落とす」ための補助であり、blind shrink にしてはいけない。

## 6. Stroke local correction

V3 は原則 generic solver だが、stroke 形状から安全に分かる補正は stroke local に入れる。

## 6.1 Thin elongated stroke

細長い stroke は、上辺/下辺の edge で切りたいことが多い。

条件:

- stroke half width が小さい
- bbox の長辺/短辺比が大きい
- EdgeLock が十分 edge-sensitive

処理:

- side split threshold を下げ、細長い brush 内の反対側を落としやすくする。

実装:

- `_v2_is_thin_elongated_unit()`
- `_v2_thin_elongated_side_edge_thresh()`

## 6.2 Subtle broad bright dab

`simple` のような広い淡色 stroke では、境界の片側がわずかに異なるだけで、default だと edge に少し乗り切らない。

条件:

- selected luma delta が小さめの正値
- stroke half width が十分大きい

処理:

- inside colour BG prior を弱く入れる。
- offset mode では unit auto EdgeLock を少し strict 側へ寄せる。

実装:

- `_v2_inside_color_bg_thresh()`
- `_v2_inside_color_bg_weight()`
- `_v2_unit_edge_lock()`

重要:

- この補正は broad subtle dab 限定にする。
- `snow_edge` のように大きい輝度差のある雪側 boundary には適用しない。
- `lowcontrast` のように本当に弱い edge を拾う必要がある case には適用しない。

## 7. Same-side gap fill

tree 系では、枝の間にある空など、edge だけでは support が分断されやすい。

V3/V2 では selected-side gap fill を使う。

考え方:

- support に近い。
- colour score が selected side らしい。
- edge 近傍、または component confidence を満たす。
- hard seed と接続できる。
- 面積 cap を超えない。

実装:

- `draw_quick_select_v2._v2_fill_same_side_gaps()`
- `draw_quick_select_v3._fill_selected_color_voids()`

注意:

- gap fill は便利だが、blind dilation にしてはいけない。
- EdgeLock が高いほど弱い gap も拾えるが、面積・色・接続条件で制限する。

## 8. Alpha と binary support の分離

V3 の重要な分離は、binary support と support alpha を別物として扱うこと。

- binary support: 選択された領域。後続処理や合成の基礎。
- support alpha: soft/diffuse edge の見た目の馴染み。

Edge Bias や edge softness は alpha に影響できるが、binary support を無制限に増やしてはいけない。

実装:

- `draw_quick_select_v3._support_alpha_from_edge_softness()`
- debug plane `support_alpha`

## 9. Debug / NPZ / Corpus

## 9.1 NPZ の役割

`qs_input_*.npz` は、本番 UI で solver に渡った入力を保存し、同じ条件で replay するためのもの。

保存する主な値:

- guide
- mask
- seed_mask
- strokes
- radius
- strength
- pixel_scale
- strength_mode
- edge_lock_auto
- edge_lock_effective
- edge_lock_offset
- edge_bias

仕様:

- `docs/draw-quick-select-debug-npz-spec.md`

注意:

- 古い npz は `strength_mode` を持たない場合がある。
- 古い npz を replay すると、現在の GUI の `0=auto` と違う意味になることがある。
- GUI 相当を見る時は `strength_mode=offset` と `strength=0` の条件を意識する。

## 9.2 Debug planes

V3 で重要な debug plane:

- `edge_lock_auto`
- `edge_lock_effective`
- `edge_lock_offset`
- `edge_lock_mode_offset`
- `edge_policy_ridge_threshold`
- `edge_policy_restore_threshold`
- `edge_policy_side_threshold`
- `edge_policy_outside_keep_threshold`
- `boundary_bias_px`
- `v3_same_color_void_fill`
- `v3_boundary_bias_delta`
- `support_alpha`
- `v3_stroke_count`
- `v3_runtime_ms`

これらは「なぜその境界になったか」を説明するための契約。

## 9.3 Corpus commands

代表的な確認:

```bash
pixi run python scripts/draw_qs_corpus.py report --solver v3 --names simple easy snow_edge lowcontrast tree2_sky roof2 --no-idempotence --no-determinism
pixi run python scripts/draw_qs_corpus.py label-report --solver v3 --label-dir edge_refine_debug/label_exports --names simple simple2 easy snow_edge lowcontrast tree tree2_sky tree2_tree roof2 roof
pixi run python scripts/draw_qs_corpus.py label-diff --solver v3 --label-dir edge_refine_debug/label_exports --out edge_refine_debug/label_eval_v3_current --names simple simple2 easy snow_edge lowcontrast tree2_sky roof2
pixi run python scripts/draw_qs_corpus.py pair tree2_sky tree2_tree --solver v3
```

通常の回帰確認:

```bash
pixi run python -m unittest discover -s tests -p test_edge_refine.py
git diff --check
```

## 10. 速度設計

現在の最大コストは graph solve と大きい ROI の処理。

改善済みのポイント:

- colour score / luma delta / colour weight / restore threshold が同じ周辺 shell を何度も median しないよう、`_color_score_and_luma_delta()` でまとめる。
- Quick Radius / EdgeLock / Edge Bias の UI slider は、drag 中の再描画を debounce する。

まだ残る課題:

- `snow_edge` のような 20万 node 級は min-cut が支配的。
- 単純 downscale は品質を崩すため、常用の高速化としては危険。
- stroke local ROI のさらなる縮小、preview 中だけの coarse solve、Edge Bias の postprocess cache などが候補。

実装箇所:

- `main.apply_mask2_edge_refine_slider()`
- `main.kv` の Quick Radius / Edge Lock / Edge Bias slider
- `draw_quick_select._color_score_and_luma_delta()`

## 11. 実装時のチェックリスト

新しい調整を入れる時は、少なくとも次を見る。

1. `simple`: 広い淡色 stroke が edge に乗るか。
2. `simple2`: 細長い stroke が余計に削れないか。
3. `easy`: 雪側/空側の基本ケースが保たれるか。
4. `snow_edge`: 雪側 edge が後退しないか。
5. `lowcontrast`: 弱い edge を拾う余地が残るか。
6. `tree` / `tree2_sky` / `tree2_tree`: 枝と空、逆側マスク、隙間 fill が破綻しないか。
7. `roof` / `roof2`: 同系色 edge で大きく悪化しないか。
8. pair metric: 反対側マスクを足した時に seam gap が増えないか。
9. zoom/pixel_scale: 表示倍率で形が変わらないか。
10. radius sweep: edge のない場所で膨らまないか。

## 12. やってはいけないこと

- 1つの global colour model で複数 stroke をまとめて再解釈しない。
- EdgeLock で alpha softness や boundary bias まで直接動かさない。
- Edge Bias で colour membership を緩めすぎない。
- Radius を support dilation として扱わない。
- expected PNG に合わせるだけの sample-specific patch を増やさない。
- zoom 差を「見た目だけ」の問題として放置しない。
- debug plane を消して原因追跡不能にしない。

## 13. 現在の主要ファイル

- `cores/mask2/edge_refine.py`: Quick Select entry point。V3 default switch。
- `cores/mask2/draw_quick_select.py`: min-cut core、edge policy、radius scale、postprocess。
- `cores/mask2/draw_quick_select_v2.py`: add-only V2 solver、auto EdgeLock、stroke local correction。
- `cores/mask2/draw_quick_select_v3.py`: per-stroke composition、Boundary Bias、support alpha。
- `cores/mask2/draw_qs_metrics.py`: corpus metrics / label metrics。
- `scripts/draw_qs_corpus.py`: report、label diff、pair metric、export。
- `tests/test_edge_refine.py`: solver regression tests。
- `docs/draw-quick-select-debug-npz-spec.md`: debug npz 仕様。
- `docs/draw-quick-select-v3-policy.md`: V3 の短い policy contract。

## 14. 今後の改善候補

優先度が高いもの:

- erase stroke を V3 化する。ただし add とは別 profile にする。
- EdgeLock の effective 値を plane max だけでなく stroke/unit ごとに見やすくする。
- preview 中の coarse solve または postprocess cache を入れる。
- zoom/pixel_scale の regression test を corpus 側でさらに強化する。
- roof 系の同系色 edge を、sample-specific でなく edge/colour confidence の一般則として改善する。

この資料の基本方針は、調整を増やす時ほど「何のパラメータが、どの範囲に、どの根拠で効くのか」を狭く保つこと。

