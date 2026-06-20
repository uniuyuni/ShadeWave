# Draw Quick Select V4 設計（北極星）

この資料は、V3 までの実測・知見を踏まえて到達した「理想のクイックセレクト」の設計戦略を、
V4 を作る前の確定仕様としてまとめる。V3 実装の詳細は
`docs/draw-quick-select-v3-implementation-summary.md`、その純化の検証ログは
memory `draw-quick-select-v3-idealization` を参照。

## 0. 製品ビジョン（無理のない理想）

> **基本はオート（ラフブラシ→自動領域）／ユーザーが足し消しで補完／エッジの不透明度は自動。**

万能な単一手法は無い（実測で確定）。「auto を完璧にする」のでなく
**「良い auto ＋ 軽い補正 ＋ 自動マット」**にするのが、古典で無理なく届く理想。
完璧を1手法に背負わせるから破綻する。難所（意味的境界）は別系統 SAM3 に委譲する。

### 手法の適材適所（実測で裏づけ済み）

| 手法 | 得意 | 苦手 |
| --- | --- | --- |
| min-cut（領域法） | 色/領域で決まる滑らかな閉領域、空との境界 | 同色・断続エッジ、エッジ追従 |
| edge-following（livewire/最短経路） | 見えるエッジを辿る、小ギャップ橋渡し | クラッター/平行エッジ誤吸着、大ギャップ直線化、エッジ皆無で無力 |
| 学習（SAM3） | 意味的境界（屋根の終わり）を文脈で補完 | 重い依存、予測しにくさ（別系統で担当） |

**roof 系の実測**: 真の境界の **61-71% は edge map に見えている**（生 Sobel 中央値 0.26-0.35）。
つまり信号は入っている。取れないのは「センサーが見えない」でなく
「**min-cut（領域法）が断続エッジを追従しない**」から。→ V4 は edge-following を併用する。

## 1. V4 アーキテクチャ

per-stroke 独立解（V3 の不変条件を継承）で、次の段に再構成する。

```
ラフブラシ（add/erase, 順序保持）
  └ per stroke:
     1. local color model（FG=seed側 / BG=反対側）         ← V3 流用
     2. min-cut で region を取る（色/領域で取れる所を確定）   ← V3 コア流用
     3. ★境界 edge-snap（min-cut 境界を種に、edge-cost 上の
        最短経路=livewire で近傍の見えるエッジへ寄せる）       ← V4 の新コア
     4. binary support（権威）
合成: union(add) \ union(erase)、seam は同じ境界energyで出会う   ← V3 流用(Phase6)
alpha: trimap band を自動マット（natural_edge_matte 系）          ← V3 流用・改善
```

### 1.1 ★エッジは「壁」でなく「スナップ対象」（V4 の肝）

V3 の最大の構造問題は **エッジを min-cut の壁（barrier）にする**こと。これが
実機の2症状を生む（下記 §3）。V4 では：

- min-cut は**領域の所属**を決める（色/prior 主導）。エッジで add をブロックしない。
- 取れた**境界線を種**に、**edge-cost map 上の最短経路（Dijkstra=livewire）**で
  近傍の見えるエッジへ境界を寄せる（band 内）。
- エッジが見える所は乗り、無い所は min-cut 境界（=ブラシ寄り）のまま。
- クラッター誤吸着は **min-cut の領域拘束（band/色）が抑える**（livewire 単体より堅牢）。

これが「min-cut × livewire の組み合わせ」。今回 erase に入れた flood ベースの吸着
（`_snap_kept_boundary_to_edges`）の **shortest-path 一般化**であり、add/erase 両方に効かせる。

## 2. ブラシ補正を予測可能に（V3 の不安定要因を撤去）

「ユーザーが足し消しで補完」前提なので、**ブラシ意図を尊重し、予測できる**ことが最優先。

- **side-split を保守化**: ブラシ芯が明確にエッジを越えて多数派なら反対側扱いしない
  （＝意図的に越えて足したものは増やす）。V3 doc の「明らかに反対側だけ落とす」を厳格化。
  完全同期で EdgeLock 操作が seed-side 分割まで動かす不安定を断つ。
- **「乗せ」と「べた塗り」を EdgeLock で連続・予測可能に分離**: 高=エッジ吸着、低=ブラシ通り。
  Phase 1 の連続 auto-EdgeLock を土台に、**ブラシ近傍の side-split/受理も連続化**（V3 未対応）。
- **エッジは add をブロックしない**（§1.1）。越えたければブラシを伸ばせば越える。

## 3. 直す実機症状（V3 で確認済み）

V4 はこの2つ（同根：エッジ扱いの二値バタつき）を構造的に解消する。

1. **エッジ近辺でブラシ足しても増えない** = side-split がエッジ越え分を反対側として切り捨て。
   → §2 の side-split 保守化 ＋ §1.1 の非ブロック化。
2. **エッジをすっ飛ばしてブラシ形のまま** = エッジ未受理で snap も clip も効かず。
   → §1.1 の境界 edge-snap（受理を band 内 shortest-path に置換、二値判定をやめる）。

## 4. 自動エッジ不透明度（マット）

- V3 の `_apply_natural_edge_matte`（color-line matting）を継承＝既に**自動でソフト alpha**。
- 改善は band 幅の diffuse 適応・髪/毛。ただし**検証用 hair/fur サンプルが必須**
  （無いと過剰 feather か no-op になり実証できない＝Phase 4 の教訓）。
- binary support は不変（alpha は別物）＝V3 の分離原則を厳守。

## 5. V3 から継承する資産（捨てない）

- **Phase 0 ガードレール**: 連続性ハーネス / `sweep-continuity` CLI / zoom 不変 / 順序 /
  add 合成保存テスト。V4 も同じ corpus 回帰をゲートにする。
- **Phase 1 連続 auto-EdgeLock**: `_auto_edge_lock_from_stats`（崖無し・regime 中心忠実）。
- **Phase 6 erase**: V3-native / 描画→消去→追加描画の順序 / 保守的エッジ吸着。
- debug plane 契約 / NPZ 仕様 / label IoU 閾値。

## 6. やらないこと / 限界（正直に）

- roof 同色を **auto で完璧にしない**（古典では原理的に不可と確定）。「良い auto＋補正」で十分。
- 意味的に繋ぐしかない難所は **SAM3**（別系統）。
- 速度は **scipy min-cut 据置**（numba push-relabel は HL+gap でも C-Dinic を超えられず確定）。
- zoom 完全不変は full-res 常時 solve が必要（重い・旧 revert）＝深追いしない。2x は既に良好。

## 7. 構築と検証の進め方

1. **実機の失敗シーンを dump 化**してコーパスに登録（特に §3 の2症状、ブラシ近傍）。
   ```bash
   QS_DUMP_INPUT=edge_refine_debug pixi run python main.py
   pixi run python scripts/draw_qs_corpus.py add <name>
   ```
2. V4 を `cores/mask2/draw_quick_select_v4.py` として V3 横で育てる（env スイッチ）。
3. 各段で corpus 回帰（label IoU 閾値 + Phase 0 連続性/不変/順序）をゲート。
   サンプル固有 patch は禁止（V3 doc 12章）。
4. shortest-path 境界スナップ（§1.1）→ side-split 保守化（§2）→ 実機再チェック→調整、の順。

V4 の基本方針は V3 と同じ：「**何のパラメータが、どの範囲に、どの根拠で効くか**を狭く保つ」。
加えて V4 の核は「**エッジを壁でなくスナップ対象に**し、ブラシ意図を尊重する」こと。

## 8. 実装状況 (2026-06-13)

### ★検証基盤の訂正：corpus label は循環だった
実機で「simple ですらエッジを捉えてない」との指摘で真の手描き GT
(`edge_refine_debug/gt_new/`) を作って測り直したところ、**corpus の expected は
export-labels の旧ソルバ出力コピー≒循環参照**で、IoU を実態と無関係に高く出していた。
真GTでは **tree2_sky IoU=0.16 / tree2_tree=0.42**（循環ラベルは 0.96/0.99）。
**今後の検証は真GT必須**。サイズ診断: roof/simple は領域≈正(sup/GT 1.0-1.2)で
境界だけズレ、**tree2 は領域過剰(1.8-5.5倍)**＝境界でなく領域の問題。

### V4 = 大域 DP リボン edge-trace（§1.1 の本実装）
- `draw_quick_select_v4.py`：V3 領域解＋`_trace_boundary_to_edges`（境界をリボン展開し
  法線±W で「エッジに乗るほど安い」コストの最小経路を DP＝livewire 相当、連結・滑らか・
  ブラシ近傍拘束。clutter は smoothness で回避）。現在は通常実行の既定。
- **真GT で本物の改善（局所スナップの wash と違う）**：
  - simple b_f1 0.684→**0.774**(+0.09)、roof 0.467→**0.550**(+0.08)、simple2 維持。
  - ＝**領域が正しい clean-edge ケース（roof/simple）で一貫して効く**。
- tree2（foliage）は領域過剰＋枝の intricate シルエットで trace 効かず＝**SAM3 領分**。
- 既定: `QS_V4_SNAP_BAND=32`/`QS_V4_TRACE_DISTPRIOR=0.30`/`QS_V4_TRACE_SMOOTH=0.12`
  （広げ過ぎ/prior 弱め過ぎは clutter 崩壊）。
- **snap / snap-alpha 既定 ON**：通常実行は `QS_DRAW_V4=1`, `QS_V4_EDGE_SNAP=1`,
  `QS_V4_SNAP_ALPHA=1` 相当。切り分け時は `QS_DRAW_V4=0` / `QS_V4_EDGE_SNAP=0` /
  `QS_V4_SNAP_ALPHA=0` で旧挙動へ戻せる。

### 残課題（次手）
1. **領域過剰**（roof 上部の背景膨らみ・tree2）＝境界 trace では引き戻せない別問題。
   region 側（band/grow/同側 void fill）を真GTで見直す。
2. 真GT コーパスを増やし、循環ラベルを置換。
3. snap strength を UI 制御に（auto＋調整）。

`pixi run python main.py` で roof/simple のエッジ追従を体感。旧挙動との差分を見る場合は
`QS_DRAW_V4=0 pixi run python main.py` などで切り分ける。
