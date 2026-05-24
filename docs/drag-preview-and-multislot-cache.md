# Drag 中の低解像度プレビュー + Sync Effect Multi-Slot Cache

## ⚠ ステータス: 全 revert 済み (実現困難と判定)

2 回試行 → 2 回とも致命バグで revert。設計上の根本制約 (= `editor.texture_size` が
**raster 出力サイズ + tcg↔window 座標系基準 + display 解像度** の 3 役を兼任) のため、
ここを drag 中に書き換える戦略は座標系破綻 (= マスク作成時のクリック位置が 2 倍ズレる、
マスク CP/overlay も同様にズレる、表示画像も「クロップ画像」見えになる) を引き起こす。

実現するには以下のいずれかが必要 (今回いずれも見送り):

1. **pipeline2 内部だけ low-res で動かし、mask compositing 直前で imgc を full-res に
   upscale する** (= mask raster と CP 座標系は full のまま、pipeline_lv1/lv2 のみ低解像度
   化). pipeline.py 内部の改修が必要。
2. **Effect 基底クラスに per-resolution multi-slot cache を入れる** (= ユーザ提案の
   「diff/hash を配列化、レベルインデックスで分ける」). 60+ サブクラスのうち
   `self.diff = X` 系のパターンが多く、property 化で破綻するため別アプローチが必要。
3. **`editor.texture_size` を 3 役から分離** (= `raster_texture_size` を別属性として
   追加し、`tcg_to_window` などは元の `texture_size` を継続使用). mask code への影響大。

性能改善は別アプローチ (slider event throttle / 重い effect の async 化 / GPU 利用) で
検討する余地あり。

---

## 以下、参考用に過去の試行で書いたドキュメントを残置

---

# Drag 中の低解像度プレビュー + Sync Effect Multi-Slot Cache (試行記録)

## 概要 (TL;DR)

ユーザ操作 (スライダ drag, mask CP drag, paint, crop, mesh 等) の最中だけ pipeline を **低解像度 (default scale 0.5)** で走らせ、drag 終了後に 1 回だけ full-res で再描画する仕組み。同時に `Effect` 基底クラスの cache を 1 → 2 スロット LRU に拡張し、低解像度版と full-res 版の両方を共存させて drag 開始/終了時の cache hit 率を上げる。

既存の `begin_history_*` / `end_history_*` が drag 境界として標準化されているため、それらを wrap するだけで全 slider/mask/crop/mesh 操作を自動カバー。**main.kv / widgets 配下は無修正**。

## スコープと前提

- **全編集** に適用 (mask Geom slider, mask CP, paint, slider 全般, crop, mesh, distortion 等)
- **lv0 は常に full-res** (= 既存仕様、texture_size 非依存)。`core.crop_image` 以降だけ低解像度化
- **sync effect cache を 1 → 2 スロット** に拡張 (= 低解像度版 + full-res 版を共存)

## アーキテクチャ

### 1. drag 状態管理 (`main.py` MainWidget)

```python
self._drag_refcount = 0
self._last_draw_image_kick_ts = 0.0

def begin_drag(self, source=None):
    self._drag_refcount += 1

def end_drag(self, source=None):
    self._drag_refcount = max(0, self._drag_refcount - 1)
    if self._drag_refcount == 0:
        KVClock.schedule_once(self._on_drag_finished, 0)

def is_dragging(self) -> bool:
    return self._drag_refcount > 0

def _on_drag_finished(self, dt):
    if (time.time() - self._last_draw_image_kick_ts) >= 0.05:
        self.start_draw_image()

def _force_reset_drag(self):  # 例外パス用
    self._drag_refcount = 0
```

- **refcount 方式** で nested drag 安全
- `end_drag` の full-res 再描画は **`KVClock.schedule_once` で必ずメインスレッドに dispatch** (graphics 命令は main thread 限定)
- `_last_draw_image_kick_ts` の 50ms gating で button click (= 同期 begin→end) の冗長な 2 度描画を抑制

### 2. `begin/end_history_*` の wrap

`main.py` の 4 関数を以下の通り wrap:

- `begin_history_effect_ctrl` (slider 系): 先頭で `self.begin_drag('effect_ctrl')`
- `begin_history_layer_ctrl` (mask touch 系): 先頭で `self.begin_drag('layer_ctrl')`
- `end_history_effect_ctrl`: 本体を **try / finally** で囲み `finally` で `self.end_drag('effect_ctrl')`
- `end_history_layer_ctrl`: 同上

→ 早期 return (`current_op is None`, subname mismatch, type mismatch) でも end_drag が必ず呼ばれる → refcount リーク防止。

### 3. Pipeline 解像度切替 (`main.py draw_image_core`)

```python
_dragging = self.is_dragging()  # 1 サイクル固定の snapshot
if _dragging:
    _scale = float(config.get_config('drag_preview_scale'))
    _tw = max(64, int(config.get_config('preview_width')  * _scale))
    _th = max(64, int(config.get_config('preview_height') * _scale))
    self.ids['mask_editor2'].set_texture_size(_tw, _th)
else:
    _tw = config.get_config('preview_width')
    _th = config.get_config('preview_height')

# crop_image が前 cycle と違う解像度なら invalidate
if self.crop_image is not None:
    _ch, _cw = self.crop_image.shape[:2]
    if (_cw, _ch) != (_tw, _th):
        self.crop_image = None

img, self.crop_image = pipeline.process_pipeline(..., _tw, _th, ...)
```

- 冒頭で snapshot を取って **1 サイクル中は固定** (途中で flag が変わっても影響しない)
- `mask_editor.texture_size` も同期 (mask raster は `self.editor.texture_size` を参照するため)
- `is_zoomed=True` のときは **低解像度化をスキップ** (texture_size = crop window 実 px なので、縮小すると単なる中央クロップになって "preview" にならない)
- texture_size が前 cycle と違うときは `crop_image` を invalidate (古い解像度の crop が再利用されないように)

### 4. Sync Effect Multi-Slot Cache (`effects.py Effect 基底`)

#### 設計 (最終版: 属性ベース + 退避 cache)

**`self.diff` / `self.hash` は単純属性のまま** (旧 single-slot 互換維持)。cache の出し入れは `check_sync_necessity` の中だけで行う。

```python
self.diff = None
self.hash = None
self._cache = []  # [(hash, diff)] MRU、current は含まない
MAX_CACHE_SLOTS = 2  # = current 1 スロット + 退避 (MAX_CACHE_SLOTS-1) スロット
```

- `_save_to_cache(h, d)`: current を退避リストに front insert (重複除去 + `max_alt=MAX_CACHE_SLOTS-1` で trim)

#### `check_sync_necessity` (atomic swap)

```python
def check_sync_necessity(self, param_hash, efconfig):
    combined_hash = hash((param_hash, efconfig.upstream_hash))
    if self.hash == combined_hash and self.diff is not None:
        return False, combined_hash
    # 退避 cache lookup を SAVE より先に (= round-trip A→B→A の hit を維持)
    found = -1
    for i, (h, d) in enumerate(self._cache):
        if h == combined_hash and d is not None:
            found = i; break
    if found >= 0:
        # Atomic swap: pop cached → push current → adopt cached
        cached_h, cached_d = self._cache.pop(found)
        if self.hash is not None and self.diff is not None:
            self._save_to_cache(self.hash, self.diff)
        self.hash = cached_h
        self.diff = cached_d
        return False, combined_hash
    # Miss: current を退避
    if self.hash is not None and self.diff is not None and self.hash != combined_hash:
        self._save_to_cache(self.hash, self.diff)
    return True, combined_hash
```

#### `reeffect` を no-op 化

multi-slot 化で「強制 cache クリア」が逆効果になるため。サブクラスが Mod/Faces 等の外部リソースを reset する目的で override しているため**メソッド自体は残す**。

#### 重要: なぜ property 化を採用しなかったか (踏んだ地雷)

最初の設計では `self.diff` / `self.hash` を property + pending buffer 化していたが、以下のサブクラスパターンで **致命的に壊れた**:

1. **base `Effect.make_diff`** (`self.diff = img` のみ、hash 未設定) → 既定の no-op effect が壊れる
2. **`RGB2HLSEffect`, `HLS2RGBEffect`, `PipelineCurveEffect`, `pipeline_vs_and_saturation` 系**: `self.diff = result` だけで hash 不使用
3. **`Inpaint`, `Patchmatch`, `BM3D`, `ColorMatch`, `DistortionPainter` 等 10+ 箇所**: `self.diff = result` → `self.hash = param_hash` の順

property の pending buffer は「hash 未設定状態で diff だけ来た場合は破棄」する仕様だったため、上記すべてで diff が None 返却 → effect スキップ → **全パラメータ変更が反映されない致命バグ** に。

→ シンプルな「属性のまま + check_sync_necessity 経由でしか cache 触らない」設計に変更。これで:
- raw `self.hash != param_hash` 派は旧 single-slot 挙動を維持 (drag では恩恵なしだが副作用ゼロ)
- check_sync_necessity 派は multi-slot で round-trip cache hit
- base `make_diff` や `self.diff = X` の直接代入系は影響なし

### 5. config

`config.py` の defaults に追加:

```python
'drag_preview_scale': 0.5,  # drag 中 (slider/CP/paint 中) の preview 解像度倍率
```

`set_config('drag_preview_scale', v)` で動的変更可。`_apply_config` への追加 hook は不要 (`draw_image_core` が毎回 `get_config` で読む)。

## 実装上の注意 / ハマりどころ

### ★ 致命バグ 3 件 (実装中に踏んだ) ★

#### Bug-1: `preview` Image widget が texture_size 自己追従

`main.kv:346` で `Image: size: self.texture_size`。低解像度 texture (320×320) を assign すると **Image widget 自身が 320×320 に縮小** され、preview_widget (640×640) 内に黒帯付きで小さな画像が表示される。ユーザは「中央クロップになっている」と認識する。

**修正**: `draw_image_core` で drag 中処理後、表示前に img を full-res (config preview_width×height) に `cv2.resize(INTER_LINEAR)` で bilinear upscale する。これで widget の `size` が常に full-res になり、見た目は通常解像度のままで内部処理だけ低解像度の本来の挙動になる。

#### Bug-2: `upstream_hash` に解像度・zoom 状態が含まれない

`pipeline_lv0` の最初は `efconfig.upstream_hash = hash(id(img))` (= 元画像 id) で、lv0 effects を通って propagation される。**`core.crop_image` で生成される `imgc` の解像度や `is_zoomed` の情報は upstream_hash に乗らない**。そのため lv1+ effects の `combined_hash = hash((param_hash, upstream_hash))` が解像度切替/zoom toggle で変わらず、**cache hit で古い解像度の diff が返ってきて全パラメータ変更が反映されない**致命バグ。

**修正**: `process_pipeline` の `pipeline2` 呼び出し前で `efconfig.upstream_hash = hash((efconfig.upstream_hash, imgc.shape, bool(is_zoomed)))` を入れる。これで lv1+ effects の cache key が解像度/zoom 込みになり、適切に invalidate される。multi-slot _cache も round-trip で atomic swap hit が機能する。

#### Bug-3: `Effect` 基底クラスの property 化が大量の subclass を破壊

最初の設計で `self.diff` / `self.hash` を property + pending buffer 化したが、以下のサブクラスパターンで diff が None 返却 → effect スキップ:

1. **base `Effect.make_diff`** (`self.diff = img` のみ、hash 未設定)
2. **`RGB2HLSEffect`, `HLS2RGBEffect`, `PipelineCurveEffect`, `pipeline_vs_and_saturation` 系** (hash 不使用、diff のみ)
3. **`Inpaint`, `Patchmatch`, `BM3D`, `ColorMatch`, `DistortionPainter` 等 10+ 箇所** (`self.diff = result` → `self.hash = param_hash` の順)

**修正**: property 化を撤回し、`self.diff` / `self.hash` は単純属性のまま。cache の出し入れは `check_sync_necessity` の atomic swap 内だけで完結する設計に変更。

---

### A. `pipeline.py` の `pre_diff is not diff` (line ~526)

```python
pre_diff = lv0[n].diff
diff = lv0[n].make_diff(rgb, param, efconfig)
if pre_diff is not diff:
    lv1reset = True
```

multi-slot で cache **promote** が起きた場合、pre_diff (旧 front) != diff (promote 後の新 front) で `lv1reset = True` が立つ。downstream の `reeffect()` は no-op 化したので問題なし、ただし `process_pipeline` line 335 の `if crop_image is None or lv1reset == True:` で **crop が無駄に再実行される** ケースがある。

→ 補助対策として `draw_image_core` 側で「texture_size 不一致時のみ crop_image invalidate」を入れた。これで同解像度なら crop 再利用が利く。

### B. `core.crop_image` の出力サイズ

`is_zoomed=True` のときは crop_image が texture_size より小さくなる場合がある (画像実サイズ 1:1 で切り出す)。`(_cw, _ch) != (_tw, _th)` の判定は texture_size 切替直後のみ意図通り発火するので機能する。

### C. graphics instruction threading

`_set_active_composit_matrix` / `_refresh_overlays_main_thread` / `_on_drag_finished` 等の **graphics 命令を含む処理は必ず main thread**。`KVClock.schedule_once(callback, 0)` で確実に dispatch する。background pipeline thread から直接触ると `TypeError: Cannot change graphics instruction outside the main Kivy thread`。

### D. invalidation 挙動

`self.diff = None; self.hash = None` は単純属性代入のため旧 single-slot と同じく即時 None 化。退避 cache は別管理なのでそのまま保持され、次に同一 combined_hash が `check_sync_necessity` 経由で来ると swap で復活する。

invalidation が「本当に古い値を捨てたい」場面 (例: file load) は、primary_param 変化で combined_hash が全 effect で変わるため自然と cache miss → recompute される (= 明示的な全 clear API は不要)。

### E. 「diff 先 hash 後」パターンへの対応

サブクラスで `self.diff = X` → `self.hash = Y` の順に書く実装が 10+ 箇所あるが、属性ベース設計のため順序非依存で動作する。`check_sync_necessity` を使わないこれらのサブクラスは multi-slot 恩恵を受けないが、副作用もない。

### F. `lv1reset` の連鎖

旧コードでは lv1reset = True が downstream `reeffect()` を呼んで全 cache 強制クリアしていた。multi-slot では reeffect は no-op なので、downstream のキャッシュも残り、解像度往復で cache hit が利く。ただし `pipeline_lv0` 内では `reeffect` が呼ばれる構造のままなので、`lv1reset=True` 自体は伝播する (subclass の override が外部リソース reset などをやる可能性のため互換維持)。

### G. memory フットプリント

sync effect 1 個あたり 2 スロット = 約 2x。preview サイズ (典型 640×640×4ch float32 ≈ 6.5 MB) × 効果数 (~50) → 約 650 MB が最大目安。実際は async cache や lv0 cache (lv0 は full-res 1 slot のみ) も加算される。drag 中・往復のたびに warm cache が増えるため、長時間使用でメモリ膨張する可能性あり。**将来的に「N 分アクセスなしの slot は破棄」等の LRU 進化を検討余地あり**。

### H. `draw_image_core` の冒頭 snapshot 必須

`_dragging = self.is_dragging()` を局所変数で固定し、その後すべての分岐で使う。`is_dragging()` を都度呼ぶと cycle 途中で drag が終わって texture_size と pipeline 内部結果が食い違うリスクがある。

## 触ったファイル

| File | 変更内容 |
|---|---|
| `config.py` | defaults に `drag_preview_scale: 0.5` を追加 (1 行) |
| `main.py` | MainWidget の state + `begin_drag/end_drag/is_dragging/_on_drag_finished/_force_reset_drag` メソッド追加 + `begin/end_history_effect_ctrl/layer_ctrl` 4 関数 wrap + `draw_image_core` の snapshot/分岐 + `start_draw_image/sync_draw_image` の timestamp 記録 |
| `effects.py` | `Effect` 基底クラスの `__init__/diff/hash` property + `_commit_pending` + `check_sync_necessity` + `try_async_execution` (hash→diff 順統一) + `reeffect` を no-op に |
| `widgets/mask_editor2.py` | (関連、前タスク #6 A) `_set_active_composit_matrix` で matrix 不変時の schedule スキップ |

それ以外は変更なし (`main.kv` 0 件、各 effect subclass 0 件、`pipeline.py` 0 件)。

## 検証項目 (実機)

1. **mask Geom slider drag**: 低解像度で滑らかに追従、release で 1 回 sharper になる
2. **CP drag** (Circular/Gradient): 同様
3. **FreeDraw stroke**: drag 中低解像度、touch_up で full-res
4. **Crop/Mesh/Lens distortion**: 同様 (begin/end_history_effect_ctrl 経由で自動カバー)
5. **Exposure / Contrast 等 image slider**: 全 slider で効く
6. **is_zoomed=True で drag**: texture_size 縮小 → crop window も縮小 → 軽くなる
7. **button click (begin→end 同期)**: 1 回だけ描画 (低解像度→full-res の 2 度描画にならない)
8. **drag 後の cache hit**: 同じスライダ値に戻すと即座に描画完了 (multi-slot cache hit)
9. **drag 中スライダを行き来 (30°→60°→30°)**: 前回値で cache hit
10. **invalidation path** (例: switch_lens_modifier=OFF): 前解像度 cache が壊れず保持
11. **memory 増**: sync cache ~2x。実測して許容範囲か確認
12. **既存 sync effect 挙動の回帰なし**: 各 effect の make_diff は無修正で動くべき

## 関連タスク / 将来の改善

- **multi-slot LRU の time-based eviction**: 長時間アクセスのない slot を破棄してメモリ削減
- **drag scale をパラメータ別に変える**: 重い effect だけ更に低解像度化 (現状は全 effect 一律)
- **progressive rendering**: drag 終了後の full-res 描画を 2-pass にして低解像度 → full の間を visual ぼかしで埋める
- **async cache の LRU 化**: 現状 `AsyncPipelineManager.cache` は明示 discard しない限り永続。drag で size が増える可能性あり
