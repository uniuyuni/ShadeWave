# マスク Mesh 編集まわりの座標系・サイズ伝播 調査レポート

ユーザー指摘 「ウィンドウサイズを変えると mesh ガイドがずれる / 画像 mesh では起きない」
を受けた構造調査。タイミングではなく **state 伝播の欠落** が根因という結論。

----

## 1. 関係する座標系

| 座標系 | 単位 | 由来 | 範囲 |
|---|---|---|---|
| TCG-normalized | 比率 | `params.norm_param` / `denorm_param` | [-0.5, +0.5] (画像中心 ±半幅) |
| TCG-pixel | px | `denorm_param(norm)` の結果 | [-imax, +imax] (imax = max(orig)/2) |
| image-px | px | `tcg_info['original_img_size']` 基準 | [0, original_img_size] |
| texture-px | px | preview の縮小画像のサイズ | [0, texture_size] |
| window-px | px (DPI 適用後) | `widget.to_window(widget.pos)` 経由 | OS ウィンドウ全体 |

座標系間の変換関数 (params.py):
- `denorm_param`: TCG-norm → TCG-px
- `tcg_to_ref_image(apply_disp_info=False)`: TCG-norm → image-px [0, 2*imax]
- `tcg_to_window(... widget, texture_size, tcg_info)`: TCG-norm → window-px (DPI 適用)
- `window_to_tcg`: window-px → TCG-norm

----

## 2. マスクの生成フロー (preview)

```
primary_param (= 画像 param)
        ↓
MaskEditor2.set_primary_param(primary_param, disp_info)
        ↓
self.tcg_info = params.param_to_tcg_info(primary_param)   # MaskEditor2 自身が tcg_info を保持
self._image_only_matrix = copy(tcg_info['matrix'])         # mask Geometry 用 base
        ↓
mask_editor.set_texture_size(N, N)   # config.get_preview_texture_size() 由来 (常に正方形)
        ↓
[各 frame]
CompositMask.get_mask_image():
    # mask Geometry の matrix swap
    saved = tcg_info['matrix']
    tcg_info['matrix'] = M_mask @ self._image_only_matrix
    try:
        composit = np.zeros((texture_size[1], texture_size[0]), float32)
        for child, op in self.mask_list:
            mimage = child.get_mask_image()   # 各子マスクが自分の tcg_to_texture 経由で描画
            composit = op(composit, mimage)
        composit = _apply_mask_mesh_warp(composit, editor, effects_param)
        return composit
    finally:
        tcg_info['matrix'] = saved
```

要点:
- 子マスクの `tcg_to_texture` は **MaskEditor2 の `self.texture_size`** と **self.tcg_info** を参照
- composit ラスタは **`(editor.texture_size[1], editor.texture_size[0])`** で生成 (常に正方形)
- mask Geometry matrix は CompositMask の get_mask_image 内で一時的に tcg_info に注入され、children に伝播

----

## 3. Mesh widget の coordinate 計算経路

`MeshWarpWidget._redraw_mesh` (mesh_warp_widget.py:329) → `_get_window_pos(tx, ty)`:

```python
def _get_window_pos(self, tx, ty):
    return params.tcg_to_window(tx, ty, self, self.texture_size, self.tcg_info)
```

`params.tcg_to_window(cx, cy, widget, texture_size, tcg_info)` (params.py:771):

```python
imax = max(tcg_info['original_img_size'])/2
cx, cy = denorm_param(tcg_info, (cx, cy))      # → TCG-px
cx, cy = center_rotate(cx, cy, tcg_info)        # rotation + matrix
cx, cy += imax                                  # → [0, 2*imax]
cx -= disp_info[0]; cy -= disp_info[1]          # crop offset
cx *= disp_info[4]; cy *= disp_info[4]          # → preview-px
_, _, offset_x, offset_y = crop_size_and_offset_from_texture(*texture_size, disp_info)
cx += offset_x; cy += offset_y                  # ★ texture_size 依存
cy = texture_size[1] - cy                       # ★ texture_size 依存 (Y flip)
margin_x = (widget.size[0]/dpi - texture_size[0]) / 2   # ★ texture_size & widget.size 依存
margin_y = (widget.size[1]/dpi - texture_size[1]) / 2
cx += margin_x; cy += margin_y
cx *= dpi; cy *= dpi
wx, wy = widget.to_window(*widget.pos)          # ★ widget.pos 依存
cx += wx; cy += wy
return (cx, cy)
```

**`tcg_to_window` の出力は `widget.size`, `widget.pos`, `texture_size`, `tcg_info` のいずれが stale でも壊れる**。

`MeshWarpWidget` 内で `self.texture_size` を使う箇所 (mesh_warp_widget.py):
- L227 `_get_tcg_pos` (touch → TCG)
- L231 `_get_window_pos` (TCG → window)
- L250 `_on_touch_down` の hit test
- L297 `_on_touch_move` の drag
- L333, L335 `_redraw_mesh`

これら全部が **`self.texture_size`** をキャッシュしたまま参照する。

----

## 4. texture_size の伝播 (画像 mesh と マスク mesh の差)

### 4.1 画像 mesh editor

```
main.MainWidget.resize() (line 3147)
    ↓
update_preview_texture_size() → config.set_preview_texture_size(N, N)
    ↓
self.refresh_preview_overlays() (line 694)
    ↓
texture_size = config.get_preview_texture_size()
mask_editor.set_texture_size(*texture_size)             ← MaskEditor2
geometry_effect.update_geometry_editor_texture_size()   ← 画像 mesh widget
    ↓ (effects.py:1523)
self.geometry_editor.set_texture_size(config.get_preview_texture_size())
    ↓ (mesh_warp_widget.py:169)
self.texture_size = texture_size
self._redraw_mesh()
```

→ 画像 mesh widget の `texture_size` は window resize 時に **自動で同期される**。

### 4.2 マスク mesh editor (現状)

```
main.MainWidget.resize()
    ↓
self.refresh_preview_overlays()
    ↓
# mask_editor.set_texture_size(...) は呼ばれる (MaskEditor2 ベースは同期される)
# geometry_effect.update_geometry_editor_texture_size() も呼ばれる (画像 mesh だけ)
# ★ self.mask_mesh_editor.set_texture_size(...) は呼ばれない ← BUG
```

`refresh_preview_overlays` には **マスク mesh widget の同期 hook が無い**。
コンストラクタ時点で `MeshWarpWidget(config.get_preview_texture_size(), ...)` で
渡された値を `self.texture_size` に保存したまま、resize 時に更新されない。

これがユーザー観察「ウィンドウサイズを変えると mesh がずれる」の根本原因。

----

## 5. Sample コードによる検証

`tcg_to_window` の出力が `texture_size` 不一致でどう壊れるか数値で示す:

```python
# Pseudo-trace: ウィンドウリサイズ前後の CP (0, 0) の window 座標を比較

# 初期状態 (起動直後)
tcg_info = {
    'original_img_size': (4000, 3000),
    'disp_info': (0, 0, 4000, 3000, 0.128),   # disp_info[4] = preview/max(orig) = 512/4000
    'rotation': 0, 'rotation2': 0, 'flip_mode': 0,
    'matrix': np.eye(3),
}
texture_size = (512, 512)
widget.size = (512, 512)        # widget = preview_widget サイズ同期
widget.pos = (100, 100)         # 例: preview_widget の絶対位置

# tcg_to_window((0, 0), ...) の中央 CP 計算
cx, cy = 0, 0                                       # input
cx, cy = denorm_param((0, 0)) = (0, 0)              # TCG-px
cx, cy = center_rotate((0, 0)) = (0, 0)             # rotation = 0
cx, cy = 0 + 2000, 0 + 2000 = (2000, 2000)          # +imax
cx, cy = cx - 0, cy - 0 = (2000, 2000)              # crop offset = 0
cx, cy = 2000 * 0.128, 2000 * 0.128 = (256, 256)    # preview scale
offset_x, offset_y = crop_size_and_offset_from_texture(512, 512, disp_info) = (0, 0)
cx, cy = 256, 256                                   # + offset = same
cy = 512 - 256 = 256                                # Y flip
margin = (512/1 - 512)/2 = 0                        # widget.size == texture_size
cx, cy = 256 + 0, 256 + 0 = (256, 256)
cx, cy = 256*1, 256*1 = (256, 256)                  # × dpi = 1
wx, wy = widget.to_window(100, 100) = (100, 100)
cx, cy = 256+100, 256+100 = (356, 356)              # 中央 CP は window (356, 356)

# === ウィンドウを少し大きくして preview_widget が 768x768 になったとする ===
# (window resize → preview_widget.size = (768, 768), 新 preview_texture_size = (768, 768))
# config: set_preview_texture_size(768, 768)
# MaskEditor2.set_texture_size(768, 768) は呼ばれる
# ★ mask_mesh_editor.set_texture_size(...) は呼ばれない
#   → mask_mesh_editor.texture_size = (512, 512) のまま

# 一方、MeshWarpWidget の widget.size は size_hint で size 同期されている
# my impl: pw.bind(size=_sync_layout) で mw.size = pw.size に更新済 → (768, 768)
# tcg_info の disp_info も refresh_preview_overlays 経由で更新済 → disp_info[4] = 768/4000

# 新パラメータで tcg_to_window((0, 0)):
disp_info_new = (0, 0, 4000, 3000, 0.192)           # 768/4000
texture_size_stale = (512, 512)                     # ★ stale
widget.size = (768, 768)
widget.pos = (100, 100)

cx, cy = 0, 0
cx, cy = denorm = (0, 0)
cx, cy = center_rotate = (0, 0)
cx, cy = +imax = (2000, 2000)
cx, cy = cx * 0.192 = (384, 384)                    # 新 disp_info[4]
offset = crop_size_and_offset_from_texture(512, 512, disp_info_new) = ???
# crop_size_and_offset_from_texture は texture_size 基準で内部計算するため stale
# disp_info_new だと texture サイズ 768 を期待するが texture_size_stale=512 で計算する
# → offset 値が大きくズレる、結果として cx + offset が画面外の値になる

cy = 512 - cy                                       # ★ 512 (stale) で flip するので原点ズレ
margin = (768/1 - 512)/2 = 128                      # 余分な margin (本来は 0)
cx, cy = cx + 128, cy + 128                         # 中央 CP が右下に 128px ズレる

→ 中央 CP のはずの (0, 0) が window 上で (356+128, 356+128) ≒ (484, 484) になる
   (本当は (532, 532) なので、ズレ量はさらに複合的)
```

つまり、**`texture_size` 単独の stale でも CP 位置は数百 px 単位でズレる**。
これが「ガイドが画面外に行く」の正体。

----

## 6. その他の stale 候補

下記も同様に widget life-cycle 中に変わりうるが、現状の `MeshWarpWidget` は
コンストラクタ時点のスナップショットを保持し続ける:

| プロパティ | 変動契機 | 同期 hook (画像 mesh) | 同期 hook (マスク mesh) |
|---|---|---|---|
| `widget.size` | parent resize | Kivy auto (size_hint=(1,1)) | 私の `_sync_layout` bind |
| `widget.pos` | parent resize | Kivy auto (pos_hint center) | 私の `_sync_layout` bind |
| `self.texture_size` | window resize | `update_geometry_editor_texture_size` | **無し ← BUG** |
| `self.tcg_info` | crop / Mask Geom / rotation | constructor 時のコピー (= 共有 dict なので参照経由で更新) | 同左 (たまたま参照で OK) |

ただし `self.tcg_info` は `params.param_to_tcg_info(param)` で新規 dict を返すため、
実際は **コンストラクタ時点の primary_param のスナップショット** であり、`primary_param`
が後で更新されると `mw.tcg_info` は古いままになる。

→ 厳密には `tcg_info` も再同期 hook が必要だが、現状の用途では更新頻度が低いので
   実害が出にくい。優先度は texture_size より低い。

----

## 7. 修正方針

### 7.1 最低限の修正 (今回の症状の解消)

`refresh_preview_overlays` (main.py:694) に下記を追加:

```python
# mask Mesh editor の texture_size も同期する
if self.mask_mesh_editor is not None and hasattr(self.mask_mesh_editor, 'set_texture_size'):
    self.mask_mesh_editor.set_texture_size(texture_size)
```

これで window resize 時に画像 mesh と同じく自動同期される。

### 7.2 補強 (堅牢性向上)

- `_disable_mask_mesh_editor` で `update_preview_texture_size(force=True)` + `refresh_preview_overlays()` 呼出は維持 (Mesh モード抜けて画像 mesh に切り替える際の seam 解消)
- `_enable_mask_mesh_editor` で `mw.size_hint = (1, 1)` (= 画像 mesh と同じ流儀) に戻し、`bind(size=_sync_layout)` は撤去するのも検討余地あり。同期は親レイアウト経由で自動になる。

### 7.3 タイミング系 hack の整理 (将来)

現在の `schedule_once × 4` は本質ではなく、texture_size 同期で大半は解決するはず。
ただし「初回マウント時に preview_widget サイズが確定する前に MeshWarpWidget が _redraw_mesh する」問題は別途 race condition として残るので、schedule_once は最低限残す価値あり。

----

## 8. ユーザー指摘の根本仮説への回答

ユーザー仮説:
> 画像サイズと画面サイズの関係が画像meshとマスクmeshで違うことを考慮していない

これは **完全に正しい**。具体的には:

- 画像 mesh widget の `texture_size` (= 画面表示サイズ) は `refresh_preview_overlays` 経由
  で window resize 時に再同期される
- マスク mesh widget には同等の hook が無く、起動時の `texture_size` を保持し続ける
- ウィンドウ拡縮で **画面サイズが変わっても画像サイズ (original_img_size) は変わらず**、
  両者の比 `disp_info[4]` が変動する。`texture_size` だけ stale だと `tcg_to_window`
  が大きくズレる

→ 画像 mesh = 同期される / マスク mesh = 同期されない という非対称が原因。
構造的な問題で、タイミングの問題ではない。

----

## 9. 結論

- ユーザー観察「タイミングではない、画像 mesh と マスク mesh の構造差」は正しい
- 根本原因は `refresh_preview_overlays` で `self.mask_mesh_editor.set_texture_size(...)` を呼んでいないこと
- 7.1 の修正でほぼ解消するはず
- ついでに `tcg_info` 同期も将来課題として残るが、現状の症状には影響しない
