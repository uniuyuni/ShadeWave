# GPU Display Color Management Investigation

## 背景

Mask Geometry 操作時のプロファイルログでは、表示直前の色変換が大きな遅延要因になっていた。

- 通常表示変換: `color_ms` が約 88-120ms
- ヒストグラム: `hist_ms` が約 27-38ms
- 簡易 fast display: `color_ms` が約 20ms 前後まで低下

このため、CPU 側で毎フレーム `colour_functions.RGB_to_RGB()` を走らせる代わりに、内部処理画像を ProPhoto/linear のまま保持し、表示変換を OS / ICC / GPU 側へ移せないかを調査した。

## 調査結果

### Kivy Texture は ICC profile を持たない

Kivy 2.3.1 の `Texture` は OpenGL texture で、指定できるのは `rgb`, `rgba` などの pixel format と、`ubyte`, `float` などの buffer format である。`Texture.blit_buffer()` は raw buffer を GPU texture へ upload する API で、texture に ICC profile や色空間タグを付ける仕組みは見当たらない。

参照:

- [Kivy Texture documentation](https://kivy.org/doc/stable/api-kivy.graphics.texture.html)

ローカル環境でも Kivy は `2.3.1`、Window provider は `sdl2` だった。Kivy source 内を検索した限り、ICC / ColorSync / display profile を表示変換に使う実装は見当たらなかった。

### SDL2 は ICC profile を取得できるが、変換はしない

SDL2 には `SDL_GetWindowICCProfile()` があり、ウィンドウが表示されている画面の raw ICC profile を取得できる。ただしこれは profile data の取得 API であり、OpenGL texture の pixel 値を自動変換してくれるものではない。

参照:

- [SDL2 SDL_GetWindowICCProfile](https://wiki.libsdl.org/SDL2/SDL_GetWindowICCProfile)

### OpenGL は基本的にアプリ側で色管理する

Apple の TN2313 では、Quartz / Core Image / Core Animation / AV Foundation / AppKit のような ColorSync 統合 framework は color managed だが、低レベル framework を使う場合は描画前に明示的に色管理する必要がある、という扱いになっている。OpenGL については、gamma / matrix / ColorSync recipe / 3D LUT を shader に実装する選択肢が説明されている。

参照:

- [Apple TN2313: Best Practices for Color Management in OS X and iOS](https://developer.apple.com/library/archive/technotes/tn2313/_index.html)

### Windows の自動色管理も Kivy/OpenGL にそのまま期待しにくい

Windows の Advanced Color では OS 側で色空間変換を行う仕組みがある。ただし通常の ICC workflow では、アプリが display profile を取得し、色空間変換と gamut mapping を行う前提で説明されている。Kivy の OpenGL texture を ProPhoto として OS にタグ付けして自動変換させる経路は見つからなかった。

参照:

- [Microsoft: ICC profile behavior with Advanced Color](https://learn.microsoft.com/en-us/windows/win32/wcs/advanced-color-icc-profiles)

## 結論

現状の Kivy / SDL2 / OpenGL 経路では、内部画像を ProPhoto のまま `Texture.blit_buffer()` して、OS / ICC に表示変換を丸投げするのは難しい。

実用的な方向は、OS 丸投げではなく **GPU shader へ表示色変換を移す** こと。

## 簡易設計

### 目的

プレビュー表示時の CPU 色変換コストを削る。

現在:

```text
pipeline output(ProPhoto/linear float)
  -> CPU colour_functions.RGB_to_RGB()
  -> sRGB/display encoded float
  -> Texture.blit_buffer()
  -> Kivy/OpenGL draw
```

提案:

```text
pipeline output(ProPhoto/linear float)
  -> Texture.blit_buffer()
  -> fragment shader で display 変換
  -> Kivy/OpenGL draw
```

### 第1段階: Matrix + Transfer Curve Shader

対象は preview 表示だけ。export や保存処理は既存の高品質 CPU 変換を維持する。

実装イメージ:

1. `draw_image_core()` では preview texture 用に ProPhoto/linear float を upload する。
2. `Preview` 表示 widget を `RenderContext` などの shader 対応 canvas にする。
3. uniform に `src_to_display_matrix` と transfer curve mode を渡す。
4. fragment shader で:
   - ProPhoto/linear RGB に 3x3 matrix を適用
   - 負値を clamp
   - sRGB / gamma 2.2 / gamma 1.8 などに encode
5. histogram は従来通り CPU 側で必要なときだけ計算する。

利点:

- 今の fast display と同等の処理を GPU へ移せる
- UI 操作中だけでなく通常 preview も軽くできる可能性がある
- 実装範囲が preview 表示面に限定される

制約:

- `colour_functions.RGB_to_RGB(... apply_gamut_mapping=True)` と完全一致はしない
- display ICC の複雑な LUT / TRC / gamut mapping は再現しない
- texture upload のコストは残る

### 第2段階: 3D LUT Shader

より正確な表示変換が必要なら、CPU 側で display transform を 3D LUT 化し、shader で trilinear sampling する。

実装イメージ:

1. `src_space`, `display_color_gamut`, `cat`, rendering intent 相当の設定から 3D LUT を生成する。
2. LUT は display 設定が変わるまで cache する。
3. shader で ProPhoto/linear RGB を LUT lookup して表示値へ変換する。

利点:

- matrix + transfer curve より CPU 高品質変換に近づけやすい
- display profile の非線形 TRC や gamut mapping をある程度表現できる

制約:

- Kivy/OpenGL ES 環境で 3D texture が安定して使えるか確認が必要
- 3D texture が難しい場合、2D tiled LUT で代替する
- LUT 生成コストと cache invalidation 設計が必要

### 第3段階: Display ICC Profile 対応

SDL2 の `SDL_GetWindowICCProfile()` や OS API で display ICC を取得し、ColorSync / LittleCMS などで shader 用 recipe または LUT を作る。

注意点:

- Kivy から SDL window handle / ICC profile 取得 API に触れる経路の確認が必要
- マルチディスプレイ移動時に display profile が変わる
- OS ごとの差が大きい
- ここは最初から狙わず、第1段階が効くことを確認してから検討する

## 実装候補

### 候補 A: Preview 専用 Shader Widget

現在の `ids["preview"]` に texture を渡している箇所を、表示用 shader を持つ widget に置き換える。

影響範囲:

- `main.py`
- `main.kv`
- preview widget 周辺

リスク:

- Kivy の canvas / `RenderContext` と既存 overlay / transform wrapper の相性確認が必要
- texture orientation / `flip_vertical()` / crop wrap 表示との整合確認が必要

### 候補 B: 既存 Image/Rectangle の Shader Context 化

既存の preview 表示構造を大きく変えず、該当 canvas だけ shader context にする。

影響範囲:

- `main.kv`
- preview 描画 instruction 周辺

リスク:

- Kivy の標準 `Image` widget が期待する shader と衝突する可能性がある
- overlay との描画順を壊さないように検証が必要

## 推奨する次の一手

まず第1段階の matrix + transfer curve shader を小さく試す。

検証条件:

- 同じ画像で CPU fast display と shader display の差分を比較する
- sRGB / Display P3 / Adobe RGB 設定で破綻しないか確認する
- Mask Geometry ドラッグ中に `color_ms` がほぼ消えるか確認する
- overlay / CP / axis の描画順が崩れないか確認する

この段階で効果が小さい場合、主因は texture upload / mask raster / pipeline cache の方に移るので、低解像度 preview や mask preview 専用 route を優先する。
