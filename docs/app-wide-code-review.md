# Platypus / Shade Wave 総合コードレビュー

作成: 2026-06-22 ／ 対象: vendored（`.pixi/` `external/` `build/`）を除く実プロジェクトコード（約55–60K LOC）
方法: 構造レベルの高シグナル横断スキャン（重複関数・アンチパターン・責務配置）＋中核ファイルの精査。
※ 行単位の網羅ではない。深掘り候補は末尾参照。

## 優先度サマリ

| # | 種別 | 重要度 | 概要 | 位置 |
|---|------|--------|------|------|
| A | 正しさ/セキュリティ | 高 | `eval()` を永続化パラメータに実行 | effects.py:1954 |
| B | アーキテクチャ | 高(大工事) | main.py の god 構造（import 不可＋234メソッド god class） | main.py |
| C | テスト健全性 | 中 | B が誘発するテストの実装文字列依存（27ファイルで複製） | tests/ |
| D | 重複 | 中 | effect_backends アダプタ 9本×共通I/F（1339行） | effect_backends/*_adapter.py |
| E | 重複 | 中 | helpers モデルヘルパ 5–6本×共通I/F | helpers/*_helper.py |
| F | 重複/罠 | 中 | `read_pmck_dict` 同名・異挙動 | preset_utils.py:166 / rating_io.py:68 |
| G | 重複 | 低 | util 小重複（gaussian_blur_cv/apply_lut 等） | core.py 他 |
| H | 堅牢性 | 中 | 握りつぶし `except: pass` 124箇所 | main.py 22 他 |
| I | 小バグ | 低 | mutable default 引数 ×2 | facer_helper.py:31 / distortion_painter.py:492 |
| — | 既知 | 高 | AIJobManager restart wedge（別途対応中） | ai_job_manager/manager.py |

---

## 1. 正しさ / セキュリティ

### A. `eval()` を永続化パラメータに実行 — `effects.py:1954`
```python
def _param_to_aspect_ratio(self, param):
    ar = self._get_param(param, 'aspect_ratio')
    return eval(ar if ar != "None" else "0")
```
- `aspect_ratio` はスピナー文字列（"16/9" 等）として `param['aspect_ratio']` に格納され（effects.py:1985, 2022）、**param は `.pmck` に永続化→再ロードされる**。
- 通常はスピナー固定値で安全だが、`eval` は「ファイル由来文字列の実行」経路になり得る（手編集／共有 .pmck に `__import__('os').system(...)` 混入可）。不正値で汎用 eval エラーで停止する脆さもある。
- **対応**: 安全パーサに置換。
```python
def _param_to_aspect_ratio(self, param):
    ar = self._get_param(param, 'aspect_ratio')
    if not ar or ar == "None":
        return 0
    try:
        if "/" in ar:
            n, d = ar.split("/", 1)
            return float(n) / float(d)
        return float(ar)
    except (ValueError, ZeroDivisionError):
        return 0
```

---

## 2. アーキテクチャ / 責務の所在（「それ呼び出し側でやるの？」）

### B. `main.py` の god 構造 — `main.py:1, 208-221, 356`
- アプリ全体が `if __name__ == '__main__':` 配下に定義。`if __name__ != '__main__':` 側は **`ImportBlocker`**（main.py:209-221）で通常 import を阻止 → **main.py を素直に import できない**。
- 中核 **`MainWidget(MDBoxLayout)` が 234 メソッド** の god class（main.py:356）。UI ウィジェットが低レベル処理を直接実行:
  - `np.*` ×33、`cv2.*` ×3、`open()` ×5、`os.path/remove/rename/replace` ×15、`Thread` ×5。
- 画像演算・ファイル削除/改名・スレッド管理は本来 `cores/` / `utils/` のサービス層の責務。UI に集約され、テスト・再利用・並行安全を阻害。
- **対応（段階的）**: まず「ファイルI/O」「履歴操作」「画像処理呼び出し」を `MainWidget` から薄いコントローラ/サービスへ抽出。一気の分割は不要、小ステップで。

### C. B の副作用 — テストが実装文字列に密結合
- `_load_class_function` が **27 テストファイルで複製**、`_function_source`/`_load_function`/`_class_source` も多数。
- main.py を import できないため、各テストが**正規表現/AST で関数ソースを抽出して `exec`**。実装の文字列表現に依存し脆い。
- **対応**: B を解けば、この複製は共有 util 1つ、ないし通常 import に置換可能。少なくとも今すぐ `tests/` 共有ヘルパへ一本化できる。

---

## 3. 重複コード

### D. `effect_backends/*_adapter.py` ×9（計 1339 行）— 最大の重複源
- 各アダプタが `native_available` / `backend_status` / `_backend_preference` / `native_enabled` / `_native_strict` / `_metal_device_available` / `_metal_backend_enabled` を再実装。
- 行数: image_transform 216 / low_frequency_transfer 280 / colour_functions 162 / cross_filter 162 / color_separation 128 / film_grain 113 / tone 109 / subpixel_shift 99 / vignette 70。
- **対応**: `BackendAdapter` 共通基底（バックエンド選択・strict 判定・status 一元化）へ。大幅縮小可。

### E. `helpers/*_helper.py` ×5–6 — モデルヘルパ I/F 重複
- `setup` / `predict` / `predict_helper` / `_soft_edit_mask` / `_ensure_result_size` が scunet/nafnet/nano_banana/qwen/runware で重複。
- **対応**: `ModelHelper` 基底 or Protocol に集約。

### F. `read_pmck_dict` 同名・異挙動 — `preset_utils.py:166` / `rating_io.py:68`
- preset_utils 版: `read_path(..., default_empty=True)` ＋ `ensure_primary_param`（非None保証・primary_param 整形）。
- rating_io 版: `read_path(pmck_path)` のみ（`Optional`・生データ）。
- **同名だが契約が異なる罠**。実体は `pmck_store` に集約済みなので、片方を改名（例 `read_pmck_dict_or_none`）して衝突解消。

### G. util 小重複（検証のうえ統合）
- `gaussian_blur_cv` ×3、`apply_lut` ×3、`adjust_tone` ×3、`smoothstep` ×2、`release_ai_model_runtimes` ×2。
- 一部は `*_reference.py`（native の参照実装）で意図的重複。`core.py`↔reference 間の genuine 重複のみ共通 util へ。

---

## 4. 堅牢性 / 小バグ

### H. 握りつぶし `except …: pass` 124箇所（非テスト）
- hotspots: main.py 22 / params.py 14 / macos.py 14 / mask_editor2.py 10 / async_worker.py 8 / preset_utils.py 6 / viewer.py 4 / memory_manager.py 4 / export.py 4 / ai_job_manager/manager.py 4。
- ストレス時の不具合切り分けを困難にする。**対応**: 最低限 `logging.exception(...)` を残し、握りつぶす根拠をコメント化。

### I. mutable default 引数 ×2
- `helpers/facer_helper.py:31` `def draw_face_mask(faces, exclude_names=[])`
- `widgets/distortion_painter.py:492` `def __init__(self, ..., recorded=[], ...)`
- インスタンス間で共有される古典バグ。`=None` ＋本体初期化に。

---

## 5. 既知（前段までの調査）
- **AIJobManager restart wedge**（`worker_restart_count` 非リセット＋CANCELLED 分岐の `dispatched_job_id` 取りこぼし）— 別途対応中。
- **H3 サイドカーマージ** 無制限バックログ＋full-res zstd 直列（`sidecar_merge.py`）。

---

## 推奨着手順
1. **A**（eval 撤去）— 小修正・影響大。
2. **D / E**（アダプタ・ヘルパ基底化）— 重複最大塊、保守コスト即効。
3. **B / C**（main.py 責務分割）— 最も効くが大工事。ファイルI/O→履歴→画像処理の順に小さく抽出。
4. **F / H / I** — 低コストで踏みやすい罠を順次。

## 深掘り候補（未実施）
行単位レビュー対象として: `effects.py`(4787) / `widgets/mask_editor2.py`(5748) / `cores/core.py`(2770) / `pipeline.py`(1495) / `params.py`(1006)。指定あれば個別に精査する。
