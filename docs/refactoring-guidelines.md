# Shade Wave リファクタリング指針

## Context（なぜこの指針が必要か）

Shade Wave（v2.14.19）は Kivy ベースの RAW/写真編集アプリで、長期間の機能追加により以下の構造的負債が蓄積している：

- **god file 化**: `effects.py`（4963行/64エフェクトクラス）、`main.py`（`MainWidget` 単一クラスに213メソッド）、`widgets/mask_editor2.py`（5295行）、`cores/core.py`（128関数）、`main.kv`（215KB）
- **大量の定型重複**: 各 Effect が `get_param_dict`/`set2widget`/`set2param`/`make_diff` をほぼ同型でコピペ（58〜62回反復）
- **密結合**: `effects ↔ pipeline ↔ params` の循環 import、グローバル可変状態 `config._config`（multiprocessing Manager dict）を20+モジュールが参照
- **並行性の繊細さ**: マルチプロセスworker + SharedMemory + `threads.py` の4ロックに、明文化されていない不変条件がある
- **型・契約の弱さ**: 型ヒントほぼゼロ、param は schema なしの plain dict が「ワイヤフォーマット」

本ドキュメントは **「保守性・可読性の改善」と「パフォーマンス」を主目的** とし、機能追加・将来の自分の作業を楽にするために、リファクタ全般で守るべき原則・気をつけるべき点・アンチパターン回避策を定める。**一括書き換えではなく、振る舞いを変えずに少しずつ改善する**ことを前提とする。

---

## 0. 最優先の大原則（すべてに優先）

1. **振る舞いを変えない**。リファクタは「外から見た出力・タイミングを保ったまま内部構造を整える」作業。最適化・整理と機能変更を同じコミットに混ぜない。
2. **回帰テストを先に確保してから動かす**。テストのない箇所はリファクタ前に特性化テスト（characterization test）を1本書く。`test_edge_refine.py` / `test_headless_real_pipeline_flow.py` が良い手本。
3. **小さく刻む**。1コミット=1責務。god file はファイル全体ではなく「1クラス群／1機能の抽出」単位で進める。レビュー（自分の差分確認）が効く粒度を保つ。
4. **画像の数値同一性で検証する**。UIではなく headless パイプラインで before/after の出力配列を突き合わせる（許容誤差付き）。これが本アプリのリファクタの生命線。

---

## 1. 気をつけるべき点（高リスク領域・触る前に必読）

これらは「壊すと気づきにくい」場所。変更時は専用の検証を用意する。

### 1.1 並行性・SharedMemory（最危険）
- `threads.py` のロックには**役割と再入性の意味**がある：`primary_param_lock`(RLock=描画スレッドとUIの param/imgset 共有)、`mask_editor_matrix_lock`(RLock)、`numba_lock`、`editor_lock`。**ロックの境界・粒度・取得順序を変えない**。デッドロックとレースは headless テストで再現しにくい。
- `async_worker.py` の SharedMemory ライフサイクル（create→open→close→unlink、`active_shms`、`latest_tasks` でのキャンセル）は**契約が暗黙**。内部実装を変えても**キュー/SHMの受け渡し契約（誰がunlinkするか）は厳守**。リーク=RAM肥大、二重unlink=クラッシュ。
- `@kvmainthread` デコレータ（main.py 6箇所）と `Clock.schedule_once/interval`(`interrupt` モード)に依存したタイミングがある。**スケジュール呼び出しの順序・遅延の除去や統合は安易にやらない**。

### 1.2 循環 import
- `effects.py:31 → import pipeline`、`pipeline.py:14 → import effects`、`params.py → import effects` が成立しているのは「`import module` 形式＋実体化が全import後」という綱渡りのため。**`from x import y` への変更や、モジュールレベルでの相互呼び出し追加は初期化を壊す**。依存を切る方向（下記5章）でのみ触る。

### 1.3 グローバル可変状態
- `config._config`（Manager dict、プロセス跨ぎ共有）と `_main_widget` 参照は全体に染み出している。`_apply_config()` がUI副作用を持つ。**「設定を読むだけ」のつもりが副作用やプロセス間同期に波及**しうる。getter/setter 経由を崩さない。
- `cores/core.py` の `_lensfun_db_instance` 等の遅延キャッシュは**スレッドセーフでない**。マルチスレッドから触る箇所を増やさない。

### 1.4 param dict と永続化
- param は実行（pipeline）と永続化（`.pmck`/msgpack）の両方に使う**スキーマレス dict**。キー名・型・デフォルト値の変更は**保存済みサイドカーとの後方互換**に直結。`SPECIAL_PARAM`/`DO_NOT_COPY_SPECIAL_PARAM`/`REMAIN_PARAM`/`HEAVY_PRIMARY_PARAM_KEYS`（params.py）の意味を保つ。重い payload の条件付きシリアライズ（`_param2_has_substantive_heavy_payload`）を壊さない。

### 1.5 エフェクトのキャッシュ無効化
- 各 Effect は `diff`/`hash` による結果キャッシュを持つ。`make_diff` を整理する際、**ハッシュ計算に含めるパラメータ集合を変えると、変更が反映されない/無駄に再計算するバグ**になる。ハッシュ対象は機械的に保つ。

---

## 2. god file の分割指針（保守性の本丸）

**「物理的にファイルを割る」前に「責務を見つけて束ねる」**。行数削減自体が目的ではない。

- **effects.py**: 巨大な原因は64クラス×4定型メソッドの重複（3章で対処）。分割するなら**機能カテゴリ単位**（色温度系/トーン系/ノイズ系/AI系…）でサブモジュール化し、起動時の `effects = [{}×5]` 登録だけを集約点に残す。ディスパッチは現状の「dict[level][name]→ポリモーフィズム」が良い設計なので**維持**。
- **main.py の `MainWidget`(213メソッド)**: 責務で mixin / 別オブジェクトに割る。観測された自然な境界：
  - ファイル/ロード調整（`_load_stage_*`）
  - プレビュー/テクスチャ描画（`blit_image`, `update_preview_texture_size`, ヒストグラム）
  - マスクエディタ連携（`_finish_ai_inpaint_mask_mode` 等）
  - async worker 結果の受領（`update_async_results`）
  - メモリ圧検知（`_check_memory_pressure`）
  まず**状態を持たない純粋ヘルパ**から外に出すのが安全。Kivy widget 階層を壊さないため、いきなり継承構造を変えない。
- **mask_editor2.py**: 8マスク型に散らばる `on_touch_down/move/up`・`serialize/deserialize`・幾何変換（`_matrix_transformed_ellipse` 等）を**共通 mixin / ヘルパ関数**へ。`get_mask_image` が3バリエーション存在する → 統合候補だが**出力同一性をテストで固定してから**。
- **core.py**: クラス無しの関数群。色空間変換チェーン（RGB↔HLS↔XYZ）の重複を**小さな純関数に括り出す**。numba `@njit` の適用が不統一 → ホットパスを計測してから方針統一（4章）。
- **main.kv（215KB）**: 巨大KVは `#:include` で画面/パネル単位に分割。ただし id 参照と `MainWidget` メソッドのバインドが密なので、**分割は機能ブロック単位で1つずつ**。

---

## 3. 重複排除：Effect 定型の共通化（最大の効果）

`get_param_dict`/`set2widget`/`set2param` が約60回コピペされている。ここが**最も費用対効果が高い**。

- **宣言的なパラメータ定義**へ寄せる。各 Effect が「パラメータ名・デフォルト値・widget 名」の**テーブル（dict/dataclass/descriptor）を1つ宣言**し、基底クラスの `get_param_dict`/`set2widget`/`set2param` が**そのテーブルを駆動**する Template Method にする。差分ロジック（`make_diff`）だけを各クラスが書く形へ。
- 移行は**全クラス一斉ではなく数クラスずつ**。基底の汎用実装を入れ、個別の定型メソッドを削除しても**`get_param_dict()` の戻り値が完全一致**することをテストで担保（既存値を dump して比較）。
- 期待効果：effects.py から推定1500行以上の定型を除去、新エフェクト追加コストの激減。**振る舞いは不変**。

---

## 4. パフォーマンス指針（主目的の一つ）

**計測なき最適化をしない**。本アプリには既に計時の足場がある。

- `pipeline.py` の `_PIPELINE_TIMING_*`／`_timing_record_effect` と `_DEBUG_PIPELINE_STATS` を活用し、**まずホットスポットを実測**（どの level/effect が支配的か）。推測で numba 化や配列コピー削減をしない。
- **重複計算の削減**を優先：同一 param で再計算される `make_diff`、毎回の color space 変換、`type_convert` の多重呼び出し（core.py に頻出）。キャッシュは既存の `diff`/`hash` 機構に乗せる（独自キャッシュを増やさない）。
- **配列のコピー/dtype 変換**を減らす（in-place 可能箇所、float32 統一）。ただし param/imgset の共有配列はロック保護下にあるので in-place 化は副作用に注意。
- numba `@njit` は**ホットかつ純粋な数値ループに限定**。`numba_lock` 直列化のコスト（JIT初回）も考慮。
- GPU/MPS パス（`gpu_device`、SCUNet、depth_pro の `device='mps'` ハードコード）は**設定経由に寄せつつ、CPU フォールバックの数値差**に注意。出力同一性テストは GPU 有無で別基準にする。
- 最適化PRも**1.2の出力同一性検証**を必ず通す（許容誤差を明示）。

---

## 5. 結合度を下げる指針

- **循環の解消方向**：共有される定数・型・軽量ユーティリティを依存の末端（新規 `params_types.py` など、または既存 leaf）へ移し、`effects`/`pipeline`/`params` が互いの**実体**を import しなくて済むようにする。インターフェース（基底 `Effect`、param の型）を中立な場所に置く。
- **グローバル設定の縮小**：`config.get/set` 経由を保ちつつ、新規コードでは**設定を引数で受け取る**（暗黙のグローバル参照を増やさない）。テスト容易性（headless config 注入）にも効く。
- **param を構造化**：plain dict のまま全廃は非現実的だが、**新規・改修箇所では TypedDict / dataclass で「読み取りビュー」を作り**、キーtypoと型を静的に検出できるようにする。永続化フォーマット（dict）自体は当面維持。

---

## 6. 可読性の地ならし（低リスク・先行着手可）

破壊リスクが低く、後続リファクタの土台になるもの。**ここから始めてよい**。

- **型ヒントの段階導入**：まず公開API・関数シグネチャ（core.py の主要関数、pipeline のエントリ）から。`Dict`/`Any` の形だけでなく意味のある型を。mypy/pyright を緩い設定で導入し**新規コードのみ厳格化**。
- **エラー処理の改善**：bare `except:`（12箇所、SHM/IPCクリーンアップ等）は**捕捉例外を具体化**し、握り潰す場合も `logger.debug` で痕跡を残す。ただし**クリーンアップ系の握り潰しは意図的なものもある**ので、消す前に「なぜ広く捕まえているか」を確認。
- **print→logging**：散在する37の `print()` を per-module `logger`（pipeline.py が手本）へ。
- **コメントの言語方針**：英語コード＋日本語コメントの混在は現状維持でよい（統一は低優先）。ただし**新規の公開APIには簡潔なdocstring**を。
- **デッドコード/コメントアウト**は、関連箇所を触るついでに削除（履歴は git にある）。先回りの一括削除はしない。

---

## 7. テストとの付き合い方（リファクタの安全装置）

- 既存は `unittest`（64ファイル、pytest設定なし）。**新規にpytest化は不要**、`python -m unittest discover` で回す。
- リファクタ対象に回帰テストが無ければ、**着手前に特性化テストを1本**：headless config（`install_headless_config` パターン）で実パイプラインを通し、出力配列をスナップショット比較。
- effect 共通化（3章）は **`get_param_dict()` 値ダンプの一致**を専用テストで固定。
- 並行性（1.1）を触る場合のみ、**ロック競合・SHMリークの簡易ストレス**（`test_file_selection_memory_stress.py` 流用）を追加。

---

## 8. 進め方（運用ルール）

1. 触る領域に回帰/特性化テストがあるか確認 → 無ければ書く。
2. 1責務だけ変える小コミット。`define.py` の VERSION 方針（GEMINI.md）に従いバージョン更新。
3. headless 出力同一性で検証（許容誤差明示）。UIは最後に実機確認。
4. 「整理」と「機能変更」「最適化」はコミットを分ける。
5. 1.1〜1.5 の高リスク領域は、変更ごとに専用検証を必ず添える。

---

## 検証方法（リファクタ前後で共通に回す）

```bash
# 全テスト（プロジェクトルートで）
cd /Users/uniuyuni/PythonProjects/ShadeWave
python -m unittest discover -s tests -p 'test_*.py'

# headless 実パイプライン（出力同一性の要）
python tests/test_headless_real_pipeline_flow.py

# エッジ精細化の回帰ハーネス（重い変更時）
python tests/test_edge_refine.py

# ファイル選択メモリストレス（並行性/SHMを触った時）
python tests/test_file_selection_memory_stress.py
```

- パイプライン計時：`pipeline.py` の `_PIPELINE_TIMING_*` / `_DEBUG_PIPELINE_STATS` 環境変数を有効化し、最適化前後で level/effect 別の時間を比較。
- 出力同一性：headless で before/after の処理結果配列を `np.allclose`（許容誤差を明示）で突き合わせる。GPU/MPS 経路は別基準。
- 実機確認は最後に：`/run` でアプリ起動し、代表的な編集操作（読み込み→補正→マスク→書き出し）が壊れていないか目視。

---

## まとめ（優先順位の目安）

1. **先行・低リスク**：6章（型ヒント／logging／bare except 具体化）と、各領域の特性化テスト整備。
2. **最大効果**：3章（Effect 定型の共通化）— 保守性に直結、振る舞い不変で進めやすい。
3. **構造改善**：2章（god file 分割）と5章（循環/グローバル削減）— 小さく刻む。
4. **パフォーマンス**：4章 — 計測駆動で、出力同一性を守りつつ。
5. **常時厳守**：0章の大原則と1章の高リスク注意点。
