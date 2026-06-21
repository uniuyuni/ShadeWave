# 背景AI-NR (ai_nr_process) ストレス耐性リスク監査

## Context

選択中以外の画像も裏で AI-NR を実行する機能（フォルダ取り込み時／バッチペースト時に
非選択画像を `enqueue_ai_noise_file` で背景キュー投入）が追加された。これにより、
これまで「1枚＝1ジョブ」前提だった `cores/ai_job_manager` が、**多数のジョブが連続・並行的に
流れる**ワークロードに晒される。本書は、ストレス時（大量画像フォルダ・連続画像切替・
頻繁なキャンセル・frozen ビルドのスレッドモード）に**壊れそうなコード上のポイント**を、
深刻度順に列挙する。実装変更ではなく「問題箇所の洗い出し」が目的。

### 背景AI-NR の実行経路（確認済み）
1. `main.py:2428 on_import_path_applied` → **daemon スレッド**で `_enqueue_resumable_ai_noise_jobs_for_folder`
   がフォルダ内の全対象ファイルを走査し、`ai_job_manager.enqueue_ai_noise_file()` を呼ぶ。
2. `main.py:2410 _enqueue_ai_noise_jobs_after_batch_paste` も同様（メインスレッド側）。
3. `manager.py:246 enqueue_ai_noise_file` が BACKGROUND_PRIORITY=100 のファイルジョブを作成。
4. worker (`worker.py:100`) がファイルを実読込→SCUNet→結果 shm を返す。
5. `manager.py:332 poll_results` が COMPLETE を受け、`completed_results` に保存＋
   `main.py:563 _handle_ai_job_result` 経由で `AISidecarMergeQueue` に .pmck マージ投入。

---

## 深刻度: CRITICAL（ストレス下でクラッシュ／OOM の可能性）

### C1. `completed_results` が full-res 画像の無制限キャッシュ（最重要）
- 場所: `manager.py:41`（定義）, `manager.py:364`（書込）, `manager.py:154`（読出のみ）。
- 完了ジョブごとに **full-res float32 配列**を `completed_results[(kind,path,content_key)]` に
  保存し、**一切 evict されない**。フォルダ走査は対象ファイル全件を投入するため、N枚処理 →
  N枚分の full-res 画像がメインプロセスに常駐。24MP×3ch×4byte ≒ 288MB/枚 → 数十枚で OOM。
- `main.py:440 _check_memory_pressure` は `cache_system` のみで、この dict には触れない（確認済み）。
- ストレス誘発: 大量画像フォルダの取り込み。
- 対策候補: LRU/件数上限、または pmck マージ完了後に即 drop（背景ジョブは pmck 永続化が本命で
  メモリ常駐は不要）。foreground 再利用が必要なら「現在画像周辺のみ」に限定。

### C2. `AIJobManager` がスレッド非安全なのに2スレッドから変更される（データ競合）
- 背景ジョブ投入 `enqueue_ai_noise_file` は **フォルダ走査 daemon スレッド**（`main.py:2431`）から、
  `poll_results`/`update_async_results` は **Kivy メインスレッド**（`main.py:404`）から呼ばれる。
- 両者が同じ `self.jobs` / `job_by_target` / `pending_payloads` / `dispatched_job_id` /
  `input_queue` を**ロックなしで**変更（`manager.py` 全体、`grep Lock` → 0件）。
- 具体的破綻: `_dispatch_next_payload` の `min(self.pending_payloads.values())`（`manager.py:293`）が
  別スレッドの変更で `RuntimeError: dictionary changed size during iteration`。
  `dispatched_job_id` の競合で二重 in-flight or 永久スタール。
- 対策候補: マネージャ操作を 1 つの `threading.Lock` で保護、または背景投入をメインスレッドに
  marshal（Clock.schedule_once）。

### C3. メインスレッド外からの Kivy UI 変更
- `manager.py:268 _set_status` → `viewer_state_callback` → `main.py:554 _set_ai_job_viewer_state`
  → `viewer.set_ai_job_state_for_path(...)` を、**フォルダ走査 daemon スレッド**から実行（C2 と同経路）。
- Kivy ウィジェット変更はメインスレッド必須。ストレス時（走査で多数 enqueue）に多発 → 描画破綻/クラッシュ。
- 対策候補: コールバックを `KVClock.schedule_once` 経由でメインスレッドに戻す。

---

## 深刻度: HIGH（リソースリーク／無駄処理）

### H1. キャンセル／未知ジョブの結果 shared_memory リーク
- `poll_results` で COMPLETE メッセージでも `job is None`（`manager.py:344`）や status==CANCELLED
  （`manager.py:346`）の場合、`res["shm_name"]` を **unlink せずに `continue`**。worker は既に
  結果 shm を作成済みなので、毎回 POSIX shm セグメントがリーク。
- ストレス誘発: 連続画像切替によるキャンセル多発。`/dev/shm`（macOS は POSIX shm 上限）枯渇。
- 対策候補: skip する前に必ず result shm を open→close→unlink する共通処理を通す。

### H2. worker 再起動／終了時に未読 result/input shm を取りこぼし
- `_restart_worker` が `result_queue = Queue()`（`manager.py:73`）で差し替え、未読 COMPLETE と
  その結果 shm を破棄。実行中ジョブのキャンセル（`_cancel_job(..., restart_running=True)`）で頻発。
- 対策候補: 再起動前に result_queue をドレインして shm を unlink。

### H3. AISidecarMergeQueue の無制限バックログ＋重い直列マージ
- `submit_ai_noise_result` は無条件 submit、`_pending` 無上限（`sidecar_merge.py:37`）。各エントリが
  **full-res 配列を pin**。merge は `utils.convert_image_to_list`（zstd 圧縮, `ai_noise.py:207`）を
  full-res に対し実行し巨大 .pmck を書く。max_workers=1 で直列。
- ストレス誘発: 連続完了 → メモリ常駐＋CPU 圧迫＋ディスク I/O 嵐＋巨大サイドカー量産。
- 対策候補: 投入数バックプレッシャ、配列参照を merge 直前まで持たない、対象を限定。

---

## 深刻度: MEDIUM（正しさ・競合・性能）

- **M1 プリエンプション無し**: 単一 in-flight（`dispatched_job_id`）で、実行中の重い background
  full-res ジョブが完了するまで、新規 foreground 要求がスタール。foreground 投入は RUNNING の
  background をキャンセルしない（`manager.py:202-222`）。→ 背景処理中の画像選択が数秒固まる。
- **M2 同一パス二重ジョブ**: background が RUNNING 中に foreground `request_ai_noise` が来ると、
  `enqueue_image_job` のどの分岐にも当たらず2つ目のジョブ生成、`job_by_target` を上書き
  （`manager.py:222`）。完了時に誤ったマッピングを pop（`manager.py:367`）。
- **M3 pmck マージの TOCTOU**: stat-token CAS 再チェック（`ai_noise.py:167,174`）は `os.replace`
  と原子的でない。ユーザがその画像へ移動して前景書込した瞬間に background マージが上書き → lost update。
- **M4 スレッドモード（frozen）のゾンビ推論**: `should_use_thread_mode`（`worker.py:157`）が True の時、
  cancel/restart が新スレッドを起こす一方、旧 daemon スレッドは `stop_event` を推論途中で見ないため
  数秒走り続ける（`worker.py:106-155`）。→ UI プロセス内で SCUNet/CoreML が多重実行、過負荷・再入リスク。
- **M5 無駄計算**: enqueue 時に `ai_noise_enabled` 判定するが、worker は重い読込＋推論後に
  merge 側 `_pmck_param_accepts_result` で再判定し discard（`ai_noise.py:187`）。間に NR を切ると全工程が無駄。
- **M6 worker 死亡時の回復穴**: `_mark_dead_worker_jobs_error` は poll 時のみ起動、dispatch 済み
  未消費 payload は死活 worker のキューに消える、auto-restart せず次 enqueue 頼み（`manager.py:391-407`）。

---

## 深刻度: LOW / 観察

- **L1**: `self.jobs` / `status_by_job` / `job_by_target` がファイル単位で session 中に単調増加（小さいが無制限）。
- **L2 既存テストが実機構を stress していない**:
  - `tests/test_ai_nr_headless_stress_flow.py` は**フェイク** `QueuedThenCompleteAIManager` を使用し、
    shm/worker/sidecar/スレッドを一切経由しない。
  - `tests/test_ai_job_manager.py` は単一スレッドの単機能検証のみ。
  - → 並行 enqueue、`completed_results` 増加、cancel 時 shm リーク、off-thread コールバック、
    sidecar バックプレッシャは**どれも未カバー**。

---

## 推奨される確認・テスト（次フェーズ候補）

- 並行ストレステスト: メインスレッドで `poll_results` ループしつつ、別スレッドから
  `enqueue_ai_noise_file` を多数同時投入し、`dictionary changed size` / 二重 dispatch / クラッシュを再現。
- メモリ計測テスト: N 件 background 完了後の `completed_results` バイト数を assert（上限超過を検出）。
- shm リークテスト: cancel 多発シナリオ後に `/dev/shm`（または POSIX shm 名）残数を検証。
- off-thread コールバックテスト: `viewer_state_callback` がメインスレッド以外から呼ばれないことを assert。
- 検証コマンド例: `python -m pytest tests/test_ai_job_manager.py -q` に上記新規ケースを追加。

## 次のアクション（要確認）
本書はリスク洗い出しまで。実装（修正 or ストレステスト追加）に進むかは別途確認する。
