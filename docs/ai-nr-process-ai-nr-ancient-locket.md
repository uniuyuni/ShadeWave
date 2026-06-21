# 背景AI-NR (ai_nr_process) ストレス耐性レビュー（修正後）

## Context

選択中以外の画像も裏で AI-NR を実行する機能（フォルダ取込／バッチペースト時に非選択画像を
`enqueue_ai_noise_file` で背景投入）について、当初監査で CRITICAL 3点（C1/C2/C3）を指摘。
ユーザーがコミット **767a1f7「Harden AI job scheduling and lensfun parsing」** で「クリティカルのみ」
修正したとのことで、本書はその**修正後コードのレビュー結果**（残存問題と推奨対応）をまとめる。

対象: `cores/ai_job_manager/manager.py`, `worker.py`, `main.py`, `widgets/viewer.py`, `threads.py`。

---

## 修正済み（妥当に解決）

- **C1 `completed_results` 無制限キャッシュ → 解決**
  `OrderedDict` 化＋バイト予算（env `PLATYPUS_AI_COMPLETED_CACHE_MAX_MB`, 既定512MB）＋
  `_enforce_completed_results_budget` の退避＋単体上限超過は非キャッシュ＋`_prune_finished_jobs`(保持64)。
  `manager.py:234-281, 685-704`。
- **C2 スレッド非安全 → 解決**
  `threads.ai_job_manager_lock` は **RLock**（`threads.py:36`）。全 mutator が `with self._lock`。
  ネスト獲得（`enqueue_image_job`→`start` 等）も自己デッドロックしない。`primary_param_lock` との
  AB-BA も成立せず（コールバックが同期で param ロックを取らないため）。
- **C3 メインスレッド外 UI 変更 → 解決**
  `set_ai_job_state_for_path` に **`@kvmainthread`**（`widgets/viewer.py:898`）。daemon スレッドからの
  呼び出しでもメインスレッドへマーシャルされ、`self.data` 変更と `refresh_from_data()` は main 上で実行。
- **おまけ: H1/H2 result shm リーク → 解決**
  `_discard_result_shm` / `_drain_result_queue_discard_shm`（`manager.py:104,525,528,647-666`）。
  協調キャンセル（thread mode の cancel_event＋scunet cancel_callback）、STALE、progress、
  dead-worker 復旧も追加済み。
- **restart カウンタ枯渇 / dispatch wedge → 解決**
  COMPLETE 受信時に `worker_restart_count` を reset。intentional preempt は crash 復旧予算に数えず、
  CANCELLED result 分岐でも `dispatched_job_id` をクリアして次 payload を dispatch する。
- **preempt で落とした background ジョブの消失 → 解決**
  pending background file job を foreground が preempt した場合、同じ file job を再投入する。
- **completed result cache の LRU 化 / バイト数再集計削減 → 解決**
  hit 時に `move_to_end` し、総バイト数は `completed_results_total_bytes` で増減管理する。
- **`poll_results` の full-res copy 中ロック保持 → 緩和**
  COMPLETE result の共有メモリ `arr.copy()` は manager lock 外で行い、コピー後に短く lock を取り直して
  cache/store/status 更新する。
- **未使用 import → 解決**
  `manager.py` の未使用 `multiprocessing` import を削除。

---

## 残存問題（要対応）

### 【高 / HIGH】process-mode preempt が毎回 CoreML 再ロード
process mode は単一ジョブ cancel 不可で worker を kill/respawn → `run_ai_noise._engine`
（プロセス毎, `worker.py:53-54`）喪失で **SCUNet を毎回ロードし直す**。連続ナビで数百ms〜秒級スタール多発。
現在は壊れないように intentional preempt を crash restart budget から分離済み。ただし worker restart 自体は残るため、
速度面の根本改善には process mode の協調キャンセル設計が必要。

### 【中 / MEDIUM】
- **progress 毎の `@kvmainthread` コールバック過多** — tile progress を細かく出しすぎると UI update が多くなる。
  必要なら rate limit を入れる。

### 【低 / LOW】
- **`_prune_finished_jobs` を poll 毎に全走査** — 想定 n では実害小。気になる場合は finished queue 化する。

### （既知・今回対象外）
- **H3 サイドカーマージ**（`sidecar_merge.py` 未変更）— 無制限バックログ＋full-res zstd を単一スレッド直列。
  大フォルダで次に効いてくる。

---

## 推奨対応スコープ（次フェーズ）

- 残りの大物: process-mode preempt 時の CoreML 再ロード抑制。
- 次の低リスク改善: progress callback の rate limit と `_prune_finished_jobs` の軽量化。

## 検証

- 回帰: `pixi run python -m unittest discover -s tests -p 'test_ai_job_manager.py'`
- 追加すべきストレステスト:
  - **restart 枯渇**: process-mode mock worker で RUNNING を 3回 preempt し、3回目以降も新規 foreground が
    ディスパッチされ続ける（`dispatched_job_id` がウェッジしない／pending が ERROR 全滅しない）ことを assert。
  - 並行 enqueue（main スレッド poll ＋ daemon enqueue）でクラッシュ無し、`completed_results` がバイト予算内。
