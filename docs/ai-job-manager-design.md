# AI Job Manager Design

## Goal

AI-NR を現在の表示パイプライン用 async task から切り離し、画像ファイルに紐付いた独立 background job として実行する。

目的は次の3つ。

- 他の画像を選択しても AI-NR を中断しない。ただしユーザー明示キャンセルは効く。
- 複数ファイルで AI-NR を実行できる。ただし実行は1ファイルずつ。
- AI-NR ON の設定を複数ファイルへコピーしたとき、対象ファイルの AI-NR を裏で順次実行する。

将来的に AI-NR 以外の AI 処理も載せるため、AI-NR 専用の ad hoc queue ではなく、job kind を差し替えられる汎用構造にする。

## Current Problem

現在の AI-NR は `AINoiseReductonEffect` から既存の `AsyncWorker` / `AsyncPipelineManager` に投入される。

この経路は「現在描画中の pipeline effect を async 化する」ためのもので、次の前提を持つ。

- task key は effect name と param hash が中心。
- 同じ effect の新 task は古い task をキャンセルまたは worker restart しやすい。
- 結果 cache は現在の pipeline redraw と強く結びつく。
- AI-NR の raw result は主に現在の `primary_param` に戻る。

そのため、画像選択をまたいで「ファイルAのAI-NRを継続し、完了後にAへ安全に保存する」という用途には合わない。

## Proposed Structure

新規に `AIJobManager` を作り、既存 `AsyncWorker` とは分離する。

```text
MainWidget
  ├─ AsyncWorker
  │    └─ 現在表示中 pipeline 用 async effect
  └─ AIJobManager
       ├─ queued jobs
       ├─ active job
       ├─ result queue
       ├─ cancel flags
       └─ AIJobWorker process
```

`AIJobWorker` は1プロセスだけ起動し、queue された job を1件ずつ処理する。複数ファイルで AI-NR が要求されても並列実行はしない。

## Job Model

job は effect ではなく、画像ファイルと AI 処理種別に紐付ける。

```python
AIJob(
    job_id: int,
    kind: str,
    file_path: str,
    param_snapshot: dict,
    source_signature: str | int,
    content_key: str | int | None,
    status: str,  # queued, running, complete, cancelled, error, stale
    created_at: float,
    started_at: float | None,
)
```

`source_signature` は、job 開始時の入力画像と AI-NR 直前までの処理が同じかを判定するための値。最初の実装では `content_key` と同等でもよいが、将来的には loader revision / upstream hash / param subset hash を明示的に分ける。

## Job Kind Interface

AI-NR 以外も載せられるよう、job kind ごとの処理を小さい interface にする。

```python
class AIJobKind:
    name: str

    def should_enqueue(self, param: dict) -> bool:
        ...

    def build_job(self, file_path: str, param: dict) -> AIJob:
        ...

    def prepare_input(self, file_path: str, param_snapshot: dict):
        ...

    def run(self, prepared_input, cancel_token):
        ...

    def validate_result_target(self, job: AIJob, target_param: dict) -> bool:
        ...

    def merge_into_param(self, job: AIJob, target_param: dict, result) -> None:
        ...

    def merge_into_pmck(self, job: AIJob, result) -> bool:
        ...
```

AI-NR kind は SCUNet CoreML を使う。

## AI-NR Flow

### Enqueue

AI-NR の enqueue 条件は次の通り。

- `switch_ai_noise_reduction == True`
- `ai_noise_reduction == True`
- 入力画像が AI 処理可能
- 既存 `ai_noise_reduction_result` がない、または `ai_noise_reduction_content_key` が現在の content key と一致しない

同一 `file_path + kind` の job が既に queued なら置換する。running 中なら即停止はせず、古い running job を stale 候補にして新しい job を queue に積む。ユーザー明示キャンセルだけ running job を停止する。

### Run

worker 側で対象ファイルをロードし、AI-NR 直前までの入力を作る。その入力に対して `helpers.scunet_coreml_helper.predict_helper()` を実行し、raw result を返す。

AI-NR 強度は raw 生成には含めない。強度変更は現在どおり raw と base の blend だけで処理する。

### Completion

完了時は job 開始時の表示ファイルではなく、完了時点の current file を見て反映先を決める。

```text
AI-NR job完了
  ├─ job.file_path == current file_path
  │    ├─ current primary_param と job が一致するか検証
  │    ├─ 一致すれば primary_param へ raw result を直反映
  │    └─ redraw
  │
  └─ job.file_path != current file_path
       ├─ 対象 .pmck の現在設定と job が一致するか検証
       └─ 一致すれば .pmck へ raw result をマージ保存
```

現在表示中ファイルでは `.pmck` 経由で再読込しない。編集中の `primary_param` が `.pmck` より新しい可能性があるため、AI-NR result だけを param へ直接反映する。

```python
primary_param["ai_noise_reduction_result"] = raw_result
primary_param["ai_noise_reduction_content_key"] = job.content_key
```

その後 `start_draw_image()` する。`.pmck` 保存は即時保存してもよいが、他の編集状態との整合を優先するなら通常の `save_current_sidecar()` タイミングに任せる。

## Stale Result Guard

裏タスク実行中に編集中画像や設定が変わる可能性があるため、完了時に必ず検証する。

current param へ反映する条件。

- `self.imgset.file_path == job.file_path`
- `primary_param["switch_ai_noise_reduction"] == True`
- `primary_param["ai_noise_reduction"] == True`
- `job.content_key == current_content_key`
- `job.source_signature == current_source_signature`

非表示ファイルの `.pmck` へ保存する条件。

- `.pmck` を読み、現在の primary param を取得できる
- AI-NR が ON
- job の content key / source signature と `.pmck` 現在設定が一致

どちらにも一致しない場合、結果は stale として破棄する。

これにより次のケースを安全に扱える。

- A の AI-NR 実行中に B を選択した。
- A の AI-NR 実行中に B を選択し、完了前に A に戻った。
- A の AI-NR 実行中に A の設定を変更した。
- batch paste 後、完了前に対象ファイルの `.pmck` が別操作で更新された。

## User Cancellation

ユーザーキャンセルは job 単位または file 単位で行う。

- queued job: queue から削除
- running job: cancel flag を立て、必要なら worker process を terminate/restart
- same file の後続 queued job: ユーザー操作に応じて削除

画像選択変更では cancel しない。

## Batch Paste Integration

`paste_effect_settings()` の batch paste 完了後、更新対象のカードを走査して AI-NR job を enqueue する。

```text
batch paste
  ├─ .pmck に partial param を適用
  ├─ 対象 file_path ごとに AI-NR ON 判定
  ├─ 既存 raw result の content key を確認
  └─ 必要なら AIJobManager.enqueue("ai_noise_reduction", file_path)
```

現在表示中ファイルが対象に含まれる場合は、`primary_param` 更新後に current param から enqueue 判定する。

## Viewer State

viewer の thumbnail data に AI job 状態を追加する。

```python
ai_job_state = None | "queued" | "running" | "error"
```

`Viewer` に次の API を追加する。

```python
set_ai_job_state_for_path(file_path, state)
clear_ai_job_state_for_path(file_path)
```

`ThumbnailCard` は既存の `load_pending` 表示と同じ思想で、画像上に小さい indicator を出す。

- queued: 薄い spinner
- running: spinner 強表示
- error: 小さい error indicator
- complete / cancelled / stale: 非表示

## MainWidget Integration

`MainWidget` は起動時に `AIJobManager` を持つ。

```python
self.ai_job_manager = AIJobManager(...)
```

定期 poll で次を行う。

- job status を viewer に反映
- 完了 result を current param または `.pmck` に反映
- current param へ反映した場合は redraw
- error/cancel 状態を UI に反映

既存の `is_processing` は「ファイルロードや現在 pipeline の処理中」を示す意味が強いため、AI job queue を含めるかは UI 方針で決める。少なくとも viewer thumbnail には個別表示する。

## AINoiseReductonEffect Changes

`AINoiseReductonEffect` は既存 `AsyncWorker` へ AI-NR を submit しない方向へ寄せる。

新しい処理。

1. 有効な `ai_noise_reduction_result` が param にあれば blend して返す。
2. raw がなければ `AIJobManager` に current file の job 状態を問い合わせる。
3. job がなければ enqueue する。
4. raw 完了までは AI-NR なし、または既存 raw がある場合のみ preview blend を返す。

これにより AI-NR は「表示中 pipeline の async effect」ではなく「画像ファイルに紐付く background AI result」になる。

## Persistence

`.pmck` に保存する heavy keys は既存方針に合わせる。

- `ai_noise_reduction_result`
- `ai_noise_reduction_content_key`
- `heavy_saved_at_fidelity`

現在表示中ファイルは primary param へ直反映し、`.pmck` 保存は通常保存経路に任せる。非表示ファイルは `.pmck` に直接マージ保存する。

## Testing Plan

- 画像選択変更で AI-NR job が cancel されない。
- queued job は1件ずつ実行される。
- 同一ファイルの queued job は新しい job に置換される。
- ユーザーキャンセルで queued/running job が止まる。
- running 中に別画像へ移動しても、完了結果が誤って current param に入らない。
- running 中に元画像へ戻った場合、content key が一致する時だけ current param に反映される。
- batch paste で AI-NR ON の対象だけ enqueue される。
- 非表示ファイルの完了結果は `.pmck` にマージされる。
- stale result は `.pmck` にも current param にも入らない。
- viewer の queued/running/error indicator が更新される。

## Implementation Order

1. `cores/ai_job_manager/` を追加し、`AIJob`, `AIJobManager`, `AIJobWorker` の最小実装を作る。
2. AI-NR job kind を実装する。
3. `MainWidget` に manager と poll 処理を追加する。
4. viewer に `ai_job_state` indicator を追加する。
5. `AINoiseReductonEffect` を `AIJobManager` 経由に切り替える。
6. batch paste 後の enqueue を追加する。
7. stale guard と cancel API を固める。
8. テストを追加する。
