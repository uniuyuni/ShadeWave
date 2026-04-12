from enum import Enum

class ImageFidelity(str, Enum):
    """画像バッファの解像度段階。重いAI結果の保存・復元は FULL のみ。"""
    PREVIEW = "preview"  # RAW の埋め込みプレビュー等、フルデコード前
    FULL = "full"        # フルRAWデコード完了、またはフルRGB読込完了


class LoadStage(int, Enum):
    """
    ファイル読み込み完了コールバックの段階（旧 0 / -2 / -1 / 1 の魔術数字を置き換え）。
    process_pipeline の loading_flag（-1=下流許可）とは別物。
    """
    FIRST_PAINTABLE = 0  # RAW 埋め込みプレビュー初回、キャッシュヒット初回など
    RGB_DONE = 2         # 単発 RGB 読み込み完了（旧 -2）
    FULL_DECODE = 3      # RAW フルデコード完了・最終（旧 -1）


def coerce_load_stage(stage):
    """旧 int コールバックや生の値を LoadStage に正規化する。"""
    if isinstance(stage, LoadStage):
        return stage
    legacy = {
        0: LoadStage.FIRST_PAINTABLE,
        1: LoadStage.FIRST_PAINTABLE,  # 旧 RAW ハーフ段階（廃止）→ プレビュー相当として扱う
        -2: LoadStage.RGB_DONE,
        -1: LoadStage.FULL_DECODE,
    }
    if isinstance(stage, int) and stage in legacy:
        return legacy[stage]
    try:
        return LoadStage(int(stage))
    except (ValueError, TypeError):
        return LoadStage.FULL_DECODE

class EffectMode(int, Enum):
    PREVIEW = 0
    LOUPE = 1
    EXPORT = 2

class ExecutionMode(int, Enum):
    SYNC = 0
    ASYNC = 1
    BLOCKING = 2

# Status of upstream processing
class PipelineStatus(int, Enum):
    PREVIEW = 0   # Processing or Preview quality
    COMPLETE = 1  # High quality complete
