"""image_fidelity / param 連携。重い処理結果の保存・適用は ImageFidelity.FULL のみ。"""

from enums import ImageFidelity


def heavy_ai_allowed(param):
    """
    SCUNet・インペイント等を実行してよいか。
    以前は FULL のみ True にしていたため、RAW の埋め込みプレビュー（PREVIEW）の間は
    重い処理が一切走らず、コードを直しても見た目が「変わらない」状態になっていた。

    プレビュー／フル両方で実行を許可する。pmck への重い結果の保存は params.serialize 側で
    image_fidelity==FULL のときだけ行う（変更なし）。
    """
    fid = param.get("image_fidelity")
    if fid is None:
        return True
    return fid in (ImageFidelity.PREVIEW.value, ImageFidelity.FULL.value)


def pipeline_loading_flag(imgset):
    """
    process_pipeline / LoadingWait 用の loading_flag。
    None=未ロード, -1=下流許可（effects._loading_flag_ready_for_heavy_effects）
    """
    if imgset is None:
        return None
    return -1
