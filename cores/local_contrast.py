
import numpy as np
import cv2

import cores.core as core
import cores.hlsrgb as hlsrgb

def apply_clarity(rgb_image, clarity_amount):
    """
    RGB float32画像に明瞭度（マイクロコントラスト）を適用する関数
    Guided Filterを使用し、ハローを抑制しつつ中間調のローカルコントラストを強調する
    (Lightroomの挙動に近づけた実装)
    
    Parameters:
    -----------
    rgb_image : numpy.ndarray
        RGB画像データ (H, W, 3) shape, float32, 値域 [0.0, 1.0]
    clarity_amount : int
        明瞭度の適用度 (-1 から 1)
        負の値: ソフト効果（ぼかし寄り）
        0: 変化なし
        正の値: シャープ効果（明瞭度向上）
    
    Returns:
    --------
    numpy.ndarray
        処理後のRGB画像 (H, W, 3) shape, float32, 値域 [0.0, 1.0]
    """
    
    # 入力検証
    if not isinstance(rgb_image, np.ndarray):
        raise TypeError("rgb_image must be numpy.ndarray")
    
    if rgb_image.dtype != np.float32:
        raise TypeError("rgb_image must be float32")
    
    if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
        raise ValueError("rgb_image must have shape (H, W, 3)")
    
    if not isinstance(clarity_amount, (int, float)):
        raise TypeError("clarity_amount must be numeric")
    
    if clarity_amount == 0:
        return rgb_image.copy()
    
    # 輝度画像の生成（ガイド画像として使用）
    luminance = core.cvtColorRGB2Gray(rgb_image)
    
    # パラメータ設定
    # Clarityは比較的広い範囲のローカルコントラストを扱う
    h, w = luminance.shape[:2]
    long_side = max(h, w)
    
    # 画像サイズに適応した半径設定 (例: 長辺の1-2%程度)
    radius = max(8, int(long_side * 0.02))
    eps = 0.005 # エッジ保存の強さ
    
    # Guided Filterによるベースレイヤー（平滑化画像）の作成
    # OpenCVのximgprocを使用
    if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'guidedFilter'):
        base_layer = cv2.ximgproc.guidedFilter(guide=luminance, src=luminance, radius=radius, eps=eps, dDepth=-1)
    else:
        # Fallback if ximgproc is not available (though checked)
        base_layer = _guided_filter_optimized(luminance, luminance, radius, eps)
    
    # 詳細レイヤー（Structure）の抽出
    # 元画像 - ベースレイヤー
    detail_layer = luminance - base_layer
    
    # 中間調マスクの作成 (計算)
    # 念のためfloat32を維持
    # mid_tone_mask = 1.0 - (2.0 * np.minimum(luminance, 1.0 - luminance)) ** 2
    
    strength = np.float32(clarity_amount)
    
    # 明瞭度適用
    # 詳細成分を強調するが、輝度に応じて強度を変える
    enhanced_luminance = luminance + detail_layer * strength 
    
    # 輝度の変化率を計算
    delta = enhanced_luminance - luminance
    
    # 彩度補正（L率に応じてRGBをスケーリング）
    result = rgb_image + delta[..., np.newaxis]
    
    return result


def apply_texture(rgb_image, texture_amount):
    """
    RGB float32画像にテクスチャ強調を適用する関数
    ノイズ（超高周波）を避け、中高周波成分のみを強調する
    (Lightroomの挙動に近づけた実装)
    
    Parameters:
    -----------
    rgb_image : numpy.ndarray
        RGB画像データ (H, W, 3) shape, float32, 値域 [0.0, 1.0]
    texture_amount : int
        テクスチャの適用度 (-1 から 1)
        
    Returns:
    --------
    numpy.ndarray
        処理後のRGB画像
    """
    
    if texture_amount == 0:
        return rgb_image.copy()
    
    # パラメータ計算
    strength = np.float32(texture_amount)
    
    # 輝度変換
    luminance = core.cvtColorRGB2Gray(rgb_image)
    
    # 周波数分離（バンドパスフィルタ）
    
    # 1. ノイズ成分（超高周波）を除去するためのわずかなブラー
    # sigma_noise = 0.5
    # noise_layer = cv2.GaussianBlur(luminance, (0, 0), sigma_noise)
    
    # 2. テクスチャ成分と構造成分を分けるためのブラー
    # blur_small = cv2.GaussianBlur(luminance, (0, 0), 1.0) # ノイズ除去用
    # blur_large = cv2.GaussianBlur(luminance, (0, 0), 4.0) # 構造抽出用
    # 上記パラメータは解像度依存の可能性あるが、一旦固定で
    
    blur_small = cv2.GaussianBlur(luminance, (0, 0), 1.0) 
    blur_large = cv2.GaussianBlur(luminance, (0, 0), 4.0) 
    
    # テクスチャ成分 = (小ブラー) - (大ブラー)
    extracted_texture = blur_small - blur_large
    
    # 強調適用
    # RGB画像に対して、抽出したテクスチャ成分を加算
    # strengthが正なら強調、負ならスムージング
    
    # 単純加算だと彩度が変わらないため、RGBそれぞれに加算で輝度コントラストをつける
    factor = np.float32(1.5)
    result = rgb_image + extracted_texture[..., np.newaxis] * strength * factor
    
    return result


def _guided_filter_optimized(I, p, r, eps):
    """
    Guided Filterの最適化版（OpenCVのboxFilterを利用）
    ximgprocがない場合のフォールバック
    """
    ksize = 2 * r + 1
    
    mean_I = cv2.boxFilter(I, cv2.CV_32F, (ksize, ksize))
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (ksize, ksize))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (ksize, ksize))
    
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (ksize, ksize))
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (ksize, ksize))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (ksize, ksize))
    
    q = mean_a * I + mean_b
    return q


def apply_microcontrast(image, strength):
    """
    DxO PhotoLab風のマイクロコントラスト処理
    
    Args:
        image: RGB画像 (float32, 0-1範囲、HDR領域 > 1.0 にも対応)
        strength: 適用度 (-1 to 1)
    
    Returns:
        処理済み画像 (float32, HDR保持)
    """
    if strength == 0:
        return image.copy()
    
    # 強度を正規化 (-1.0 to 1.0)
    normalized_strength = strength
    
    # HDR対応の線形YCbCr空間で輝度(Y)のみを処理する。
    # グローバル正規化を行わないため、極端なハイライトに引っ張られにくい。
    image_f32 = image.astype(np.float32, copy=False)
    ycbcr = hlsrgb.linear_rgb_to_ycbcr(image_f32)
    y_channel = ycbcr[..., 0]

    # 多段階ガイドフィルタによる局所適応処理
    enhanced_y = _multi_scale_local_contrast(y_channel, normalized_strength)

    # 色差は維持しつつ輝度のみ更新してRGBへ戻す
    result_ycbcr = ycbcr.copy()
    result_ycbcr[..., 0] = enhanced_y
    result = hlsrgb.linear_ycbcr_to_rgb(result_ycbcr)

    return result.astype(np.float32, copy=False)

def _multi_scale_local_contrast(luminance, strength):
    """
    多段階局所コントラスト処理
    """
    if abs(strength) < 1e-6:
        return luminance
    
    if isinstance(luminance, np.ndarray):
        luminance_umat = cv2.UMat(luminance)
        h, w = luminance.shape[:2]
        input_umat = False
    else:
        luminance_umat = luminance
        h, w = luminance_umat.get().shape[:2]
        input_umat = True
    
    # 適度な効果のためのスケール設定
    scales = [
        {'radius': 8, 'eps': 0.01},
        {'radius': 20, 'eps': 0.02}
    ]
    
    total_detail = cv2.UMat(h, w, cv2.CV_32F)
    
    for scale in scales:
        # ガイドフィルタで局所平均を計算
        local_mean = _guided_filter(luminance_umat, luminance_umat, scale['radius'], scale['eps'])
        
        # 局所的な変動成分を抽出
        detail = cv2.subtract(luminance_umat, local_mean)
        total_detail = cv2.add(total_detail, detail)
    
    # 平均化
    total_detail = cv2.divide(total_detail, len(scales))
    
    # 強度に応じた処理（線形スケーリング）
    strength_factor = strength * 1.4  # 適度な強度に調整
    
    # 正負で正しく処理
    result = cv2.add(luminance_umat, cv2.multiply(total_detail, strength_factor))
    
    return result if input_umat else result.get()

def _guided_filter(I, p, r, eps):
    """
    ガイドフィルタ実装
    """
    mean_I = cv2.boxFilter(I, cv2.CV_32F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (r, r))
    mean_Ip = cv2.boxFilter(cv2.multiply(I, p), cv2.CV_32F, (r, r))
    cov_Ip = cv2.subtract(mean_Ip, cv2.multiply(mean_I, mean_p))
    
    mean_II = cv2.boxFilter(cv2.multiply(I, I), cv2.CV_32F, (r, r))
    var_I = cv2.subtract(mean_II, cv2.multiply(mean_I, mean_I))
    
    a = cv2.divide(cov_Ip, cv2.add(var_I, eps))
    b = cv2.subtract(mean_p, cv2.multiply(a, mean_I))
    
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))
    
    return cv2.add(cv2.multiply(mean_a, I), mean_b)
