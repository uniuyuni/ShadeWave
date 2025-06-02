import numpy as np
import cv2

def apply_clarity(rgb_image, clarity_amount):
    """
    RGB float32画像に明瞭度（マイクロコントラスト）を適用する関数（OpenCV使用）
    
    Parameters:
    -----------
    rgb_image : numpy.ndarray
        RGB画像データ (H, W, 3) shape, float32, 値域 [0.0, 1.0]
    clarity_amount : int
        明瞭度の適用度 (-100 から 100)
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
    
    if not -100 <= clarity_amount <= 100:
        raise ValueError("clarity_amount must be between -100 and 100")
    
    # 効果なしの場合は元画像をそのまま返す
    if clarity_amount == 0:
        return rgb_image.copy()
    
    # パラメータ計算
    strength = clarity_amount / 100.0  # -1.0 to 1.0
    
    # カーネルサイズの設定（OpenCVのGaussianBlurはカーネルサイズを指定）
    if abs(strength) <= 0.3:
        kernel_size = 5  # sigma ≈ 1.0相当
    elif abs(strength) <= 0.7:
        kernel_size = 7  # sigma ≈ 1.5相当
    else:
        kernel_size = 9  # sigma ≈ 2.0相当
    
    # 強度の調整（非線形変換で自然な効果に）
    if strength > 0:
        amount = strength * 1.2  # 最大1.2倍
    else:
        amount = strength * 0.8  # 最大-0.8倍
    
    # ガウシアンぼかしを適用
    blurred = cv2.GaussianBlur(rgb_image, (kernel_size, kernel_size), 0)
    
    # 高周波成分を抽出
    high_freq = rgb_image - blurred
    
    # 明瞭度を適用
    result = rgb_image + high_freq * amount
    
    # 値域を [0.0, 1.0] にクランプ
    result = np.clip(result, 0.0, 1.0)
    
    return result


def apply_clarity_luminance(rgb_image, clarity_amount):
    """
    輝度チャンネルのみで明瞭度を適用する高品質版（OpenCV使用）
    
    Parameters:
    -----------
    rgb_image : numpy.ndarray
        RGB画像データ (H, W, 3) shape, float32, 値域 [0.0, 1.0]
    clarity_amount : int
        明瞭度の適用度 (-100 から 100)
    
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
    
    if not -100 <= clarity_amount <= 100:
        raise ValueError("clarity_amount must be between -100 and 100")
    
    if clarity_amount == 0:
        return rgb_image.copy()
    
    # RGB to Grayscale (輝度) 変換 - OpenCVを使用
    luminance = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    
    # パラメータ計算
    strength = clarity_amount / 100.0
    
    if abs(strength) <= 0.3:
        kernel_size = 5
    elif abs(strength) <= 0.7:
        kernel_size = 7
    else:
        kernel_size = 9
    
    if strength > 0:
        amount = strength * 1.0
    else:
        amount = strength * 0.6
    
    # 輝度チャンネルで明瞭度処理
    blurred_lum = cv2.GaussianBlur(luminance, (kernel_size, kernel_size), 0)
    high_freq_lum = luminance - blurred_lum
    enhanced_lum = luminance + high_freq_lum * amount
    enhanced_lum = np.clip(enhanced_lum, 0.0, 1.0)
    
    # 輝度の変化量を計算
    # ゼロ除算を避けるため、小さな値を加算
    epsilon = 1e-7
    lum_ratio = enhanced_lum / (luminance + epsilon)
    
    # RGB各チャンネルに変化量を適用
    result = rgb_image.copy()
    for channel in range(3):
        result[:, :, channel] *= lum_ratio
        result[:, :, channel] = np.clip(result[:, :, channel], 0.0, 1.0)
    
    return result


def apply_clarity_advanced(rgb_image, clarity_amount, preserve_mask=None):
    """
    高度な明瞭度処理（エッジ検出とマスクを併用）
    
    Parameters:
    -----------
    rgb_image : numpy.ndarray
        RGB画像データ (H, W, 3) shape, float32, 値域 [0.0, 1.0]
    clarity_amount : int
        明瞭度の適用度 (-100 から 100)
    preserve_mask : numpy.ndarray, optional
        保護マスク (H, W) shape, float32, 値域 [0.0, 1.0]
        1.0: 完全に処理適用, 0.0: 処理をスキップ
    
    Returns:
    --------
    numpy.ndarray
        処理後のRGB画像 (H, W, 3) shape, float32, 値域 [0.0, 1.0]
    """
    
    # 基本的な入力検証
    if not isinstance(rgb_image, np.ndarray):
        raise TypeError("rgb_image must be numpy.ndarray")
    
    if rgb_image.dtype != np.float32:
        raise TypeError("rgb_image must be float32")
    
    if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
        raise ValueError("rgb_image must have shape (H, W, 3)")
    
    if not isinstance(clarity_amount, (int, float)):
        raise TypeError("clarity_amount must be numeric")
    
    if not -100 <= clarity_amount <= 100:
        raise ValueError("clarity_amount must be between -100 and 100")
    
    if clarity_amount == 0:
        return rgb_image.copy()
    
    # 輝度変換
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    luminance = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    
    # エッジ検出によるマスク生成
    # Sobelフィルタでエッジを検出
    grad_x = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # エッジマスクを正規化
    edge_mask = edge_magnitude / (edge_magnitude.max() + 1e-7)
    edge_mask = np.clip(edge_mask, 0.0, 1.0)
    
    # ユーザー指定のマスクと組み合わせ
    if preserve_mask is not None:
        if preserve_mask.shape != luminance.shape:
            raise ValueError("preserve_mask must have same shape as image")
        final_mask = edge_mask * preserve_mask
    else:
        final_mask = edge_mask
    
    # パラメータ設定
    strength = clarity_amount / 100.0
    
    if abs(strength) <= 0.3:
        kernel_size = 5
    elif abs(strength) <= 0.7:
        kernel_size = 7
    else:
        kernel_size = 9
    
    if strength > 0:
        amount = strength * 1.0
    else:
        amount = strength * 0.6
    
    # 明瞭度処理
    blurred = cv2.GaussianBlur(rgb_image, (kernel_size, kernel_size), 0)
    high_freq = rgb_image - blurred
    
    # マスクを適用して処理
    result = rgb_image.copy()
    for channel in range(3):
        enhanced_channel = rgb_image[:, :, channel] + high_freq[:, :, channel] * amount
        # マスクを使って元画像と合成
        result[:, :, channel] = (rgb_image[:, :, channel] * (1.0 - final_mask) + 
                                enhanced_channel * final_mask)
    
    result = np.clip(result, 0.0, 1.0)
    return result

def apply_texture(rgb_image, texture_amount):
    """
    RGB float32画像にテクスチャ強調を適用する関数
    表面の質感や細かいパターンを強調
    
    Parameters:
    -----------
    rgb_image : numpy.ndarray
        RGB画像データ (H, W, 3) shape, float32, 値域 [0.0, 1.0]
    texture_amount : int
        テクスチャの適用度 (-100 から 100)
        負の値: スムース効果
        0: 変化なし
        正の値: テクスチャ強調
    
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
    
    if not isinstance(texture_amount, (int, float)):
        raise TypeError("texture_amount must be numeric")
    
    if not -100 <= texture_amount <= 100:
        raise ValueError("texture_amount must be between -100 and 100")
    
    if texture_amount == 0:
        return rgb_image.copy()
    
    # パラメータ計算
    strength = texture_amount / 100.0
    
    # テクスチャ検出用のマルチスケール処理
    # 2つの異なるスケールでぼかしを作成
    small_blur = cv2.GaussianBlur(rgb_image, (3, 3), 0)  # 細かいテクスチャ用
    medium_blur = cv2.GaussianBlur(rgb_image, (5, 5), 0)  # 中程度のテクスチャ用
    
    # 2段階の高周波成分を抽出
    fine_texture = rgb_image - small_blur      # 非常に細かいテクスチャ
    medium_texture = small_blur - medium_blur  # 中程度のテクスチャ
    
    # 輝度でテクスチャマスクを作成
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    luminance = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    
    # 局所標準偏差でテクスチャ領域を検出
    # 畳み込みで局所的な分散を計算
    kernel = np.ones((5, 5), np.float32) / 25
    local_mean = cv2.filter2D(luminance, -1, kernel)
    local_mean_sq = cv2.filter2D(luminance**2, -1, kernel)
    local_variance = local_mean_sq - local_mean**2
    texture_mask = np.sqrt(np.maximum(local_variance, 0))
    
    # テクスチャマスクを正規化
    texture_mask = texture_mask / (texture_mask.max() + 1e-7)
    texture_mask = np.clip(texture_mask, 0.0, 1.0)
    
    # 強度調整
    if strength > 0:
        fine_amount = strength * 0.8      # 細かいテクスチャ
        medium_amount = strength * 0.4    # 中程度のテクスチャ
    else:
        fine_amount = strength * 0.6
        medium_amount = strength * 0.3
    
    # テクスチャを適用
    result = rgb_image.copy()
    for channel in range(3):
        # 2段階のテクスチャを重ね合わせ
        enhanced = (rgb_image[:, :, channel] + 
                   fine_texture[:, :, channel] * fine_amount +
                   medium_texture[:, :, channel] * medium_amount)
        
        # テクスチャマスクで選択的に適用
        result[:, :, channel] = (rgb_image[:, :, channel] * (1.0 - texture_mask) + 
                                enhanced * texture_mask)
    
    result = np.clip(result, 0.0, 1.0)
    return result


def apply_texture_advanced(rgb_image, texture_amount):
    """
    高度なテクスチャ強調（周波数分離とウェーブレット風処理）
    
    Parameters:
    -----------
    rgb_image : numpy.ndarray
        RGB画像データ (H, W, 3) shape, float32, 値域 [0.0, 1.0]
    texture_amount : int
        テクスチャの適用度 (-100 から 100)
    
    Returns:
    --------
    numpy.ndarray
        処理後のRGB画像 (H, W, 3) shape, float32, 値域 [0.0, 1.0]
    """
    
    # 入力検証（同じ）
    if not isinstance(rgb_image, np.ndarray):
        raise TypeError("rgb_image must be numpy.ndarray")
    
    if rgb_image.dtype != np.float32:
        raise TypeError("rgb_image must be float32")
    
    if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
        raise ValueError("rgb_image must have shape (H, W, 3)")
    
    if not isinstance(texture_amount, (int, float)):
        raise TypeError("texture_amount must be numeric")
    
    if not -100 <= texture_amount <= 100:
        raise ValueError("texture_amount must be between -100 and 100")
    
    if texture_amount == 0:
        return rgb_image.copy()
    
    # パラメータ
    strength = texture_amount / 100.0 * 10
    
    # 周波数分離（擬似ウェーブレット）
    # 複数のスケールでの分解
    scales = [(3, 3), (5, 5), (7, 7), (9, 9)]
    frequency_bands = []
    
    current_image = rgb_image.copy()
    
    for i, (kx, ky) in enumerate(scales):
        blurred = cv2.GaussianBlur(current_image, (kx, ky), 0)
        high_freq = current_image - blurred
        frequency_bands.append(high_freq)
        current_image = blurred
    
    # 最低周波数成分
    frequency_bands.append(current_image)
    
    # 各周波数帯域に異なる重みを適用
    weights = [1.0, 0.7, 0.4, 0.2]  # 高周波ほど強く強調
    
    # 再構成
    result = frequency_bands[-1].copy()  # 最低周波数から開始
    
    for i, (band, weight) in enumerate(zip(frequency_bands[:-1], weights)):
        if strength > 0:
            enhanced_band = band * (1.0 + strength * weight * 0.5)
        else:
            enhanced_band = band * (1.0 + strength * weight * 0.3)
        
        result += enhanced_band
    
    result = np.clip(result, 0.0, 1.0)
    return result

def apply_microcontrast(image, strength):
    """
    DxO PhotoLab風のマイクロコントラスト処理
    
    Args:
        image: RGB画像 (float32, 0-1範囲)
        strength: 適用度 (-100 to 100)
    
    Returns:
        処理済み画像 (float32, 0-1範囲)
    """
    if strength == 0:
        return image.copy()
    
    # 強度を正規化 (-1.0 to 1.0)
    normalized_strength = strength / 100.0
    
    # RGB→LAB変換（明度のみ処理）
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0] / 100.0  # 0-1範囲に正規化
    
    # 多段階ガイドフィルタによる局所適応処理
    enhanced_L = _multi_scale_local_contrast(L, normalized_strength)
    
    # LAB→RGB変換
    lab[:, :, 0] = np.clip(enhanced_L * 100.0, 0, 100)
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return np.clip(result, 0, 1)

def _multi_scale_local_contrast(luminance, strength):
    """
    多段階局所コントラスト処理
    """
    if abs(strength) < 1e-6:
        return luminance
        
    result = luminance.copy()
    
    # 適度な効果のためのスケール設定
    scales = [
        {'radius': 8, 'eps': 0.01},
        {'radius': 20, 'eps': 0.02}
    ]
    
    total_detail = np.zeros_like(luminance)
    
    for scale in scales:
        # ガイドフィルタで局所平均を計算
        local_mean = _guided_filter(luminance, luminance, scale['radius'], scale['eps'])
        
        # 局所的な変動成分を抽出
        detail = luminance - local_mean
        total_detail += detail
    
    # 平均化
    total_detail /= len(scales)
    
    # 強度に応じた処理（線形スケーリング）
    strength_factor = strength * 1.4  # 適度な強度に調整
    
    # 正負で正しく処理
    if strength > 0:
        # 正：ディテールを強調（加算）
        result = luminance + total_detail * strength_factor
    else:
        # 負：ディテールを低下（減算、つまり平滑化方向）
        result = luminance + total_detail * strength_factor  # strength_factorが負なので減算になる
    
    return np.clip(result, 0, 1)

def _guided_filter(I, p, r, eps):
    """
    ガイドフィルタ実装
    """
    mean_I = cv2.boxFilter(I, cv2.CV_32F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (r, r))
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))
    
    return mean_a * I + mean_b
