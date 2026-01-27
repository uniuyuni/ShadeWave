
import numpy as np

def __naive_sigmoid(x, gain, mid):    
    return 1.0 / (1.0 + np.exp((mid - x) * gain))

def naive_sigmoid(x, gain, mid):
    return __naive_sigmoid(x, gain, mid)

def __scaled_sigmoid(x, gain, mid):
    min_val = __naive_sigmoid(0.0, gain, mid)
    max_val = __naive_sigmoid(1.0, gain, mid)
    s = __naive_sigmoid(x, gain, mid)
    return (s - min_val) / (max_val - min_val)

def scaled_sigmoid(x, gain, mid):
    return __scaled_sigmoid(x, gain, mid)

def __naive_inverse_sigmoid(x, gain, mid):
    min_val = __naive_sigmoid(0.0, gain, mid)
    max_val = __naive_sigmoid(1.0, gain, mid)
    #s = __naive_sigmoid(jnp.clip(x, 1e-7, 1 - 1e-7), gain, mid) # Old comment
    a = (max_val - min_val) * x + min_val
    return np.log(1.0 / a - 1.0)

def naive_inverse_sigmoid(x, gain, mid):
    return __naive_inverse_sigmoid(x, gain, mid)

def __scaled_inverse_sigmoid(x, gain, mid):
    min_val = __naive_inverse_sigmoid(0.0, gain, mid)
    max_val = __naive_inverse_sigmoid(1.0, gain, mid)
    s = __naive_inverse_sigmoid(x, gain, mid)
    return ((s - min_val) / (max_val - min_val))

def scaled_inverse_sigmoid(x, gain, mid):
    return __scaled_inverse_sigmoid(x, gain, mid)



def calculate_steepness(center):
    """中心位置に基づいて勾配係数を計算"""
    # 中心からの距離に基づいて勾配を調整
    distance_from_center = center - 0.5
    
    if distance_from_center >= 0:
        # 中心からの距離に応じて勾配を調整
        # 端に近づくほど大きな値を返す
        return 8 / (1 - distance_from_center)
    
    return 8 + distance_from_center * 8

def sigmoid(x, center):
    """シグモイド関数の計算"""
    x = np.clip(x, 0, 1)
    
    k = calculate_steepness(center)
    
    # 中心を基準とした相対位置の計算
    #relative_x = (x - center) / (1 - center)
    
    # 基本のシグモイド関数
    if center < 0.5:
        # 左寄りの場合
        #s = 1 / (1 + np.exp(-k * (x / center - 1)))
        s = 1 / (1 + np.exp(-k * (x - 0.5) * 2))
    elif center > 0.5:
        # 右寄りの場合
        s = 1 / (1 + np.exp(-k * ((x - center) / (1 - center))))
    else:
        # 中央の場合
        s = 1 / (1 + np.exp(-k * (x - 0.5) * 2))
    
    # 0と1の間に正規化
    result = (s - 1 / (1 + np.exp(k))) / (1 / (1 + np.exp(-k)) - 1 / (1 + np.exp(k)))
    
    return result
