
import torch

import numpy as np

def aces_tonemapping(image, exposure=0.6, device='cpu'):
    """
    ACESトーンマッピングを画像に適用する関数。
    Args:
        image (numpy.ndarray): 入力画像（float型、shape: [H, W, 3]）。
        exposure (float, optional): 露出係数。デフォルトは0.6。
        device (str, optional): 計算に使用するデバイス（'cpu'または'cuda'）。デフォルトは'cpu'。
    Returns:
        numpy.ndarray: トーンマッピング後の画像（float型、shape: [H, W, 3]、値域は[0, 1]）。
    Note:
        - 入力画像はfloat型（0以上）である必要があります。
        - ACES（Academy Color Encoding System）方式によるトーンマッピングを行います。
        - 行列変換により色空間変換を行い、非線形処理でダイナミックレンジを圧縮します。
    """

    # テンソル変換
    tensor = torch.from_numpy(image).to(device)
    
    # 行列定義
    in_mat = torch.tensor([
        [0.59719, 0.35458, 0.04823],
        [0.07600, 0.90834, 0.01566],
        [0.02840, 0.13383, 0.83777]
    ], device=device)
    
    out_mat = torch.tensor([
        [1.60475, -0.53108, -0.07367],
        [-0.10208, 1.10813, -0.00605],
        [-0.00327, -0.07276, 1.07602]
    ], device=device)
    
    # 演算チェーン
    processed = tensor * exposure
    processed = torch.einsum('...c,rc->...r', processed, in_mat)
    processed = (processed * (2.51 * processed + 0.03)) / \
                (processed * (2.43 * processed + 0.59) + 0.14)
    processed = torch.einsum('...c,rc->...r', processed, out_mat)
    
    return torch.clamp(processed, 0, 1).cpu().numpy()

# 使用例
if __name__ == "__main__":
    # テスト画像生成（HDRシミュレーション）
    hdr = np.random.rand(1024, 1024, 3).astype(np.float32) * 10
    hdr = np.clip(hdr, 0, 10) #/ 10  # 0-1に正規化
    
    # 処理
    result_cpu = aces_tonemapping(hdr, exposure=0.8)
    
    # 結果比較
    print("CPU/GPU差分:", np.max(np.abs(result_cpu - result_cpu)))