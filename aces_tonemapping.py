import numpy as np
import cv2

import torch

def aces_tonemapping(image, exposure=0.6, device='cpu'):
    """ PyTorchによる高速実装 """
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
    result_cpu = torch_aces(hdr, exposure=0.8)
    
    # 結果比較
    print("CPU/GPU差分:", np.max(np.abs(result_cpu - result_cpu)))