from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
import numpy as np
import torch
from typing import Optional

def setup_model(
    config_file: str, 
    weights_file: Optional[str] = None,
    device_name: str = 'cpu',
    confidence_threshold: float = 0.5
) -> DefaultPredictor:
    """
    Detectron2モデルを作成する関数
    
    Args:
        config_file (str): 設定ファイルのパス
        device_name (str): デバイス名 ('cuda' または 'cpu')
        weights_file (str, optional): 重みファイルのパス（Noneの場合は設定ファイルの値を使用）
        confidence_threshold (float): 信頼度閾値
    
    Returns:
        DefaultPredictor: 初期化された予測オブジェクト
    """
    # 設定の初期化
    cfg = get_cfg()
    
    # 設定ファイルの読み込み
    cfg.merge_from_file(config_file)
    
    # 重みファイルの設定（引数で指定された場合のみ上書き）
    if weights_file is not None:
        cfg.MODEL.WEIGHTS = weights_file
    
    # デバイス設定
    cfg.MODEL.DEVICE = device_name
    
    # 信頼度閾値の設定
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    
    # 設定を凍結
    cfg.freeze()
    
    # 予測オブジェクトの作成
    return DefaultPredictor(cfg)

def run_inference(
    predictor: DefaultPredictor, 
    image: np.ndarray
) -> dict:
    """
    画像推論を実行する関数
    
    Args:
        predictor (DefaultPredictor): 作成した予測オブジェクト
        image (np.ndarray): RGBフォーマットのfloat32画像 [0-1]範囲
    
    Returns:
        dict: 推論結果を含む辞書
    """
    # 入力画像の前処理
    image_uint8 = (image * 255).astype(np.uint8)
    
    # 推論実行
    outputs = predictor(image_uint8)
    
    # GPUテンソルをCPUに移動
    for key in outputs:
        if isinstance(outputs[key], torch.Tensor):
            outputs[key] = outputs[key].cpu().numpy()
    
    return outputs

def create_mask(output) -> np.ndarray:
    """
    panoptic_seg出力から0.0〜1.0範囲のモノクロマスクを生成
    
    Args:
        panoptic_seg_output (tuple): (panoptic_seg, segments_info) のタプル
        
    Returns:
        np.ndarray: float32形式のモノクロマスク [高さ, 幅] (0.0〜1.0範囲)
    """
    if "panoptic_seg" in output:
        panoptic_seg, segments_info = output["panoptic_seg"]
        
    else:
        raise ValueError("出力に有効なセグメンテーション結果が含まれていません")
     
    # テンソルをNumPy配列に変換
    if isinstance(panoptic_seg, torch.Tensor):
        panoptic_seg = panoptic_seg.cpu().numpy()
    
    # マスク用のゼロ配列を準備
    height, width = panoptic_seg.shape
    mask = np.zeros((height, width), dtype=np.float32)
    
    # 背景クラスと物体クラスを分けて処理
    background_segments = []
    object_segments = []
    
    for segment in segments_info:
        if segment.get('isthing', False):
            object_segments.append(segment)
        else:
            background_segments.append(segment)
    
    # 背景クラスの値割り当て (0.0〜0.499)
    for i, segment in enumerate(background_segments):
        segment_id = segment['id']
        segment_mask = (panoptic_seg == segment_id)
        
        # 背景クラスの値: 0.001単位で増加
        value = (i + 1) * 0.001
        mask[segment_mask] = min(value, 0.499)  # 上限を0.499に制限
    
    # 物体クラスの値割り当て (0.5〜1.0)
    for i, segment in enumerate(object_segments):
        segment_id = segment['id']
        category_id = segment['category_id']
        segment_mask = (panoptic_seg == segment_id)
        
        # 物体クラスの値: 0.5 + (カテゴリID * 0.01) + (インスタンスID * 0.001)
        value = 0.5 + (category_id * 0.01) + (i * 0.001)
        mask[segment_mask] = min(value, 1.0)  # 上限を1.0に制限
    
    return mask
