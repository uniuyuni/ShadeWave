import os
import torch
import numpy as np
import cv2
import time
import splitimage
import processing_dialog

from NAFNet.basicsr.models import create_model
from NAFNet.basicsr.utils import img2tensor as _img2tensor, tensor2img
from NAFNet.basicsr.utils.options import parse

def setup_nafnet(task="Image Debluring", device="cpu"):
    """
    Sets up the NAFNet model.
    
    Args:
        task (str): Task to run. Choices:
            "Image Denoising" (SIDD),
            "Image Debluring" (GoPro),
            "Stereo Image Super-Resolution" (NAFSSR)
        device (str): Device to run the model on ('cpu' or 'cuda').
        
    Returns:
        model: Loaded NAFNet model.
    """
    base_path = "NAFNet"
    
    if task == "Image Denoising":
        opt_path = os.path.join(base_path, "options/test/SIDD/NAFNet-width32.yml")
    elif task == "Image Debluring":
        opt_path = os.path.join(base_path, "options/test/GoPro/NAFNet-width64.yml")
    elif task == "Stereo Image Super-Resolution":
        opt_path = os.path.join(base_path, "options/test/NAFSSR/NAFSSR-L_4x.yml")
    else:
        raise ValueError(f"Unknown task: {task}")
        
    if not os.path.exists(opt_path):
        raise FileNotFoundError(f"Config file not found at {opt_path}")

    opt = parse(opt_path, is_train=False)
    opt["dist"] = False
    opt["num_gpu"] = 0 if device == 'cpu' else 1
    
    # Fix path to pretrained model in config if necessary
    # The config usually has relative path "experiments/pretrained_models/..."
    # We should ensure it's absolute or correct relative to execution.
    # basicSR parse likely handles it relative to root or keeps as is.
    # Let's adjust pretrain_network_g path to be absolute if it's relative
    if 'path' in opt and 'pretrain_network_g' in opt['path']:
        model_path = opt['path']['pretrain_network_g']
        if not os.path.isabs(model_path):
             opt['path']['pretrain_network_g'] = os.path.join(base_path, model_path)
             
    # Create model
    model = create_model(opt)
    
    # The create_model usually handles moving to device if opt['num_gpu'] > 0
    # But let's ensure evaluation mode
    model.net_g.eval()
    
    return model

def predict_nafnet(model, image):
    """
    Runs NAFNet prediction on an image.
    
    Args:
        model: Loaded NAFNet model.
        image (numpy.ndarray): Input image in RGB float32 format [0, 1]. Shape (H, W, 3).
        
    Returns:
        numpy.ndarray: Restored image in RGB float32 format [0, 1].
    """
    # Preprocessing
    # Expecting H, W, 3
    if image.ndim == 2:
        image = np.expand_dims(image, axis=2)
        image = np.repeat(image, 3, axis=2)
        
    # img2tensor expects float32 [0,1] input if float32=True
    # Basicsr img2tensor: (H, W, C) -> (C, H, W)
    
    # Ensure float32
    image = image.astype(np.float32)
    
    # Convert to tensor
    img_tensor = _img2tensor(image, bgr2rgb=False, float32=True)
    img_tensor = img_tensor.unsqueeze(0) # Add batch dimension (1, C, H, W)
    
    # Feed data
    # NAFNet model expects dict with key 'lq'
    model.feed_data(data={"lq": img_tensor})
    
    # Inference
    if hasattr(model, 'test'):
        model.test()
    else:
        # Fallback if model structure is different (e.g. direct network)
        with torch.no_grad():
             model.net_g(img_tensor)

    # Get visuals
    visuals = model.get_current_visuals()
    # tensor2img converts back to uint8 [0, 255] and likely BGR/RGB depending on args
    # default tensor2img: RGB(tensor) -> BGR(numpy uint8) if rgb2bgr=True (default true)
    # We want RGB float32.
    
    # Let's handle conversion manually to stay in float32 if possible, 
    # OR use tensor2img and convert back. tensor2img is robust.
    # The requested output is RGB float32 [0, 1].
    
    sr_img_uint8 = tensor2img([visuals["result"]], rgb2bgr=False) # Keep RGB
    
    sr_img_float32 = sr_img_uint8.astype(np.float32) / 255.0
    
    return sr_img_float32

def predict_nafnet_helper(model, np_image):
    
    start_time = time.time()
    split_images, split_info = splitimage.split_image_with_overlap(np_image, 512, 512, 16)

    denoised_images = []
    for i, image in enumerate(split_images):
        print(f"NAFNet Predict {i+1} / {len(split_images)}")
        if processing_dialog:
             processing_dialog.set_processing_text(f"Step {i+1} / {len(split_images)}")
        denoised_images.append(predict_nafnet(model, image))

    result = splitimage.combine_image_with_overlap(denoised_images, split_info)
    elapsed_time = time.time() - start_time
    print(f"Completed with NAFNet {elapsed_time:.2f} seconds")
    if processing_dialog:
        processing_dialog.set_processing_text("")

    return result
