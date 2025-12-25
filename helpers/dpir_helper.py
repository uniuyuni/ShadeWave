import os
import torch
import numpy as np
import time
import splitimage

from DPIR.models.network_unet import UNetRes as net
from DPIR.utils import utils_model

def setup_dpir(device="cpu", model_name="drunet_color"):
    """
    Sets up the DPIR model.
    
    Args:
        device (str): Device to run the model on ('cpu' or 'cuda').
        model_name (str): Name of the model to load ('drunet_color' or 'drunet_gray').
        
    Returns:
        model: Loaded PyTorch model.
    """
    if 'color' in model_name:
        n_channels = 3 
    else:
        n_channels = 1
        
    model_pool = 'DPIR/model_zoo'
    model_path = os.path.join(model_pool, model_name + '.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    
    return model

def predict_dpir(model, image, x8=False, noise_level_img=25, model_name="drunet_color"):
    """
    Runs DPIR prediction on an image.
    
    Args:
        model: Loaded PyTorch model.
        image (numpy.ndarray): Input image in RGB float32 format [0, 1]. Shape (H, W, 3) or (H, W).
        x8 (bool): Whether to use x8 augmentation (not fully implemented here, kept for API compatibility/future use).
                   If True, it uses mode=3 in test_mode.
        noise_level_img (float): Gaussian noise level (0-255).
        model_name (str): 'drunet_color' or 'drunet_gray'.
        
    Returns:
        numpy.ndarray: Denoised image in RGB float32 format [0, 1].
    """
    device = next(model.parameters()).device
    
    # Preprocessing
    if 'color' in model_name:
        # Expecting H, W, 3
        if image.ndim == 2:
             image = np.expand_dims(image, axis=2)
             image = np.repeat(image, 3, axis=2)
        n_channels = 3
    else:
        # Expecting H, W or H, W, 1
        if image.ndim == 3 and image.shape[2] == 3:
            # Simple conversion to gray if passed RGB to gray model (though user should handle this ideally)
            image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        if image.ndim == 2:
            image = np.expand_dims(image, axis=2)
        n_channels = 1
        
    # Convert to tensor (1, C, H, W)
    img_L = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().unsqueeze(0)
    
    # Add noise level map
    noise_level_model = noise_level_img / 255.
    noise_map = torch.FloatTensor([noise_level_model]).repeat(1, 1, img_L.shape[2], img_L.shape[3])
    img_L = torch.cat((img_L, noise_map), dim=1)
    img_L = img_L.to(device)
    
    # Inference
    with torch.no_grad():
        if not x8 and img_L.size(2) % 8 == 0 and img_L.size(3) % 8 == 0:
            img_E = model(img_L)
        elif not x8 and (img_L.size(2) % 8 != 0 or img_L.size(3) % 8 != 0):
            img_E = utils_model.test_mode(model, img_L, refield=64, mode=5)
        elif x8:
            img_E = utils_model.test_mode(model, img_L, mode=3)

    # Postprocessing
    img_E = img_E.squeeze().permute(1, 2, 0).cpu().numpy()
    
    # Clip to [0, 1]
    #img_E = np.clip(img_E, 0, 1)
    
    if 'color' not in model_name and img_E.ndim == 3:
         img_E = img_E.squeeze()
         
    return img_E

def predict_dpir_helper(model, np_image, x8=False, noise_level_img=25, model_name="drunet_color"):

    start_time = time.time()
    split_images, split_info = splitimage.split_image_with_overlap(np_image, 512, 512, 16)

    denoised_images = []
    for i, image in enumerate(split_images):
        print(f"DPIR Predict {i+1} / {len(split_images)}")
        denoised_images.append(predict_dpir(model, image, x8, noise_level_img, model_name))

    result = splitimage.combine_image_with_overlap(denoised_images, split_info)
    elapsed_time = time.time() - start_time
    print(f"Completed with DPIR {elapsed_time:.2f} seconds")

    return result
