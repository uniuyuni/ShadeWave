import torch
import torch.nn.functional as F
import os
import numpy as np
import time
from runpy import run_path

import splitimage
import processing_dialog

def setup_restormer(task='Real_Denoising', device='cpu'):
    """
    Sets up the Restormer model.

    Args:
        task (str): Task to run. Choices:
            'Motion_Deblurring',
            'Single_Image_Defocus_Deblurring',
            'Deraining',
            'Real_Denoising',
            'Gaussian_Gray_Denoising',
            'Gaussian_Color_Denoising'
        device (str): Device to run the model on ('cpu' or 'cuda').

    Returns:
        model: Loaded PyTorch model.
    """
    # Define parameters based on task (from demo.py)
    parameters = {
        'inp_channels': 3,
        'out_channels': 3,
        'dim': 48,
        'num_blocks': [4, 6, 6, 8],
        'num_refinement_blocks': 4,
        'heads': [1, 2, 4, 8],
        'ffn_expansion_factor': 2.66,
        'bias': False,
        'LayerNorm_type': 'WithBias',
        'dual_pixel_task': False
    }

    base_path = "checkpoints/Restormer"
    weights = None

    if task == 'Motion_Deblurring':
        weights = os.path.join(base_path, 'motion_deblurring.pth')
    elif task == 'Single_Image_Defocus_Deblurring':
        weights = os.path.join(base_path, 'single_image_defocus_deblurring.pth')
    elif task == 'Deraining':
        weights = os.path.join(base_path, 'deraining.pth')
    elif task == 'Real_Denoising':
        weights = os.path.join(base_path, 'real_denoising.pth')
        parameters['LayerNorm_type'] = 'BiasFree'
    elif task == 'Gaussian_Color_Denoising':
        weights = os.path.join(base_path, 'gaussian_color_denoising_blind.pth')
        parameters['LayerNorm_type'] = 'BiasFree'
    elif task == 'Gaussian_Gray_Denoising':
        weights = os.path.join(base_path, 'gaussian_gray_denoising_blind.pth')
        parameters['inp_channels'] = 1
        parameters['out_channels'] = 1
        parameters['LayerNorm_type'] = 'BiasFree'
    else:
        raise ValueError(f"Unknown task: {task}")

    # Check if weights exist
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Model weights not found at {weights}")

    # Load architecture
    arch_path = os.path.join('Restormer', 'basicsr', 'models', 'archs', 'restormer_arch.py')
    if not os.path.exists(arch_path):
         raise FileNotFoundError(f"Model architecture not found at {arch_path}")
         
    load_arch = run_path(arch_path)
    model = load_arch['Restormer'](**parameters)

    model.to(device)

    checkpoint = torch.load(weights, map_location=device)
    model.load_state_dict(checkpoint['params'])
    model.eval()

    return model

def predict_restormer(model, image):
    """
    Runs Restormer prediction on an image.

    Args:
        model: Loaded PyTorch model.
        image (numpy.ndarray): Input image in RGB float32 format [0, 1]. Shape (H, W, 3).

    Returns:
        numpy.ndarray: Restored image in RGB float32 format [0, 1].
    """
    device = next(model.parameters()).device
    img_multiple_of = 8

    # Preprocessing
    # Expecting H, W, 3 (RGB) or H, W (Gray)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=2) # H, W, 1
    
    # Check dimensions
    # If the model expects 1 channel (Gray), but input is 3 (RGB), convert or warn.
    # We will assume user passes correct input for now, or match channels.
    # Note: validation of input channels vs model channels is complex without storing model config.
    # We'll proceed assuming correct input shape (H, W, C).

    # Convert to tensor (1, C, H, W)
    input_ = torch.from_numpy(np.ascontiguousarray(image)).float().permute(2, 0, 1).unsqueeze(0).to(device)

    # Pad if needed
    height, width = input_.shape[2], input_.shape[3]
    H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, ((width + img_multiple_of) // img_multiple_of) * img_multiple_of
    padh = H - height if height % img_multiple_of != 0 else 0
    padw = W - width if width % img_multiple_of != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

    with torch.no_grad():
        restored = model(input_)

    #restored = torch.clamp(restored, 0, 1)

    # Unpad
    restored = restored[:, :, :height, :width]

    # Convert back to numpy (H, W, C)
    restored = restored.squeeze().permute(1, 2, 0).cpu().numpy()

    return restored

def predict_restormer_helper(model, np_image):

    start_time = time.time()
    split_images, split_info = splitimage.split_image_with_overlap(np_image, 512, 512, 16)

    denoised_images = []
    for i, image in enumerate(split_images):
        print(f"Restormer Predict {i+1} / {len(split_images)}")
        processing_dialog.set_processing_text(f"Step {i+1} / {len(split_images)}")
        denoised_images.append(predict_restormer(model, image))

    result = splitimage.combine_image_with_overlap(denoised_images, split_info)
    elapsed_time = time.time() - start_time
    print(f"Completed with Restormer {elapsed_time:.2f} seconds")
    processing_dialog.set_processing_text("")

    return result
