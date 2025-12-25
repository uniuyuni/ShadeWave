
import cv2
import numpy as np
import timeit
import jax.numpy as jnp
from numba import jit, prange
import colour
from imagecodecs import CMS, cms_profile, cms_transform

import cores.core as core
import config

if __name__ == '__main__':
    """
    val = np.array(0.9000009)
    print(f"Normal div:{val/3}")

    uval = (val * ((2**16)-1)).astype(np.uint16)
    uval = uval // 3
    uval = uval.astype(np.float32) / ((2**16)-1)
    print(f"Uint16 div:{uval}")

    uval = (val * ((2**32)-1)).astype(np.uint32)
    upper = uval >> 16
    flower = (uval & 0xffff).astype(np.float32) / ((2**16)-1)

    upperv0 = upper // 3
    upperv1 = (upper + 1) // 3
    
    upperv0 = (upperv0).astype(np.float32) / ((2**16)-1)
    print(f"Uint16 div:{upperv0} s:{val/3-upperv0}")
    upperv1 = (upperv1).astype(np.float32) / ((2**16)-1)
    fval = upperv0 * (1.0 - flower) + upperv1 * flower
    print(f" Split div:{fval} s:{val/3-fval}")
    """

    img = cv2.imread("your_image.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255
    msk = np.zeros(img.shape[:2], dtype=np.float32)


    execution_time_colour = timeit.timeit(
        lambda: colour.RGB_to_RGB(img, 'ProPhoto RGB', config.get_config('display_color_gamut'), config.get_config('cat'),
                    apply_cctf_encoding=True, apply_gamut_mapping=True).astype(np.float32),
        number=10
    )

    with open('icc/ProPhoto RGB.icc', 'rb') as f:
        file_content_bytes = f.read()
    in_profile = cms_profile('sRGB')
    out_profile = cms_profile('sRGB')
    execution_time_lcms2 = timeit.timeit(
        lambda: cms_transform(img, file_content_bytes, out_profile, colorspace='RGB', outcolorspace='RGB', intent=CMS.INTENT.PERCEPTUAL),
        number=10
    )

    """
    cv2.ocl.setUseOpenCL(True)
    img2 = cv2.UMat(img)
    execution_time_cv = timeit.timeit(
        lambda: core.adjust_exposure(img2, 4),
        number=100
    )
    """

    print(f"colour, 実行時間: {execution_time_colour:.6f}秒 (100回の合計)")
    print(f"lcms2, 実行時間: {execution_time_lcms2:.6f}秒 (100回の合計)")

