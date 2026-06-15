
import os
import numpy as np
from typing import Union

def read_lut(lut_path, clip=False):
    from cores.lut_functions import read_LUT_IridasCube, LUT3D, LUT3x1D
    #from colour.io.luts.iridas_cube import read_LUT_IridasCube, LUT3D, LUT3x1D
    
    """
    Reads a LUT from the specified path, returning instance of LUT3D or LUT3x1D

    <lut_path>: the path to the file from which to read the LUT (
    <clip>: flag indicating whether to apply clipping of LUT values, limiting all values to the domain's lower and
        upper bounds
    """
    lut: Union[LUT3x1D, LUT3D] = read_LUT_IridasCube(lut_path)
    lut.name = os.path.splitext(os.path.basename(lut_path))[0]  # use base filename instead of internal LUT name

    if clip:
        if lut.domain[0].max() == lut.domain[0].min() and lut.domain[1].max() == lut.domain[1].min():
            lut.table = np.clip(lut.table, lut.domain[0, 0], lut.domain[1, 0])
        else:
            if len(lut.table.shape) == 2:  # 3x1D
                for dim in range(3):
                    lut.table[:, dim] = np.clip(lut.table[:, dim], lut.domain[0, dim], lut.domain[1, dim])
            else:  # 3D
                for dim in range(3):
                    lut.table[:, :, :, dim] = np.clip(lut.table[:, :, :, dim], lut.domain[0, dim], lut.domain[1, dim])

    return lut


def apply_lut(image, lut, log=False, overrange="clip"):
    
    """Opens the image at <image_path>, transforms it using the passed
    <lut> with trilinear interpolation, and saves the image at
    <output_path>, or if it is None, then the same folder as <image_path>.
    If <thumb> is greater than zero, then the image will be resized to have
    a max height or width of <thumb> before being transformed. Iff <log> is
    True, the image will be changed to log colorspace before the LUT.

    <lut>: CubeLUT object containing LUT
    <log>: if True, transform to log colorspace
    """

    im_array = np.asarray(image, dtype=np.float32)
    if log:
        im_array = im_array ** (1/2.2)

    if overrange == "preserve":
        domain_min = lut.domain[0]
        domain_max = lut.domain[1]
        lut_input = np.clip(im_array, domain_min, domain_max)
        result = lut.apply(lut_input)
        result = result + (im_array - lut_input)
    else:
        result = lut.apply(im_array)

    if log:
        result = result ** (2.2)

    return result.astype(np.float32)
