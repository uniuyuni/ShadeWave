"""pycubelut v0.2
=== Author ===
Yoonsik Park
park.yoonsik@icloud.com
=== Description ===
A library and standalone tool to apply Adobe Cube LUTs to common image
formats. This supports applying multiple LUTs and batch image processing.
"""

import logging
import numpy as np
from colour.io.luts.iridas_cube import read_LUT_IridasCube, LUT3D, LUT3x1D
import os
from multiprocessing import Pool
from typing import Union


def read_lut(lut_path, clip=False):
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


def process_image(image, lut, log=False):
    """Opens the image at <image_path>, transforms it using the passed
    <lut> with trilinear interpolation, and saves the image at
    <output_path>, or if it is None, then the same folder as <image_path>.
    If <thumb> is greater than zero, then the image will be resized to have
    a max height or width of <thumb> before being transformed. Iff <log> is
    True, the image will be changed to log colorspace before the LUT.

    <lut>: CubeLUT object containing LUT
    <log>: if True, transform to log colorspace
    """

    logging.debug("Applying LUT: " + lut.name)
    im_array = image
    is_non_default_domain = not np.array_equal(lut.domain, np.array([[0., 0., 0.], [1., 1., 1.]]))
    dom_scale = None
    if is_non_default_domain:
        dom_scale = lut.domain[1] - lut.domain[0]
        im_array = im_array * dom_scale + lut.domain[0]
    if log:
        im_array = im_array ** (1/2.2)
    im_array = lut.apply(im_array)
    if log:
        im_array = im_array ** (2.2)
    if is_non_default_domain:
        im_array = (im_array - lut.domain[0]) / dom_scale

    return im_array.astype(np.float32)
