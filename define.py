
APPNAME = "Shade Wave"
VERSION = "2.72.155"

SUPPORTED_FORMATS_RGB = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif', '.heic', '.jxl')
SUPPORTED_FORMATS_RAW = ('.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.raf', '.rw2', '.sr2', '.pef', '.raw', '3fr', 'fff')
# EXR(HDR リニア)は pyvips 非対応で OpenEXR 経由。RGB とは別経路で扱うので独立タプルにする。
SUPPORTED_FORMATS_EXR = ('.exr',)
