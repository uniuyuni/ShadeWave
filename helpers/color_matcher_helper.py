
import numpy as np
from PIL import Image
from color_matcher import ColorMatcher
from color_matcher.normalizer import Normalizer

def predict(img_src, img_ref):
    img_src = Normalizer(img_src).type_norm()
    img_ref = Normalizer(img_ref).type_norm()

    cm = ColorMatcher()
    img_res = cm.transfer(src=img_src, ref=img_ref, method='mkl')
    img_res = Normalizer(img_res).norm_fun().astype(np.float32)

    return img_res

if __name__ == "__main__":
    img_src = np.array(Image.open("../test/content.jpg")).astype(np.float32) / 255.0
    img_ref = np.array(Image.open("../test/style.png")).astype(np.float32) / 255.0

    img_res = predict(img_src, img_ref)

    Image.fromarray((img_res * 255).astype(np.uint8)).save("../test/result.png")
