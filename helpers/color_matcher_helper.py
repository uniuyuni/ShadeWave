
import numpy as np
from PIL import Image
from color_matcher import ColorMatcher

def predict(img_src, img_ref):
    # float32 前提。color_matcher.Normalizer は min-max ストレッチで絶対輝度を破壊する
    # ため、ColorMatcher.transfer を直接呼ぶ。
    cm = ColorMatcher()
    img_res = cm.transfer(src=img_src, ref=img_ref, method='mvgd')
    return img_res.astype(np.float32)

if __name__ == "__main__":
    img_src = np.array(Image.open("../test/content.jpg")).astype(np.float32) / 255.0
    img_ref = np.array(Image.open("../test/style.png")).astype(np.float32) / 255.0

    img_res = predict(img_src, img_ref)
    img_res = np.clip(img_res, 0.0, 1.0)

    Image.fromarray((img_res * 255).astype(np.uint8)).save("../test/result.png")
