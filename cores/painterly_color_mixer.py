import numpy as np

class PainterlyColorMixer:
    """Linear RGB専用・絵画的色調操作（バグ修正・ガマット制御付き）"""
    _M1 = np.array([[ 0.8189330101,  0.3618667424, -0.1288597137],
                    [ 0.0329845436,  0.9293118715,  0.0361456387],
                    [ 0.0482003018,  0.2643662691,  0.6338517070]], dtype=np.float32)
    _M2 = np.array([[ 0.2104542553,  0.7936177850, -0.0040720468],
                    [ 1.9779984951, -2.4285922050,  0.4505937099],
                    [ 0.0259040371,  0.7827717662, -0.8086757660]], dtype=np.float32)

    @classmethod
    def _to_oklab(cls, rgb):
        lms = np.cbrt(np.maximum(rgb, 0) @ cls._M1.T)
        return lms @ cls._M2.T

    @classmethod
    def _from_oklab(cls, lab):
        lms = lab @ cls._M2.T
        return np.power(lms, 3) @ cls._M1.T

    @classmethod
    def muddy(cls, rgb, amount=0.5, mud_tone=(0.5, 0.5, 0.5)):
        lab = cls._to_oklab(rgb)
        L, a, b = lab[...,0], lab[...,1], lab[...,2]
        mud = cls._to_oklab(np.asarray(mud_tone, dtype=np.float32))
        mL, ma, mb = mud[...,0], mud[...,1], mud[...,2]

        # L,a,b をそれぞれ泥色の方向へ直線補間
        out = cls._from_oklab(np.stack([
            L + (mL - L) * amount,
            a + (ma - a) * amount,
            b + (mb - b) * amount
        ], axis=-1))
        return np.clip(out, 0.0, 1.0)

    @classmethod
    def glaze(cls, rgb, glaze_color, alpha=0.3):
        gc = np.asarray(glaze_color, dtype=np.float32)
        return rgb * (1.0 - alpha) + rgb * gc * alpha

    @classmethod
    def pastel(cls, rgb, amount=0.5, white_point=(1.0, 1.0, 1.0)):
        lab = cls._to_oklab(rgb)
        L, a, b = lab[...,0], lab[...,1], lab[...,2]
        wp = cls._to_oklab(np.asarray(white_point, dtype=np.float32))
        wL, wa, wb = wp[...,0], wp[...,1], wp[...,2]

        out = cls._from_oklab(np.stack([
            L + (wL - L) * amount,
            a + (wa - a) * amount,
            b + (wb - b) * amount
        ], axis=-1))
        return np.clip(out, 0.0, 1.0)

    @classmethod
    def dark_grayish(cls, rgb, amount=0.5, gray_tone=(0.2, 0.2, 0.2)):
        lab = cls._to_oklab(rgb)
        L, a, b = lab[...,0], lab[...,1], lab[...,2]
        gt = cls._to_oklab(np.asarray(gray_tone, dtype=np.float32))
        gL, ga, gb = gt[...,0], gt[...,1], gt[...,2]

        out = cls._from_oklab(np.stack([
            L + (gL - L) * amount,
            a + (ga - a) * amount,
            b + (gb - b) * amount
        ], axis=-1))
        return np.clip(out, 0.0, 1.0)