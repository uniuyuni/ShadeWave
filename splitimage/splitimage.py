import numpy as np

def split_image_with_overlap(image: np.ndarray, block_height: int, block_width: int, overlap: int, crops_out: bool=False) -> tuple:
    """
    画像を指定したピクセル数のブロックに分割し、重なる部分を設ける。
    
    Parameters:
    - image: 分割対象のRGB画像 (ndarray)
    - block_height: 分割するブロックの高さ
    - block_width: 分割するブロックの幅
    - overlap: ブロック間の重なるピクセル数
    - crops_out: ブロックのクロップ情報を含めるかどうか
    
    Returns:
    - blocks: 分割されたブロックが格納されたリスト
    - split_info: ブロックの情報 (pad_height, pad_width, img_height, img_width, block_height, block_width, overlap)
    """
    img_height, img_width, channels = image.shape

    # 縦と横の端数を計算
    pad_height = (block_height - (img_height % (block_height - overlap))) % (block_height - overlap)
    pad_width = (block_width - (img_width % (block_width - overlap))) % (block_width - overlap)

    # パディングを追加して画像サイズをブロックサイズで割り切れるようにする
    padded_image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='reflect')

    # パディング後の新しいサイズ
    new_height, new_width, _ = padded_image.shape

    # ブロックごとに分割する
    blocks = []
    crops = []
    for i in range(0, new_height - overlap, block_height - overlap):
        for j in range(0, new_width - overlap, block_width - overlap):
            block = padded_image[i:i+block_height, j:j+block_width]
            blocks.append(block)
            crops.append((i, j, i+block_height, j+block_width))

    if crops_out:
        return blocks, crops, (pad_height, pad_width, img_height, img_width, block_height, block_width, overlap)
    
    return blocks, (pad_height, pad_width, img_height, img_width, block_height, block_width, overlap)

def blend_images(img1: np.ndarray, img2: np.ndarray, overlap: int, axis: int) -> np.ndarray:
    """
    2つの画像を指定された軸に沿ってブレンドする。
    
    Parameters:
    - img1: 1つ目の画像
    - img2: 2つ目の画像
    - overlap: 重なるピクセル数
    - axis: ブレンドする軸（0: 縦方向, 1: 横方向）
    
    Returns:
    - ブレンドされた画像
    """
    if axis == 0:
        alpha = np.linspace(1.0, 0.0, overlap).reshape(-1, 1, 1)
        blended = img1 * alpha + img2 * (1 - alpha)
        return blended
    elif axis == 1:
        alpha = np.linspace(1.0, 0.0, overlap).reshape(1, -1, 1)
        blended = img1 * alpha + img2 * (1 - alpha)
        return blended
    else:
        raise ValueError("Axis must be 0 (vertical) or 1 (horizontal)")

def combine_image_with_overlap(blocks: list, split_info: tuple) -> np.ndarray:
    """
    分割された画像ブロックを結合し、元の画像サイズに戻す。重なり部分をブレンドする。
    
    Parameters:
    - blocks: 分割されたブロックが格納されたリスト
    - split_info: ブロックの情報 (pad_height, pad_width, img_height, img_width, block_height, block_width, overlap)
    
    Returns:
    - 結合された画像 (ndarray)
    """
    pad_height, pad_width, img_height, img_width, block_height, block_width, overlap = split_info

    # パディング後の画像サイズを計算
    padded_height = img_height + pad_height
    padded_width = img_width + pad_width

    # 結合した画像を格納する配列
    combined_image = np.zeros((padded_height, padded_width, 3), dtype=blocks[0].dtype)

    # ブロックを結合
    block_index = 0
    for i in range(0, padded_height - overlap, block_height - overlap):
        y_start = i
        y_end = y_start + block_height

        combined_image_w = np.zeros((block_height, padded_width, 3), dtype=blocks[0].dtype)
        for j in range(0, padded_width - overlap, block_width - overlap):
            block = blocks[block_index]
            x_start = j
            x_end = x_start + block_width

            if x_start > 0:
                combined_image_w[:, x_start:x_start+overlap] = blend_images(
                    combined_image_w[:, x_start:x_start+overlap],
                    block[:, :overlap],
                    overlap,
                    axis=1
                )
                combined_image_w[:, x_start+overlap:x_end] = block[:, overlap:]
            else:
                combined_image_w[:, x_start:x_end] = block
            block_index += 1

        if y_start > 0:
            combined_image[y_start:y_start+overlap] = blend_images(
                combined_image[y_start:y_start+overlap],
                combined_image_w[:overlap],
                overlap,
                axis=0
            )
            combined_image[y_start+overlap:y_end] = combined_image_w[overlap:]
        else:
            combined_image[y_start:y_end] = combined_image_w

    return combined_image[:img_height, :img_width]
