import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl import cltypes
import imageio
import ctypes
import time

def adjust_to_multiple(image, size=8, mode='constant'):
    # 画像の高さと幅を取得
    h, w = image.shape[:2]
    
    # 8の倍数に切り上げた新しいサイズを計算
    new_h = (h + size-1) // size * size
    new_w = (w + size-1) // size * size
    
    # パディング量を計算
    pad_h = new_h - h
    pad_w = new_w - w
    
    # パディング幅を設定（次元ごとに指定）
    pad_width = [(0, pad_h), (0, pad_w)] + [(0, 0)] * (image.ndim - 2)
    
    # 画像の下側と右側をエッジ値でパディング
    padded_image = np.pad(image, pad_width=pad_width, mode=mode)
    
    return padded_image, (h, w)

def restore_original_size(padded_image, original_size):
    # 元のサイズを取得
    h_orig, w_orig = original_size
    
    # パディングされた部分を切り取って元のサイズに復元
    return padded_image[:h_orig, :w_orig, ...]

def bm3d_denoise(image, sigma=1, platform_preference='auto', device_preference=0):
    """
    BM3Dアルゴリズムを使用してNumPy画像データのノイズを除去します (高度な実装)
    
    注意: この関数を使用するには、完全なBM3Dカーネル実装が必要です。
    
    Parameters:
    -----------
    image : numpy.ndarray
        RGB形式またはグレースケール形式のノイズ画像データ (float32, 値の範囲は0.0-1.0)
    sigma : int, optional
        ノイズの標準偏差 (デフォルト: 25)
    platform_preference : str, optional
        使用するOpenCLプラットフォーム ('auto', 'nvidia', 'amd', 'intel'のいずれか)
    device_preference : int, optional
        使用するデバイスのインデックス (デフォルト: 0 [通常はGPU])
    
    Returns:
    --------
    numpy.ndarray
        ノイズ除去された画像データ (入力と同じ形状と型)
    """
    # Constants
    BLOCK_SIZE = 8
    STEP_SIZE = 3
    SPLIT_SIZE_X = (3*STEP_SIZE) #32
    SPLIT_SIZE_Y = (3*STEP_SIZE) #16
    BLOCK_SIZE_SQ = BLOCK_SIZE * BLOCK_SIZE 
    MAX_BLOCK_COUNT_1 = 16
    MAX_BLOCK_COUNT_2 = 32
    WINDOW_STEP_SIZE_1 = 3
    WINDOW_STEP_SIZE_2 = 3
    D_THRESHOLD_1 = (3 * 2500) # Hard threshold
    D_THRESHOLD_2 = (3 * 400)  # Wiener threshold
    PLATFORM_NVIDIA = 0
    PLATFORM_ATI = 1
    PLATFORM_INTEL = 2
    PLATFORM_APPLE = 3        

    # 各チャンネルを処理する関数
    def initialize(image):
        height, width = image.shape[:2]
        
        # OpenCLのセットアップ
        platforms = cl.get_platforms()
        if len(platforms) == 0:
            raise RuntimeError("No OpenCL platforms found")
        
        # プラットフォーム選択
        platform_id = 0
        USE_PLATFORM = PLATFORM_NVIDIA  # デフォルト
        
        if platform_preference != 'auto':
            for i, platform in enumerate(platforms):
                
                if platform_preference.lower() == 'nvidia' and 'nvidia' in platform.name.lower():
                    platform_id = i
                    USE_PLATFORM = PLATFORM_NVIDIA
                elif platform_preference.lower() == 'amd' and ('amd' in platform.name.lower() or 'advanced micro' in platform.name.lower()):
                    platform_id = i
                    USE_PLATFORM = PLATFORM_ATI
                elif platform_preference.lower() == 'intel' and 'intel' in platform.name.lower():
                    platform_id = i
                    USE_PLATFORM = PLATFORM_INTEL
                elif platform_preference.lower() == 'apple' and 'apple' in platform.name.lower():
                    platform_id = i
                    USE_PLATFORM = PLATFORM_APPLE
        else:
            # Auto-detect
            for i, platform in enumerate(platforms):
                if 'nvidia' in platform.name.lower():
                    platform_id = i
                    USE_PLATFORM = PLATFORM_NVIDIA
                    break
                elif 'amd' in platform.name.lower() or 'advanced micro' in platform.name.lower():
                    platform_id = i
                    USE_PLATFORM = PLATFORM_ATI
                elif 'intel' in platform.name.lower():
                    platform_id = i
                    USE_PLATFORM = PLATFORM_INTEL
                elif 'apple' in platform.name.lower():
                    platform_id = i
                    USE_PLATFORM = PLATFORM_APPLE
                    break
                
        # デバイスの選択
        devices = platforms[platform_id].get_devices(cl.device_type.GPU | cl.device_type.CPU)
        if len(devices) == 0:
            raise RuntimeError("No OpenCL devices found")
        
        device_id = min(device_preference, len(devices) - 1)
        device = devices[device_id]
                
        # コンテキストの作成
        def pfn_notify(errinfo, private_info, cb, user_data):
            print(f"OpenCL Error: {errinfo}")
        
        context = cl.Context(devices=[device], properties=None, dev_type=None)
        
        # BM3Dカーネルのロード
        with open("bm3dcl/bm3d.cl", "r") as f:
            kernel_source = f.read()
        
        # キューの作成
        queue = cl.CommandQueue(context, device)
        
        # ビルドオプション
        build_options = f"-cl-std=CL1.1 -DWIDTH={width} -DHEIGHT={height} -DUSE_PLATFORM={USE_PLATFORM} -DSIGMA={sigma} -DBLOCK_SIZE={BLOCK_SIZE}"
        
        try:
            program = cl.Program(context, kernel_source).build(options=build_options)

        except cl.RuntimeError as e:
            print(f"Kernel build error: {e}")
            print(f"Build log: {program.get_build_info(device, cl.program_build_info.LOG)}")
            return None
        
        # カーネルの作成        
        dist_kernel = program.calc_distances
        basic_kernel = program.bm3d_basic_filter
        wiener_kernel = program.bm3d_wiener_filter
        
        # ワークグループ情報の取得
        multiple = dist_kernel.get_work_group_info(cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, device)
        max_wg = dist_kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, device)
        
        # イメージバッファの作成
        image_format = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)
        #noisy_image_buffer = cl.create_image(context, cl.mem_flags.READ_ONLY, image_format, shape=(width, height))
        basic_image_buffer = cl.create_image(context, cl.mem_flags.READ_WRITE, image_format, shape=(width, height))
        wiener_image_buffer = cl.create_image(context, cl.mem_flags.WRITE_ONLY, image_format, shape=(width, height))
        
        # ワークサイズの定義
        ls = (16, 8)  # ローカルサイズ
        gx_d = ((width + STEP_SIZE - 1) // STEP_SIZE + ls[0] - 1) // ls[0] * ls[0]
        gy_d = ((height + STEP_SIZE - 1) // STEP_SIZE + ls[1] - 1) // ls[1] * ls[1]
        tot_items_d = gx_d * gy_d
        
        gx = ((width + SPLIT_SIZE_X - 1) // SPLIT_SIZE_X + ls[0] - 1) // ls[0] * ls[0]
        gy = ((height + SPLIT_SIZE_Y - 1) // SPLIT_SIZE_Y + ls[1] - 1) // ls[1] * ls[1]
        tot_items = gx * gy
                
        # バッファの作成
        similar_coords_size = MAX_BLOCK_COUNT_2 * tot_items_d * np.dtype(np.int16).itemsize * 2
        similar_coords_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=similar_coords_size)
        
        block_counts_size = tot_items_d * np.dtype(np.uint8).itemsize
        block_counts_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=block_counts_size)
        
        param = {}
        param['context'] = context
        param['queue'] = queue
        param['program'] = program
        param['dist_kernel'] = dist_kernel
        param['basic_kernel'] = basic_kernel
        param['wiener_kernel'] = wiener_kernel
        #param['noisy_image_buffer'] = noisy_image_buffer
        param['basic_image_buffer'] = basic_image_buffer
        param['wiener_image_buffer'] = wiener_image_buffer
        param['similar_coords_buffer'] = similar_coords_buffer
        param['block_counts_buffer'] = block_counts_buffer
        param['tot_items'] = tot_items
        param['tot_items_d'] = tot_items_d
        param['gx'] = gx
        param['gy'] = gy
        param['gx_d'] = gx_d
        param['gy_d'] = gy_d
        param['ls'] = ls
        param['USE_PLATFORM'] = USE_PLATFORM

        return param
    
    def finalize(param):
        #param['noisy_image_buffer'].release()
        param['basic_image_buffer'].release()
        param['wiener_image_buffer'].release()
        param['similar_coords_buffer'].release()
        param['block_counts_buffer'].release()
        del param['program']
        del param['queue']
        del param['context']

    def process_channel(channel_data, param):
        height, width = channel_data.shape
        context = param['context']
        queue = param['queue']
        program = param['program']
        dist_kernel = param['dist_kernel']
        basic_kernel = param['basic_kernel']
        wiener_kernel = param['wiener_kernel']
        #noisy_image_buffer = param['noisy_image_buffer']
        basic_image_buffer = param['basic_image_buffer']
        wiener_image_buffer = param['wiener_image_buffer']
        similar_coords_buffer = param['similar_coords_buffer']
        block_counts_buffer = param['block_counts_buffer']
        tot_items = param['tot_items']
        tot_items_d = param['tot_items_d']
        gx = param['gx']
        gy = param['gy']
        gx_d = param['gx_d']
        gy_d = param['gy_d']
        ls = param['ls']
        USE_PLATFORM = param['USE_PLATFORM']


        # ノイズ画像をデバイスにコピー
        #cl.enqueue_copy(queue, noisy_image_buffer, np.ascontiguousarray(channel_data), origin=(0, 0), region=(width, height))
        image_format = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)
        noisy_image_buffer = cl.create_image(context, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, image_format, shape=(width, height), hostbuf=np.ascontiguousarray(channel_data))

        
        # カーネル引数の設定
        hard_threshold = np.float32(D_THRESHOLD_1 * BLOCK_SIZE_SQ)
        wiener_threshold = np.int32(D_THRESHOLD_2 * BLOCK_SIZE_SQ)
        max_block_count_1 = np.int32(MAX_BLOCK_COUNT_1)
        max_block_count_2 = np.int32(MAX_BLOCK_COUNT_2)
        window_step_size_1 = np.int32(WINDOW_STEP_SIZE_1)
        window_step_size_2 = np.int32(WINDOW_STEP_SIZE_2)
                
        # 距離計算カーネル引数
        dist_kernel.set_args(
            noisy_image_buffer, similar_coords_buffer, block_counts_buffer,
            hard_threshold, max_block_count_1, window_step_size_1
        )

        # 基本フィルタカーネル引数
        if USE_PLATFORM == PLATFORM_ATI:
            accu_size = tot_items * SPLIT_SIZE_X * SPLIT_SIZE_Y * np.dtype(np.float32).itemsize
            accumulator_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=accu_size)
            weight_map_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=accu_size)
            basic_kernel.set_args(
                noisy_image_buffer, basic_image_buffer, similar_coords_buffer, block_counts_buffer,
                np.int32(gx_d), np.int32(tot_items_d), accumulator_buffer, weight_map_buffer
            )
        else:
            basic_kernel.set_args(
                noisy_image_buffer, basic_image_buffer, similar_coords_buffer, block_counts_buffer,
                np.int32(gx_d), np.int32(tot_items_d)
            )
        
        # カーネル実行
        gs_d = (gx_d, gy_d)  # 距離カーネルのグローバルサイズ
        gs = (gx, gy)        # フィルタカーネルのグローバルサイズ
        offset = (0, 0)      # オフセット
        
        # カーネル実行
        dist_event = cl.enqueue_nd_range_kernel(queue, dist_kernel, gs_d, ls, global_work_offset=offset)                
        basic_event = cl.enqueue_nd_range_kernel(queue, basic_kernel, gs, ls, global_work_offset=offset)

        # 距離計算カーネル引数の更新
        dist_kernel.set_args(
            basic_image_buffer, similar_coords_buffer, block_counts_buffer,
            wiener_threshold, max_block_count_2, window_step_size_2
        )
        
        # Wienerフィルタカーネル引数
        if USE_PLATFORM == PLATFORM_ATI:
            wiener_kernel.set_args(
                noisy_image_buffer, basic_image_buffer, wiener_image_buffer, similar_coords_buffer, block_counts_buffer,
                np.int32(gx_d), np.int32(tot_items_d), accumulator_buffer, weight_map_buffer
            )
        else:
            wiener_kernel.set_args(
                noisy_image_buffer, basic_image_buffer, wiener_image_buffer, similar_coords_buffer, block_counts_buffer,
                np.int32(gx_d), np.int32(tot_items_d)
            )
        
        # カーネル実行
        dist_event2 = cl.enqueue_nd_range_kernel(queue, dist_kernel, gs_d, ls, global_work_offset=offset)        
        wiener_event = cl.enqueue_nd_range_kernel(queue, wiener_kernel, gs, ls, global_work_offset=offset)        
        
        # 結果の読み取り
        result_channel = np.zeros((height, width), dtype=np.float32)
        cl.enqueue_copy(queue, result_channel, wiener_image_buffer, origin=(0, 0), region=(width, height))

        queue.finish()
        noisy_image_buffer.release()
        return result_channel
    

    # BLOCK_SIZE単位にする
    image, org_size = adjust_to_multiple(image, BLOCK_SIZE)

    # 初期化
    param = initialize(image)

    # 各チャンネルを処理
    if image.ndim == 3 and image.shape[2] == 3:
        result = np.zeros_like(image)
        for c in range(3):  # RGB 3チャンネル
            result[:,:,c] = process_channel(image[:,:,c], param)            
    else:
        result = process_channel(image, param)

    # 終了
    finalize(param)

    # 元のサイズに戻す
    result = restore_original_size(result, org_size)

            
    return result



# 使用例
if __name__ == "__main__":
    # 入力画像読み込み（例：512x512 RGB画像）
    input_rgb = imageio.v2.imread("your_imageA.jpg").astype(np.float32) / 255.0
    
    # ノイズ除去実行
    starttime = time.time()
    denoised = bm3d_denoise(input_rgb, sigma=0.1)
    print(f"実行時間: {time.time()-starttime:.6f}秒")
    # 5.439034秒
    
    # 結果保存
    denoised = (denoised * 255).astype(np.uint8)
    imageio.imwrite("denoised_result.png", denoised)
    
    print("Denoising completed with both Basic and Wiener filters.")
