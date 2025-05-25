import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import time

def create_ghost(
    image: np.ndarray,  # RGB float32 (0.0-1.0)
    light_source_coords: list[tuple[int, int]],
    global_intensity: float = 0.5,  # 全体的なゴーストの強度
    base_radius: int = 50,
    num_components: int = 4,  # ゴーストの構成要素（層）の数。数を増やすと複雑に。
    component_spread_factor: float = 0.3,  # 各成分の半径の広がり具合のランダム性
    blur_sigma: float = 10.0, # 各コンポーネント、または初期ゴースト層のぼかし
    chromatic_aberration_strength: float = 1.0,  # 色の広がり（虹色の幅）
    ghost_ring_thickness: float = 0.2, # ゴースト環の相対的な厚み (小さいほど細い)
    # --- レンズ特性シミュレーションパラメータ ---
    lens_center: tuple[int, int] = None,  # レンズの中心座標 (デフォルトは画像中心)
    radial_deformation_strength: float = 0.6,  # 光源からレンズ中心までの距離に応じたゴーストの変形強度 (0.0-1.0)
    max_eccentricity: float = 0.7,  # ゴーストが取りうる最大の楕円度 (0.0-1.0)
    max_offset_ratio_x: float = 1.2,  # 光源とレンズ中心間のX距離に対するゴーストXオフセットの比率
    max_offset_ratio_y: float = 1.2,  # 光源とレンズ中心間のY距離に対するゴーストYオフセットの比率
    rotation_angle: float = 0.0,  # ゴースト全体の回転角度 (度)
    curvature_strength: float = 0.0, # ゴーストの湾曲の強さ
    perspective_distortion: float = 0.0, # 遠近法によるゴーストの歪み
    ghost_decay_rate: float = 1.0, # 光源からの距離によるゴーストの減衰率 (小さいほど早く減衰)
    ghost_tail_strength: float = 0.0, # ゴーストの尾の強さ
    spherical_aberration_strength: float = 0.0, # 球面収差によるにじみ
    # --- ノイズと不規則性（トゲトゲ効果） ---
    post_blur_irregularity_strength: float = 1.0, # トゲトゲ効果の全体的な強度
    post_irregularity_noise_scale: float = 0.008, # トゲの発生頻度/密度 (小さいほど密に)
    post_irregularity_micro_displacement: float = 0.6, # トゲの根元の微細なずれ/広がり
    post_irregularity_spike_length: float = 0.6, # トゲの長さ
    post_irregularity_spike_thickness: float = 1.2, # トゲの太さ
    post_irregularity_blur_sigma: float = 0.5, # トゲトゲマップ自体のぼかし
    random_seed: int = None, # 乱数シード
) -> np.ndarray:
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0

    img_height, img_width = image.shape[:2]
    ghosted_image = image.copy()

    # 乱数シードの設定
    rng = np.random.default_rng(random_seed)

    # レンズ中心のデフォルト設定
    if lens_center is None:
        lens_center = (img_width // 2, img_height // 2)

    # 虹色の定義 (RGB)
    rainbow_colors_rgb = np.array([
        [1.0, 0.0, 0.0],    # 赤
        [1.0, 0.5, 0.0],    # 橙
        [1.0, 1.0, 0.0],    # 黄
        [0.0, 1.0, 0.0],    # 緑
        [0.0, 0.0, 1.0],    # 青
        [0.5, 0.0, 1.0],    # 藍
        [1.0, 0.0, 1.0],    # 紫
    ], dtype=np.float32)

    # 各光源に対してゴーストを生成
    for sx, sy in light_source_coords:
        # 光源とレンズ中心間の距離 (これは影響度計算のためなのでそのまま)
        dist_to_lens_center = math.sqrt((sx - lens_center[0])**2 + (sy - lens_center[1])**2)
        max_possible_dist = math.sqrt((img_width**2 + img_height**2)) / 2
        influence_factor = np.clip(dist_to_lens_center / max_possible_dist, 0.0, 1.0)

        # ゴーストのオフセットの計算 (max_offset_ratio_x/y を使用し、負の値も許容)
        base_offset_x = (lens_center[0] - sx) * max_offset_ratio_x
        base_offset_y = (lens_center[1] - sy) * max_offset_ratio_y

        current_light_source_ghost_pre_irregularity = np.zeros_like(image)
        spike_layer_total = np.zeros_like(image) # Accumulate spikes here

        # グリッドの作成 (一度だけ計算)
        x_coords, y_coords = np.meshgrid(np.arange(img_width), np.arange(img_height))

        # 光源からの距離マップを事前に計算
        dist_from_source = np.sqrt((x_coords - sx)**2 + (y_coords - sy)**2)

        for i in range(num_components):
            component_radius = base_radius * (0.8 + i * 0.1) * (1.0 + influence_factor * radial_deformation_strength * 0.5) \
                             * (1.0 + rng.uniform(-component_spread_factor, component_spread_factor))

            component_offset_x = base_offset_x * (1.0 + i * 0.2)
            component_offset_y = base_offset_y * (1.0 + i * 0.2)

            ghost_center_x = sx + component_offset_x
            ghost_center_y = sy + component_offset_y

            current_eccentricity = max_eccentricity * influence_factor * rng.uniform(0.8, 1.0)
            current_rotation_angle = rotation_angle + rng.uniform(-10, 10)

            major_axis = component_radius
            minor_axis = np.maximum(component_radius * (1.0 - current_eccentricity), 1.0)

            theta_rad = math.radians(current_rotation_angle + rng.uniform(-5, 5))
            cos_theta = math.cos(theta_rad)
            sin_theta = math.sin(theta_rad)

            dx_map = x_coords - ghost_center_x
            dy_map = y_coords - ghost_center_y

            if ghost_tail_strength > 0:
                tail_factor_map = np.clip((dist_from_source / (base_radius * 2.0)) * ghost_tail_strength, 0.0, 1.0)
                non_zero_dist_mask = dist_from_source > 0

                tail_offset_x_map = np.zeros_like(dx_map)
                tail_offset_y_map = np.zeros_like(dy_map)

                tail_offset_x_map[non_zero_dist_mask] = -(x_coords[non_zero_dist_mask] - sx) / dist_from_source[non_zero_dist_mask] * component_radius * 0.2 * tail_factor_map[non_zero_dist_mask]
                tail_offset_y_map[non_zero_dist_mask] = -(y_coords[non_zero_dist_mask] - sy) / dist_from_source[non_zero_dist_mask] * component_radius * 0.2 * tail_factor_map[non_zero_dist_mask]

                dx_map += tail_offset_x_map
                dy_map += tail_offset_y_map

            rotated_dx_map = dx_map * cos_theta + dy_map * sin_theta
            rotated_dy_map = -dx_map * sin_theta + dy_map * cos_theta

            dist_norm_map = np.sqrt((rotated_dx_map / major_axis)**2 + (rotated_dy_map / minor_axis)**2)

            perspective_factor_map = 1.0 - (dist_norm_map - 1.0) * perspective_distortion * influence_factor * np.sign(dist_norm_map - 1.0)
            dist_norm_map *= np.clip(perspective_factor_map, 0.5, 1.5)

            curvature_offset_x_map = curvature_strength * influence_factor * rotated_dx_map * dist_norm_map**2 * 0.1
            curvature_offset_y_map = curvature_strength * influence_factor * rotated_dy_map * dist_norm_map**2 * 0.1

            rotated_dx_map += curvature_offset_x_map
            rotated_dy_map += curvature_offset_y_map

            spherical_aberration_offset_map = spherical_aberration_strength * influence_factor * (dist_norm_map - 1.0) * rng.uniform(-0.5, 0.5)
            dist_norm_final_map = dist_norm_map + spherical_aberration_offset_map

            ring_distance = np.abs(dist_norm_final_map - 1.0)

            thickness_scale = np.maximum(0.0, 1.0 - ring_distance / (ghost_ring_thickness * (0.5 + chromatic_aberration_strength * 0.5)))

            color_mask = thickness_scale > 0

            if np.any(color_mask):
                normalized_distance_from_center = (dist_norm_final_map - 1.0) / ghost_ring_thickness
                scaled_distance_for_color = normalized_distance_from_center * chromatic_aberration_strength
                color_progress = (scaled_distance_for_color * 0.5 + 0.5)
                color_idx_map = color_progress * (len(rainbow_colors_rgb) - 1)
                color_idx_map = np.clip(color_idx_map, 0, len(rainbow_colors_rgb) - 1)

                color_idx_floor_map = np.clip(color_idx_map.astype(int), 0, len(rainbow_colors_rgb) - 1)
                color_idx_ceil_map = (color_idx_floor_map + 1) % len(rainbow_colors_rgb)
                frac_map = color_idx_map - color_idx_floor_map

                color_map = rainbow_colors_rgb[color_idx_floor_map] * (1 - frac_map)[:, :, np.newaxis] + \
                           rainbow_colors_rgb[color_idx_ceil_map] * frac_map[:, :, np.newaxis]

                edge_fade_width = ghost_ring_thickness * 0.3
                edge_fade_factor = np.where(
                    ring_distance < ghost_ring_thickness - edge_fade_width,
                    1.0,
                    np.maximum(0.0, (ghost_ring_thickness - ring_distance) / edge_fade_width)
                )

                decay_factor_map = np.exp(-(dist_from_source / (base_radius * ghost_decay_rate))**2)

                component_intensity_decay = math.exp(-i * 0.2)
                current_component_blur_sigma = blur_sigma * (1.0 + i * 0.5)

                alpha_map = thickness_scale * edge_fade_factor * decay_factor_map * global_intensity * component_intensity_decay
                alpha_map = np.clip(alpha_map, 0.0, 1.0)

                min_alpha_threshold = 0.002
                strong_alpha_mask = alpha_map > min_alpha_threshold

                if np.any(strong_alpha_mask):
                    component_ghost_layer = color_map * alpha_map[:, :, np.newaxis]
                    component_ghost_layer[~strong_alpha_mask] = 0

                    # --- Moved Post-processing Irregularity (Spikes) inside the component loop ---
                    if post_blur_irregularity_strength > 0 and np.any(component_ghost_layer > 0):
                        component_alpha_channel = np.mean(component_ghost_layer, axis=2)
                        
                        # Use a higher threshold for Canny to detect sharper edges on individual components
                        edges = cv2.Canny((component_alpha_channel * 255).astype(np.uint8), 50, 150) # Adjusted thresholds
                        edge_coords_y, edge_coords_x = np.where(edges > 0)

                        if len(edge_coords_x) > 0:
                            num_spikes_to_generate = int(len(edge_coords_x) * post_irregularity_noise_scale * 100)
                            num_spikes_to_generate = min(num_spikes_to_generate, len(edge_coords_x))
                            
                            if num_spikes_to_generate > 0:
                                spike_indices = rng.choice(len(edge_coords_x), num_spikes_to_generate, replace=False)
                                
                                for idx_s in spike_indices: # Changed loop variable to avoid conflict
                                    ex_orig, ey_orig = edge_coords_x[idx_s], edge_coords_y[idx_s]

                                    displacement_angle = rng.uniform(0, 2 * math.pi)
                                    disp_magnitude = rng.uniform(0, post_irregularity_micro_displacement * 5)

                                    ex = ex_orig + int(disp_magnitude * math.cos(displacement_angle))
                                    ey = ey_orig + int(disp_magnitude * math.sin(displacement_angle))
                                    
                                    if not (0 <= ex < img_width and 0 <= ey < img_height):
                                        continue

                                    base_color = component_ghost_layer[ey, ex]
                                    base_alpha = component_alpha_channel[ey, ex]

                                    if base_alpha > 0.01:
                                        dir_x = ex - ghost_center_x # Spikes point away from component center
                                        dir_y = ey - ghost_center_y

                                        length = math.sqrt(dir_x**2 + dir_y**2)
                                        if length > 0:
                                            dir_x /= length
                                            dir_y /= length
                                        else:
                                            angle = rng.uniform(0, 2 * math.pi)
                                            dir_x = math.cos(angle)
                                            dir_y = math.sin(angle)

                                        spike_len = rng.uniform(post_irregularity_spike_length * 0.5, post_irregularity_spike_length * 1.5) * base_radius * 0.1 * 1.5
                                        spike_thick = np.maximum(0.5, rng.uniform(post_irregularity_spike_thickness * 0.5, post_irregularity_spike_thickness * 1.5) * 0.5 * 1.5)

                                        end_x = ex + dir_x * spike_len
                                        end_y = ey + dir_y * spike_len
                                        
                                        num_steps = max(2, int(spike_len))
                                        for step in range(num_steps + 1):
                                            t = step / num_steps
                                            current_x = int(ex + dir_x * spike_len * t)
                                            current_y = int(ey + dir_y * spike_len * t)

                                            if 0 <= current_x < img_width and 0 <= current_y < img_height:
                                                spike_intensity = base_alpha * (1.0 - t) * post_blur_irregularity_strength * 2.0
                                                
                                                for dy_t in range(max(-1, int(-spike_thick)), max(2, int(spike_thick) + 1)):
                                                    for dx_t in range(max(-1, int(-spike_thick)), max(2, int(spike_thick) + 1)):
                                                        px, py = current_x + dx_t, current_y + dy_t
                                                        
                                                        if 0 <= px < img_width and 0 <= py < img_height:
                                                            dist_from_line_center = math.sqrt(dx_t**2 + dy_t**2)
                                                            if dist_from_line_center < spike_thick / 2.0:
                                                                fade_factor_thickness = 1.0 - (dist_from_line_center / (spike_thick / 2.0))
                                                                spike_layer_total[py, px] = np.maximum(spike_layer_total[py, px], base_color * spike_intensity * fade_factor_thickness)
                    # --- End of moved block ---

                    if current_component_blur_sigma > 0:
                        ksize = max(3, int(current_component_blur_sigma * 2) + 1)
                        if ksize % 2 == 0:
                            ksize += 1
                        if ksize > 1:
                            component_ghost_layer = cv2.GaussianBlur(component_ghost_layer, (ksize, ksize), current_component_blur_sigma)

                    current_light_source_ghost_pre_irregularity += component_ghost_layer
        
        # Apply the accumulated spike layer here, after all components for this light source are processed
        # and then blur the total spike layer if needed
        if post_irregularity_blur_sigma > 0:
            ksize_spike_blur = max(3, int(post_irregularity_blur_sigma * 2) + 1)
            if ksize_spike_blur % 2 == 0:
                ksize_spike_blur += 1
            if ksize_spike_blur > 1:
                spike_layer_total = cv2.GaussianBlur(spike_layer_total, (ksize_spike_blur, ksize_spike_blur), post_irregularity_blur_sigma)
        
        current_light_source_ghost_pre_irregularity = np.clip(current_light_source_ghost_pre_irregularity + spike_layer_total, 0.0, 1.0)


        alpha = np.sum(current_light_source_ghost_pre_irregularity, axis=2)
        alpha = np.clip(alpha * global_intensity, 0.0, 1.0)
        alpha_channel = alpha[:, :, np.newaxis]

        # Use alpha blending to overlay the ghost effect onto the original image
        # This will correctly handle cases where ghost intensity might be high causing clipping
        ghosted_image = (ghosted_image * (1.0 - alpha_channel)) + (current_light_source_ghost_pre_irregularity * alpha_channel)

    return np.clip(ghosted_image, 0.0, 1.0)

if __name__ == "__main__":
    img_height, img_width = 500, 700

    def create_base_image(light_coords, height, width):
        img = np.zeros((height, width, 3), dtype=np.float32)
        for lp_x, lp_y in light_coords:
            cv2.circle(img, (lp_x, lp_y), 15, (1.0, 1.0, 1.0), -1)
        return img

    print("Generating various ghost effects with improved irregularities and shapes:")

    # サンプル1: 添付画像のような半円・欠けのあるギザギザゴースト
    light_pos_for_sample1 = (600, 100) # 右上
    base_image_sample1 = create_base_image([light_pos_for_sample1], img_height, img_width)

    print("\n--- Sample 1: Half-circle/Partial Ghost with Sharp Radial Irregularities (like DSC01878.jpg) ---")
    start_time = time.time()
    ghosted_image_sample1 = create_ghost(
        base_image_sample1,
        light_source_coords=[light_pos_for_sample1],
        global_intensity=0.7,
        base_radius=180, # 半径を大きくして画面外にはみ出させる
        num_components=1,
        component_spread_factor=0.1,
        blur_sigma=1.0, # 初期ブラーを弱め
        chromatic_aberration_strength=1.5,
        ghost_ring_thickness=0.5, # Increased thickness to test the fix
        lens_center=(img_width // 2, img_height // 2),
        radial_deformation_strength=0.8,
        max_eccentricity=0.6,
        max_offset_ratio_x=1.5,
        max_offset_ratio_y=1.5,
        ghost_decay_rate=1.5,
        perspective_distortion=0.3,
        curvature_strength=0.1,
        ghost_tail_strength=0.0,
        spherical_aberration_strength=0.05,
        post_blur_irregularity_strength=1.5, # ギザギザ強度を上げる
        post_irregularity_noise_scale=0.01, # 放射状ノイズの頻度 (小さいほど粗いギザギザ)
        post_irregularity_micro_displacement=1.0, # ギザギザによる微細な位置ずれ強度
        post_irregularity_blur_sigma=0.0, # ギザギザ後のブラーはなしでシャープに
        random_seed=42 # シード固定で再現性確保
    )
    end_time = time.time()

    plt.figure(figsize=(10, 7))
    plt.imshow(ghosted_image_sample1)
    plt.title('Sample 1: Half-circle/Partial Ghost with Sharp Radial Irregularities (like DSC01878.jpg)')
    plt.axis('off')
    plt.show()
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print("-" * 50)

    # サンプル2: 複数の光源と異なる形状のゴースト
    light_positions_multi = [
        (600, 100), # 右上
        (50, 400),  # 左下
        (300, 250)  # 中央寄り
    ]
    base_image_multi = create_base_image(light_positions_multi, img_height, img_width)

    print("\n--- Sample 2: Multiple Ghosts with Varied Radial Irregularities and Shapes ---")
    start_time = time.time()

    final_multi_ghost_image = base_image_multi.copy()

    for idx, (lp_x, lp_y) in enumerate(light_positions_multi):
        single_light_base_image = np.zeros((img_height, img_width, 3), dtype=np.float32)
        cv2.circle(single_light_base_image, (lp_x, lp_y), 15, (1.0, 1.0, 1.0), -1)

        ghost_for_light = create_ghost(
            single_light_base_image,
            light_source_coords=[(lp_x, lp_y)],
            global_intensity=0.6,
            base_radius=120,
            num_components=2,
            blur_sigma=3.0,
            chromatic_aberration_strength=1.0,
            ghost_ring_thickness=0.3, # Increased thickness here too
            lens_center=(img_width // 2, img_height // 2),
            radial_deformation_strength=0.7,
            max_eccentricity=0.5,
            max_offset_ratio_x=1.0,
            max_offset_ratio_y=1.0,
            ghost_decay_rate=2.0,
            perspective_distortion=0.2,
            curvature_strength=0.1,
            ghost_tail_strength=0.0,
            spherical_aberration_strength=0.1,
            post_blur_irregularity_strength=1.0,
            post_irregularity_noise_scale=0.015, # 放射状ノイズの頻度
            post_irregularity_micro_displacement=0.8,
            post_irregularity_blur_sigma=0.5, # 最終ブラーはごくわずか
            random_seed=42 + idx # 各ゴーストで異なるシード
        )
        final_multi_ghost_image += ghost_for_light

    final_multi_ghost_image = np.clip(final_multi_ghost_image, 0.0, 1.0)

    end_time = time.time()

    plt.figure(figsize=(10, 7))
    plt.imshow(final_multi_ghost_image)
    plt.title('Sample 2: Multiple Ghosts with Varied Radial Irregularities and Shapes')
    plt.axis('off')
    plt.show()
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print("-" * 50)

    # --- 追加サンプル 3-13 ---
    sample_params = {
        "Sample 3: Soft, Large Flare (like DSC02027.jpg)": {
            "global_intensity": 0.8, "base_radius": 250, "num_components": 1,
            "blur_sigma": 3.0, "chromatic_aberration_strength": 2.5, "ghost_ring_thickness": 0.5, # Changed
            "radial_deformation_strength": 0.3, "max_eccentricity": 0.2, "max_offset_ratio_x": 2.0, "max_offset_ratio_y": 2.0,
            "ghost_decay_rate": 0.8, "perspective_distortion": 0.1, "curvature_strength": 0.05,
            "ghost_tail_strength": 0.0, "spherical_aberration_strength": 0.0,
            "post_blur_irregularity_strength": 0.0, # ギザギザなし
            "post_irregularity_blur_sigma": 15.0, # 最終ブラー強め
            "light_source_coords": [(img_width // 2, img_height // 2)], # 中央光源
            "lens_center": (img_width // 2, img_height // 2),
            "random_seed": 100
        },
        "Sample 4: Circular Ghost with Strong Chromatic Aberration & Sharp Radial Irregularities": {
            "global_intensity": 0.6, "base_radius": 80, "num_components": 5,
            "blur_sigma": 1.0, "chromatic_aberration_strength": 2.5, "ghost_ring_thickness": 0.2, # Changed
            "lens_center": (img_width // 2, img_height // 2), # レンズ中心
            "radial_deformation_strength": 0.0, # 円形を保つ
            "max_eccentricity": 0.0, # 円形を保つ
            "max_offset_ratio_x": 1.5, "max_offset_ratio_y": 1.5,
            "ghost_decay_rate": 1.0, "perspective_distortion": 0.0, "curvature_strength": 0.0,
            "ghost_tail_strength": 0.1, "spherical_aberration_strength": 0.2,
            "post_blur_irregularity_strength": 0.8, "post_irregularity_noise_scale": 0.005, # より細かいギザギザ (放射状ノイズの頻度)
            "post_irregularity_micro_displacement": 0.7, # エッジ強調
            "post_irregularity_blur_sigma": 0.0, # シャープに
            "light_source_coords": [(img_width // 4, img_height // 4)],
            "random_seed": 101
        },
        "Sample 5: Subtle, Multiple Small Circular Ghosts": {
            "global_intensity": 0.3, "base_radius": 30, "num_components": 7,
            "blur_sigma": 0.5, "chromatic_aberration_strength": 0.8, "ghost_ring_thickness": 0.5, # Changed
            "lens_center": (img_width // 2, img_height // 2),
            "radial_deformation_strength": 0.0, # 円形を保つ
            "max_eccentricity": 0.0, # 円形を保つ
            "max_offset_ratio_x": 0.8, "max_offset_ratio_y": 0.8,
            "ghost_decay_rate": 3.0, "perspective_distortion": 0.2, "curvature_strength": 0.1,
            "ghost_tail_strength": 0.05, "spherical_aberration_strength": 0.05,
            "post_blur_irregularity_strength": 0.5, "post_irregularity_noise_scale": 0.01, # 細かいノイズ (放射状ノイズの頻度)
            "post_irregularity_micro_displacement": 0.4,
            "post_irregularity_blur_sigma": 0.2, # ごくわずか
            "light_source_coords": [(50, 200), (200, 50), (400, 400), (650, 150)], # 複数光源
            "random_seed": 102
        },
        "Sample 6: Perfect Half-Circle Ghost": {
            "global_intensity": 0.7, "base_radius": 150, "num_components": 1,
            "blur_sigma": 5.0, "chromatic_aberration_strength": 1.0, "ghost_ring_thickness": 0.5, # Changed
            "lens_center": (img_width // 2, img_height // 2),
            "radial_deformation_strength": 0.0, # 円形を保つ
            "max_eccentricity": 0.0, # 円形を保つ
            "max_offset_ratio_x": 0.5, "max_offset_ratio_y": 0.5,
            "ghost_decay_rate": 1.0, "perspective_distortion": 0.0, "curvature_strength": 0.0,
            "ghost_tail_strength": 0.0, "spherical_aberration_strength": 0.0,
            "post_blur_irregularity_strength": 0.0, # ギザギザなし
            "post_irregularity_blur_sigma": 0.0,
            "light_source_coords": [(img_width // 2 + 100, img_height // 2)], # 中央より右に光源
            "random_seed": 103
        },
        "Sample 7: Circular Ghost with 'Urchin-like' Radial Irregularities": { # ウニのようなギザギザ
            "global_intensity": 0.8, "base_radius": 120, "num_components": 2,
            "blur_sigma": 1.0, "chromatic_aberration_strength": 1.5, "ghost_ring_thickness": 0.3, # Changed
            "lens_center": (img_width // 2, img_height // 2),
            "radial_deformation_strength": 0.0, # 円形を保つ
            "max_eccentricity": 0.0, # 円形を保つ
            "max_offset_ratio_x": 1.0, "max_offset_ratio_y": 1.0,
            "ghost_decay_rate": 1.5, "perspective_distortion": 0.0, "curvature_strength": 0.0,
            "ghost_tail_strength": 0.0, "spherical_aberration_strength": 0.05,
            "post_blur_irregularity_strength": 2.5, # 非常に強いギザギザ
            "post_irregularity_noise_scale": 0.005, # 超細かいノイズ (放射状ノイズの頻度)
            "post_irregularity_micro_displacement": 2.5, # 強い位置ずれ
            "post_irregularity_blur_sigma": 0.0, # 最終ブラーなしでシャープに
            "light_source_coords": [(img_width // 2, img_height // 2)],
            "random_seed": 104
        },
        "Sample 8: Ring-like Ghost with Random Partial Cuts and Radial Irregularities": {
            "global_intensity": 0.6, "base_radius": 100, "num_components": 3,
            "blur_sigma": 2.0, "chromatic_aberration_strength": 1.2, "ghost_ring_thickness": 0.3, # Changed
            "lens_center": (img_width // 2, img_height // 2),
            "radial_deformation_strength": 0.0,
            "max_eccentricity": 0.0,
            "max_offset_ratio_x": 1.0, "max_offset_ratio_y": 1.0,
            "ghost_decay_rate": 2.0, "perspective_distortion": 0.0, "curvature_strength": 0.0,
            "ghost_tail_strength": 0.0, "spherical_aberration_strength": 0.0,
            "post_blur_irregularity_strength": 1.0, "post_irregularity_noise_scale": 0.008,
            "post_irregularity_micro_displacement": 0.6,
            "post_irregularity_blur_sigma": 0.5, # 少しだけブラー
            "light_source_coords": [(img_width // 2, img_height // 2)],
            "random_seed": 105
        },
        "Sample 9: Multiple Thin Concentric Circular Rings": {
            "global_intensity": 0.5, "base_radius": 90, "num_components": 6,
            "component_spread_factor": 0.05, # 成分の広がりを小さく
            "blur_sigma": 0.5, "chromatic_aberration_strength": 1.0, "ghost_ring_thickness": 0.02, # 薄いリング
            "lens_center": (img_width // 2, img_height // 2),
            "radial_deformation_strength": 0.0, "max_eccentricity": 0.0, "max_offset_ratio_x": 1.0, "max_offset_ratio_y": 1.0, # 円形
            "ghost_decay_rate": 2.5, "perspective_distortion": 0.0, "curvature_strength": 0.0,
            "ghost_tail_strength": 0.0, "spherical_aberration_strength": 0.0,
            "post_blur_irregularity_strength": 0.0, # ギザギザなし
            "post_irregularity_blur_sigma": 0.0, # 最終ブラーなし
            "light_source_coords": [(img_width // 2, img_height // 2)],
            "random_seed": 106
        },
        "Sample 10: Circular Ghost with Strong Diffraction & Subtle Radial Irregularities": {
            "global_intensity": 0.7, "base_radius": 130, "num_components": 2,
            "blur_sigma": 1.0, "chromatic_aberration_strength": 1.3, "ghost_ring_thickness": 0.3, # Changed
            "lens_center": (img_width // 2, img_height // 2),
            "radial_deformation_strength": 0.0,
            "max_eccentricity": 0.0,
            "max_offset_ratio_x": 1.0, "max_offset_ratio_y": 1.0,
            "ghost_decay_rate": 1.8, "perspective_distortion": 0.0, "curvature_strength": 0.0,
            "ghost_tail_strength": 0.0, "spherical_aberration_strength": 0.08,
            "post_blur_irregularity_strength": 1.2, # ギザギザ強調
            "post_irregularity_noise_scale": 0.005, # より細かいノイズ (放射状ノイズの頻度)
            "post_irregularity_micro_displacement": 0.8,
            "post_irregularity_blur_sigma": 0.0, # シャープに
            "light_source_coords": [(img_width // 2, img_height // 2)],
            "random_seed": 107
        },
        "Sample 11: Muted, Atmospheric Circular Halo": {
            "global_intensity": 0.2, "base_radius": 200, "num_components": 1,
            "blur_sigma": 40.0, "chromatic_aberration_strength": 0.3, "ghost_ring_thickness": 1.0,
            "lens_center": (img_width // 2, img_height // 2),
            "radial_deformation_strength": 0.0,
            "max_eccentricity": 0.0,
            "max_offset_ratio_x": 1.0, "max_offset_ratio_y": 1.0,
            "ghost_decay_rate": 0.3, "perspective_distortion": 0.0, "curvature_strength": 0.0,
            "ghost_tail_strength": 0.0, "spherical_aberration_strength": 0.0,
            "post_blur_irregularity_strength": 0.0,
            "post_irregularity_blur_sigma": 20.0,
            "light_source_coords": [(img_width // 2, img_height // 2)],
            "random_seed": 108
        },
        "Sample 12: Circular Ghost with Asymmetrical Cut and Strong Edge Blur": {
            "global_intensity": 0.7, "base_radius": 160, "num_components": 1,
            "blur_sigma": 5.0, "chromatic_aberration_strength": 1.0, "ghost_ring_thickness": 0.5, # Changed
            "lens_center": (img_width // 2, img_height // 2),
            "radial_deformation_strength": 0.0, # 円形を保つ
            "max_eccentricity": 0.0, # 円形を保つ
            "max_offset_ratio_x": 1.0, "max_offset_ratio_y": 1.0,
            "ghost_decay_rate": 1.0, "perspective_distortion": 0.0, "curvature_strength": 0.0,
            "ghost_tail_strength": 0.0, "spherical_aberration_strength": 0.0,
            "post_blur_irregularity_strength": 0.0, # ギザギザなし
            "post_irregularity_blur_sigma": 0.0,
            "light_source_coords": [(img_width // 2 + 100, img_height // 2)], # 中央より右に光源
            "random_seed": 109 # 異なるシードでカット角度が変わる
        },
        "Sample 13: Ring-like Ghost with Forced Partial Cut and Radial Irregularities (like Sample 8, but partial)": {
            "global_intensity": 0.6, "base_radius": 100, "num_components": 3,
            "blur_sigma": 2.0, "chromatic_aberration_strength": 1.2, "ghost_ring_thickness": 0.3, # Changed
            "lens_center": (img_width // 2, img_height // 2),
            "radial_deformation_strength": 0.0,
            "max_eccentricity": 0.0,
            "max_offset_ratio_x": 1.0, "max_offset_ratio_y": 1.0,
            "ghost_decay_rate": 2.0, "perspective_distortion": 0.0, "curvature_strength": 0.0,
            "ghost_tail_strength": 0.0, "spherical_aberration_strength": 0.0,
            "post_blur_irregularity_strength": 1.0, "post_irregularity_noise_scale": 0.008,
            "post_irregularity_micro_displacement": 0.6,
            "post_irregularity_blur_sigma": 0.5,
            "light_source_coords": [(img_width // 2 + 50, img_height // 2 - 50)], # 少しずらした光源
            "random_seed": 110 # 新しいシード
        },
    }

    for name, params in sample_params.items():
        print(f"\n--- {name} ---\n")
        current_light_coords = params.pop("light_source_coords")
        current_lens_center = params.pop("lens_center", None)
        current_random_seed = params.pop("random_seed", None)

        current_base_image = create_base_image(current_light_coords, img_height, img_width)

        start_time = time.time()
        ghosted_image = create_ghost(
            current_base_image,
            light_source_coords=current_light_coords,
            lens_center=current_lens_center,
            random_seed=current_random_seed,
            **params
        )
        end_time = time.time()

        plt.figure(figsize=(10, 7))
        plt.imshow(ghosted_image)
        plt.title(name)
        plt.axis('off')
        plt.show()
        print(f"Processing time: {end_time - start_time:.2f} seconds")
        print("-" * 50)