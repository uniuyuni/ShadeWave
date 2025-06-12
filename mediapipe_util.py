
import cv2
import numpy as np
import mediapipe as mp

def adjust_eyes_scale(image, scale=1.2, radius=50):
    # ランドマーク取得
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    results = face_mesh.process((image * 255).astype(np.uint8))

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        h, w, _ = image.shape
        
        # 左目のランドマーク抽出 :cite[8]
        left_eye_indices = list(mp_face_mesh.FACEMESH_LEFT_EYE)
        left_eye_points = np.array([(int(landmarks.landmark[i].x * w), 
                                    int(landmarks.landmark[i].y * h)) 
                                    for idx_pair in left_eye_indices for i in idx_pair])
        
        # 目の中心計算
        center = np.mean(left_eye_points, axis=0).astype(int)
        
        # 拡大マップ作成 (scale: 拡大率, radius: 効果範囲)
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)
        
        for y in range(h):
            for x in range(w):
                dx = x - center[0]
                dy = y - center[1]
                distance = dx**2 + dy**2
                
                if distance < radius**2:
                    # 中心付近を拡大
                    map_x[y, x] = center[0] + dx / scale
                    map_y[y, x] = center[1] + dy / scale
                else:
                    map_x[y, x] = x
                    map_y[y, x] = y
        
        # リマップで変形適用
        enlarged_eye = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

        return enlarged_eye
    
    return image

def adjust_face_oval(image, scale=0.8, debug=False):

    # ランドマーク取得
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        min_detection_confidence=0,  # 検出閾値（低くするとFPS↑）
        static_image_mode=True,        # 静止画モードで最適化
        refine_landmarks=True,          # ランドマークの精度を高める
    ) as face_mesh:
        # 検出
        results = face_mesh.process((image * 255).astype(np.uint8))

        # 適用度を下げるため、scaleを0.1倍している
        ratio = scale * 0.1

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            h, w = image.shape[:2]
            
            # 輪郭インデックスの取得（重複なし）
            oval_indices = np.unique([idx for pair in mp_face_mesh.FACEMESH_FACE_OVAL for idx in pair])
            
            # 浮動小数点数座標で取得
            oval_points = []
            for i in oval_indices:
                landmark = landmarks.landmark[i]
                oval_points.append([landmark.x * w, landmark.y * h])
                
            _oval_points = np.array(oval_points, dtype=np.float32)

            # 顔の中心計算
            center = np.mean(_oval_points, axis=0)

            # 固定点追加前の座標数
            l = len(oval_points)

            # 縮小後の座標計算
            new_oval_points = []
            for pt in _oval_points:
                direction = center - pt
                new_pt = pt + direction * ratio
                new_oval_points.append(new_pt)

                # 固定点追加
                FIX_POINT_SCALES = [-0.5, 0.2, 0.4, 0.8]
                for scale in FIX_POINT_SCALES:
                    oval_points.append(pt - direction * scale)

            # 固定点コピー
            for i in range(l*len(FIX_POINT_SCALES)):
                new_oval_points.append(oval_points[l+i])
            
            oval_points = np.array(oval_points, dtype=np.float32)
            new_oval_points = np.array(new_oval_points, dtype=np.float32)

            if debug:        
                # 変形前の点を描画
                for pt in oval_points:
                    cv2.circle(image, (int(pt[0]), int(pt[1])), 2, (0,255,0), -1)

                # 変形後の点を描画
                for pt in new_oval_points:
                    cv2.circle(image, (int(pt[0]), int(pt[1])), 2, (0,0,255), -1)
            
            # 入力形式を(1, N, 2)に変換
            source_pts = oval_points.reshape(1, -1, 2)
            target_pts = new_oval_points.reshape(1, -1, 2)
            
            # TPS変形
            tps = cv2.createThinPlateSplineShapeTransformer()
            tps.setRegularizationParameter(1e-3)
            matches = [cv2.DMatch(i, i, 0) for i in range(len(oval_points))]
            tps.estimateTransformation(target_pts, source_pts, matches)

            # 画像変形（背景処理追加）
            warped_image = tps.warpImage(image, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
            
            return warped_image
    
    return image
