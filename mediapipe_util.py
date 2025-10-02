
import cv2
import numpy as np
import mediapipe as mp

EXCLUDE_POINT_INDICES_FOR_JAWLINE = [0, 25, 22, 24, 20, 19, 26, 18, 32, 26, 35, 5, 7, 4, 6, 2, 1, 15, 8]
EXCLUDE_POINT_INDICES_FOR_JAW = [0, 25, 22, 24, 20, 19, 26, 18, 32, 26, 35, 5, 7, 4, 6, 2, 1, 15, 8, 5, 9, 3, 16, 10, 13, 23, 27, 21, 33, 28, 31, 30, 12]

FIX_POINT_SCALES_FOR_OVAL = [-0.5, 0.2, 0.4, 0.8]
FIX_POINT_SCALES_FOR_EYES = [-0.8, 0.4, 0.8, 1.6]
FIX_POINT_SCALES_FOR_LIPS = [-0.5, 0.8, 1.6]


def setup_face_mesh(image):

    # ランドマーク取得
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0,  # 検出閾値（低くするとFPS↑）
        static_image_mode=True,        # 静止画モードで最適化
        refine_landmarks=True,          # ランドマークの精度を高める
    )
    
    # 検出
    results = face_mesh.process((image * 255).astype(np.uint8))

    return (mp_face_mesh, face_mesh, results)

def clear_face_mesh(fms):
    mp_face_mesh, face_mesh, results = fms
    face_mesh.close()
    del face_mesh

def adjust_face_jawline(fms, image, scale, debug=False):
    mp_face_mesh, face_mesh, results = fms

    return adjust_face_mesh(fms, image, scale, None, mp_face_mesh.FACEMESH_FACE_OVAL, EXCLUDE_POINT_INDICES_FOR_JAWLINE, FIX_POINT_SCALES_FOR_OVAL, debug)

def adjust_face_jaw(fms, image, scale, debug=False):
    mp_face_mesh, face_mesh, results = fms

    return adjust_face_mesh(fms, image, scale, None, mp_face_mesh.FACEMESH_FACE_OVAL, EXCLUDE_POINT_INDICES_FOR_JAW, FIX_POINT_SCALES_FOR_OVAL, debug)

def adjust_left_eye(fms, image, scale, debug=False):
    mp_face_mesh, face_mesh, results = fms

    return adjust_face_mesh(fms, image, -scale, (0.4, 0.4), mp_face_mesh.FACEMESH_LEFT_EYE, None, FIX_POINT_SCALES_FOR_EYES, debug)

def adjust_right_eye(fms, image, scale, debug=False):
    mp_face_mesh, face_mesh, results = fms

    return adjust_face_mesh(fms, image, -scale, (0.4, 0.4), mp_face_mesh.FACEMESH_RIGHT_EYE, None, FIX_POINT_SCALES_FOR_EYES, debug)

def adjust_lips(fms, image, scale, debug=False):
    mp_face_mesh, face_mesh, results = fms

    return adjust_face_mesh(fms, image, -scale, (0.2, 0.2), mp_face_mesh.FACEMESH_LIPS, None, FIX_POINT_SCALES_FOR_LIPS, debug)

def adjust_face_mesh(fms, image, scale, org_scale, mesh, exclude_point_indeces, fix_point_scales, debug=False):

    if scale == 0:
        return image

    mp_face_mesh, face_mesh, results = fms

    # 適用度を下げるため、scaleを0.1倍している
    ratio = scale * 0.1

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]

        h, w = image.shape[:2]
        
        # 輪郭インデックスの取得（重複なし）
        oval_indices = np.unique([idx for pair in mesh for idx in pair])
        
        # 浮動小数点数座標で取得
        oval_points = []
        for i in oval_indices:
            landmark = landmarks.landmark[i]
            oval_points.append([landmark.x * w, landmark.y * h])

        # 一旦ndarrayに変換
        _oval_points = np.array(oval_points, dtype=np.float32)

        # 顔の中心計算
        center = np.mean(_oval_points, axis=0)

        # 元座標を拡張
        if org_scale:
            org_scale = -np.array(org_scale) # イメージと逆
            new_oval_points = []
            for pt in oval_points:
                direction = center - pt
                new_pt = pt + direction * org_scale
                new_oval_points.append(new_pt)
            oval_points = new_oval_points

        # もう一回
        _oval_points = np.array(oval_points, dtype=np.float32)

        # 固定点追加前の座標数
        l = len(oval_points)

        # 縮小後の座標計算
        new_oval_points = []
        for pt in _oval_points:
            direction = center - pt
            new_pt = pt + direction * ratio
            new_oval_points.append(new_pt)

            # 固定点追加
            if fix_point_scales:
                for scale in fix_point_scales:
                    oval_points.append(pt - direction * scale)

        # 固定点コピー
        if fix_point_scales:
            for i in range(l*len(fix_point_scales)):
                new_oval_points.append(oval_points[l+i])

        # 下輪郭以外を削除
        if exclude_point_indeces:
            oval_points = [n for i, n in enumerate(oval_points) if i not in exclude_point_indeces]
            new_oval_points = [n for i, n in enumerate(new_oval_points) if i not in exclude_point_indeces]

        oval_points = np.array(oval_points, dtype=np.float32)
        new_oval_points = np.array(new_oval_points, dtype=np.float32)

        if debug:        
            # 変形前の点を描画
            for pt in oval_points:
                cv2.circle(image, (int(pt[0]), int(pt[1])), 2, (0,255,0), -1)

            # 変形後の点を描画
            for pt in new_oval_points:
                cv2.circle(image, (int(pt[0]), int(pt[1])), 2, (0,0,255), -1)

            # 点番号を描画
            #for i, pt in enumerate(_oval_points):
            #    cv2.putText(image, f"{i}", (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0))
        
        # 入力形式を(1, N, 2)に変換
        source_pts = oval_points.reshape(1, -1, 2)
        target_pts = new_oval_points.reshape(1, -1, 2)
        
        # TPS変形
        tps = cv2.createThinPlateSplineShapeTransformer(1e-3)
        matches = [cv2.DMatch(i, i, 0) for i in range(len(oval_points))]
        tps.estimateTransformation(target_pts, source_pts, matches)

        # 画像変形（背景処理追加）
        warped_image = tps.warpImage(image, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        
        return warped_image
    
    return image
