
import cv2
import numpy as np
import torch
import facer

def create_faces(rgb_float32, device='cpu'):

    height, width = rgb_float32.shape[:2]
    small_image = cv2.resize(rgb_float32, (width // 4, height // 4), interpolation=cv2.INTER_AREA)

    images = facer.hwc2bchw(torch.from_numpy((small_image * 255).astype(np.uint8))).to(device=device)  # image: 1 x 3 x h x w

    face_detector = facer.face_detector('retinaface/mobilenet', threshold=0.9, device=device)
    with torch.inference_mode():
        faces = face_detector(images)
    if faces['rects'].shape[0] == 0:
        return 0

    face_parser = facer.face_parser('farl/lapa/448', device=device)
    with torch.inference_mode():
        faces = face_parser(images, faces)

    #faces['seg']['probs'] = torch.softmax(faces['seg']['logits'], dim=1)
    faces['seg']['predictions'] = faces['seg']['logits'].argmax(dim=1)  # 形状 [N, H, W]

    faces['size'] = (width, height)

    return faces

def draw_face_mask(faces, exclude_names=[]):

    seg_results = faces['seg']
    label_names = seg_results['label_names']
    #probs = faces['seg']['probs']
    predictions = faces['seg']['predictions']

    face_classes = ['face', 'nose', 'rb', 'lb', 're', 'le', 'ulip', 'llip', 'imouth']
    face_classes = [i for i in face_classes if i not in exclude_names]
    if len(face_classes) == 0:
        return np.zeros((faces['size'][1], faces['size'][0]), dtype=np.float32)
    
    face_ids = [label_names.index(name) for name in face_classes]
    face_masks = []
    for mask in predictions:
        face = torch.isin(mask, torch.tensor(face_ids))
        face = face.float()  # 0/1マスクに変換
        face_masks.append(face)
 
    mask = np.maximum.reduce([face_masks[i].cpu().numpy() for i in range(len(face_masks))])
    #cv2.imwrite(f"face_mask.jpg", (mask * 255).astype(np.uint8))

    #return (mask, faces['size'])
    return cv2.resize(mask, faces['size'], interpolation=cv2.INTER_NEAREST)

def make_bboxs(faces):
    bboxs = []
    for i in range(len(faces['rects'])):
        bbox = faces['rects'][i]
        bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        bboxs.append(bbox)
    return bboxs

def delete_face(faces, index):
    faces['rects'] = np.delete(faces['rects'], index, axis=0)
    faces['points'] = np.delete(faces['points'], index, axis=0)
    faces['seg']['logits'] = np.delete(faces['seg']['logits'], index, axis=0)
    faces['seg']['predictions'] = np.delete(faces['seg']['predictions'], index, axis=0)
