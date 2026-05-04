import cv2
import numpy as np


def _bounded_crop(image, box):
    height, width = image.shape[:2]
    x, y, w, h = [int(v) for v in box]
    x = max(0, x)
    y = max(0, y)
    w = max(1, w)
    h = max(1, h)
    return image[y : min(height, y + h), x : min(width, x + w)]


def align_face(image, keypoints=None, box=None, size=160):
    if image is None or image.size == 0:
        return None

    if not keypoints or "left_eye" not in keypoints or "right_eye" not in keypoints:
        face = _bounded_crop(image, box) if box is not None else image
        return cv2.resize(face, (size, size)) if face is not None and face.size else None

    left_eye = keypoints["left_eye"]
    right_eye = keypoints["right_eye"]
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    eyes_center = (
        int((left_eye[0] + right_eye[0]) / 2),
        int((left_eye[1] + right_eye[1]) / 2),
    )

    matrix = cv2.getRotationMatrix2D(eyes_center, angle, 1)
    rotated = cv2.warpAffine(image, matrix, image.shape[1::-1])
    face = _bounded_crop(rotated, box) if box is not None else rotated
    return cv2.resize(face, (size, size)) if face is not None and face.size else None
