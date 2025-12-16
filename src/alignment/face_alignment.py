import cv2
import numpy as np

def align_face(image, keypoints, size=160):
    left_eye = keypoints["left_eye"]
    right_eye = keypoints["right_eye"]

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    eyes_center = (
        int((left_eye[0] + right_eye[0]) / 2),
        int((left_eye[1] + right_eye[1]) / 2)
    )

    M = cv2.getRotationMatrix2D(eyes_center, angle, 1)
    rotated = cv2.warpAffine(image, M, image.shape[1::-1])

    x, y = keypoints["nose"]
    face = rotated[y-size//2:y+size//2, x-size//2:x+size//2]
    return cv2.resize(face, (size, size))
