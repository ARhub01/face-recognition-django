import os
import numpy as np
from src.embeddings.arcface_embedder import ArcFaceEmbedder
from src.detection.mtcnn_detector import MTCNNDetector
from src.alignment.face_alignment import align_face
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "uploads")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed_embeddings.npz")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

detector = MTCNNDetector()
embedder = ArcFaceEmbedder()

embeddings = []
labels = []

for img_file in os.listdir(DATA_DIR):
    img_path = os.path.join(DATA_DIR, img_file)
    if not os.path.isfile(img_path):
        continue

    label = os.path.splitext(img_file)[0]

    try:
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: cannot read {img_path}")
            continue

        faces = detector.detect(image)
        if len(faces) == 0:
            print(f"No face found in {img_path}")
            continue

        face = faces[0]
        face_img = align_face(image, face["keypoints"])

        emb = embedder.embed(face_img)
        if emb is not None:
            embeddings.append(emb)
            labels.append(label)

    except Exception as e:
        print(f"Error processing {img_path}: {e}")

embeddings = np.array(embeddings)
labels = np.array(labels)

np.savez(OUTPUT_FILE, embeddings=embeddings, labels=labels)
print(f"Saved embeddings: {OUTPUT_FILE}, shape={embeddings.shape}")
