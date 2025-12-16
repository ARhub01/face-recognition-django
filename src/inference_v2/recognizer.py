# src/inference_v2/recognizer.py

from src.detection.mtcnn_detector import MTCNNDetector
from src.alignment.face_alignment import align_face
from src.embeddings.arcface_embedder import ArcFaceEmbedder
from src.indexing.faiss_index import FaceIndex
from src.embeddings.emotion_recognizer import EmotionRecognizer
import numpy as np
import os

class FaceRecognizerV2:
    def __init__(self, embeddings_path="data/processed_embeddings.npz"):
        self.detector = MTCNNDetector()
        self.embedder = ArcFaceEmbedder()
        self.index = FaceIndex()
        self.emotion_model = EmotionRecognizer()

        # Load embeddings for identity recognition
        try:
            if os.path.exists(embeddings_path):
                data = np.load(embeddings_path, allow_pickle=True)
                embeddings = data.get("embeddings", [])
                labels = data.get("labels", [])

                if len(embeddings) > 0:
                    self.index.add(np.array(embeddings), list(labels))
                else:
                    print("[Warning] No embeddings found in file:", embeddings_path)
            else:
                print(f"[Warning] Embeddings file not found: {embeddings_path}")
        except Exception as e:
            print(f"[Error loading embeddings: {e}]")

    def recognize(self, image):
        faces = self.detector.detect(image)
        results = []

        for f in faces:
            try:
                # Align face
                face_img = align_face(image, f.get("keypoints", None))
                if face_img is None or face_img.size == 0:
                    raise ValueError("Empty face after alignment")

                # Identity prediction
                emb = self.embedder.embed(face_img)
                if emb is not None and self.index.index is not None:
                    label, score = self.index.search(emb.reshape(1, -1))
                else:
                    label, score = "Unknown", 0.0

                # Emotion prediction
                emotion_label, emotion_conf = self.emotion_model.predict(face_img)

                results.append({
                    "box": f.get("box", [0,0,0,0]),
                    "label": label,
                    "confidence": score,
                    "emotion": emotion_label,
                    "emotion_conf": emotion_conf
                })

            except Exception as e:
                print(f"[Recognition Error]: {e}")
                results.append({
                    "box": f.get("box", [0,0,0,0]),
                    "label": "Unknown",
                    "confidence": 0.0,
                    "emotion": "Unknown",
                    "emotion_conf": 0.0
                })

        return results
