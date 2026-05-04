from functools import lru_cache
import os
import numpy as np

from src.detection.mtcnn_detector import MTCNNDetector
from src.alignment.face_alignment import align_face
from src.embeddings.arcface_embedder import ArcFaceEmbedder
from src.indexing.faiss_index import FaceIndex
from src.embeddings.emotion_recognizer import EmotionRecognizer
from src.utils import config_path, load_config, PROJECT_ROOT


class FaceRecognizerV2:
    def __init__(self, embeddings_path=None):
        self.config = load_config()
        image_size = int(self.config.get("face_detection", {}).get("image_size", 224))
        threshold = float(self.config.get("recognition", {}).get("threshold", 0.5))
        embedding_dim = int(self.config.get("model", {}).get("embedding_dim", 512))

        if embeddings_path is None:
            embeddings_path = config_path(
                self.config,
                "paths.embeddings",
                os.path.join(PROJECT_ROOT, "data", "processed", "processed_embeddings.npz"),
            )

        index_path = config_path(self.config, "paths.faiss_index", os.path.join(PROJECT_ROOT, "models", "faiss.index"))
        labels_path = config_path(self.config, "paths.faiss_labels", os.path.join(PROJECT_ROOT, "models", "faiss_labels.pkl"))

        self.image_size = image_size
        self.detector = MTCNNDetector()
        self.embedder = ArcFaceEmbedder()
        self.index = FaceIndex(dim=embedding_dim, index_path=index_path, labels_path=labels_path, threshold=threshold)
        self.emotion_model = EmotionRecognizer(
            model_name=self.config.get("emotion", {}).get("model", "arpanghoshal/Emo0.1"),
            enabled=bool(self.config.get("emotion", {}).get("enabled", True)),
        )

        try:
            if os.path.exists(embeddings_path):
                data = np.load(embeddings_path, allow_pickle=True)
                embeddings = data["embeddings"] if "embeddings" in data else np.empty((0, embedding_dim))
                labels = data["labels"] if "labels" in data else []

                if len(embeddings) > 0 and not self.index.loaded_from_disk:
                    self.index.add(np.array(embeddings), list(labels), persist=False)
                else:
                    print("[Warning] No embeddings found in file:", embeddings_path)
            else:
                print(f"[Warning] Embeddings file not found: {embeddings_path}")
        except Exception as e:
            print(f"[Error loading embeddings: {e}]")

    def recognize(self, image):
        if image is None or image.size == 0:
            return []

        faces = self.detector.detect(image)
        results = []

        for f in faces:
            box = f.get("box", [0, 0, 0, 0])
            try:
                face_img = align_face(
                    image,
                    keypoints=f.get("keypoints") or {},
                    box=box,
                    size=self.image_size,
                )
                if face_img is None or face_img.size == 0:
                    raise ValueError("Empty face after alignment")

                emb = self.embedder.embed(face_img)
                if emb is not None:
                    label, score = self.index.search(emb.reshape(1, -1))
                else:
                    label, score = "Unknown", 0.0

                emotion_label, emotion_conf = self.emotion_model.predict(face_img)

                results.append(
                    {
                        "box": [int(v) for v in box],
                        "label": str(label),
                        "confidence": float(score),
                        "emotion": str(emotion_label),
                        "emotion_conf": float(emotion_conf),
                    }
                )

            except Exception as e:
                print(f"[Recognition Error]: {e}")
                results.append(
                    {
                        "box": [int(v) for v in box],
                        "label": "Unknown",
                        "confidence": 0.0,
                        "emotion": "Unknown",
                        "emotion_conf": 0.0,
                    }
                )

        return results


@lru_cache(maxsize=1)
def get_recognizer():
    return FaceRecognizerV2()
