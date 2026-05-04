import cv2
import numpy as np


class ArcFaceEmbedder:
    def __init__(self, ctx_id=-1):
        self.app = None
        self.dim = 512
        try:
            from insightface.app import FaceAnalysis

            self.app = FaceAnalysis(name="buffalo_l")
            self.app.prepare(ctx_id=ctx_id)
        except Exception as exc:
            print(f"[Warning] InsightFace unavailable, using lightweight fallback embeddings: {exc}")

    def embed(self, face):
        if face is None or face.size == 0:
            return None

        if self.app is not None:
            faces = self.app.get(face)
            if faces:
                return faces[0].embedding.astype("float32")
            return None

        resized = cv2.resize(face, (64, 64))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        embedding = hist.flatten().astype("float32")
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm else embedding
