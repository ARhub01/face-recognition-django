import numpy as np
import os
import pickle


try:
    import faiss
except Exception:
    faiss = None


class FaceIndex:
    def __init__(self, dim=512, index_path="models/faiss.index", labels_path="models/faiss_labels.pkl", threshold=0.5):
        self.dim = dim
        self.index_path = index_path
        self.labels_path = labels_path
        self.threshold = threshold
        self.embeddings = np.empty((0, dim), dtype="float32")
        self.index = None
        self.labels = []
        self.loaded_from_disk = False

        if faiss and os.path.exists(index_path) and os.path.exists(labels_path):
            self.index = faiss.read_index(index_path)
            with open(labels_path, "rb") as f:
                self.labels = pickle.load(f)
            self.loaded_from_disk = True
        elif faiss:
            self.index = faiss.IndexFlatIP(dim)
        else:
            print("[Warning] FAISS unavailable; using NumPy cosine search")

    @staticmethod
    def _normalize(embeddings):
        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def add(self, embeddings, labels, persist=False):
        embeddings = self._normalize(embeddings)
        if embeddings.shape[1] != self.dim:
            print(f"[Warning] Skipping embeddings with dimension {embeddings.shape[1]} (expected {self.dim})")
            return

        if self.index is not None and faiss:
            self.index.add(embeddings)
        self.embeddings = np.vstack([self.embeddings, embeddings])
        self.labels.extend(labels)
        if persist:
            self.save()

    def search(self, embedding):
        embedding = self._normalize(embedding)
        if embedding.shape[1] != self.dim or not self.labels:
            return "Unknown", 0.0

        if self.index is not None and faiss:
            distances, indices = self.index.search(embedding, 1)
            score = float(distances[0][0])
            index = int(indices[0][0])
        else:
            scores = self.embeddings @ embedding[0]
            index = int(np.argmax(scores))
            score = float(scores[index])

        if index < 0 or score < self.threshold:
            return "Unknown", 0.0

        return self.labels[index], score

    def save(self):
        if self.index is None or not faiss:
            return
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.labels_path, "wb") as f:
            pickle.dump(self.labels, f)
