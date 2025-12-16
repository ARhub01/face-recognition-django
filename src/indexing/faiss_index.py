import faiss
import numpy as np
import os
import pickle

class FaceIndex:
    def __init__(self, dim=512, index_path="models/faiss.index", labels_path="models/faiss_labels.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.labels_path = labels_path

        if os.path.exists(index_path) and os.path.exists(labels_path):
            self.index = faiss.read_index(index_path)
            with open(labels_path, "rb") as f:
                self.labels = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(dim)
            self.labels = []

    def add(self, embeddings, labels):
        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.labels.extend(labels)
        self.save()

    def search(self, embedding):
        embedding = embedding.astype("float32")
        faiss.normalize_L2(embedding)
        D, I = self.index.search(embedding, 1)

        if I[0][0] == -1:
            return "Unknown", 0.0

        return self.labels[I[0][0]], float(D[0][0])

    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.labels_path, "wb") as f:
            pickle.dump(self.labels, f)
