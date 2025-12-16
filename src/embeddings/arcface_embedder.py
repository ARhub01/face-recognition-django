from insightface.app import FaceAnalysis

class ArcFaceEmbedder:
    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0)

    def embed(self, face):
        faces = self.app.get(face)
        return faces[0].embedding if faces else None
