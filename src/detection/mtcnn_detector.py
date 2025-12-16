from mtcnn import MTCNN
import cv2

class MTCNNDetector:
    def __init__(self):
        self.detector = MTCNN()

    def detect(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(rgb)

        faces = []
        for d in detections:
            faces.append({
                "box": d["box"],
                "keypoints": d["keypoints"]
            })
        return faces
