import cv2


class MTCNNDetector:
    def __init__(self, min_face_size=40):
        self.min_face_size = min_face_size
        self.detector = None
        try:
            from mtcnn import MTCNN

            self.detector = MTCNN()
        except Exception as exc:
            print(f"[Warning] MTCNN unavailable, using OpenCV Haar cascade: {exc}")
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, image):
        if image is None or image.size == 0:
            return []

        if self.detector is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            boxes = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size),
            )
            return [{"box": [int(x), int(y), int(w), int(h)], "keypoints": {}} for x, y, w, h in boxes]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(rgb)

        faces = []
        for d in detections:
            x, y, w, h = d["box"]
            if w < self.min_face_size or h < self.min_face_size:
                continue
            faces.append(
                {
                    "box": [max(0, int(x)), max(0, int(y)), int(w), int(h)],
                    "keypoints": d.get("keypoints", {}),
                }
            )
        return faces
