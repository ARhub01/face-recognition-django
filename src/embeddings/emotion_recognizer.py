import cv2
import numpy as np
from PIL import Image


EXPRESSION_LABELS = {
    "angry": "Angry",
    "anger": "Angry",
    "disgust": "Disgust",
    "disgusted": "Disgust",
    "fear": "Fear",
    "fearful": "Fear",
    "happy": "Happy",
    "happiness": "Happy",
    "joy": "Happy",
    "neutral": "Neutral",
    "calm": "Neutral",
    "sad": "Sad",
    "sadness": "Sad",
    "surprise": "Surprise",
    "surprised": "Surprise",
    "uncertain": "Neutral",
}


class EmotionRecognizer:
    def __init__(self, model_name="arpanghoshal/Emo0.1", enabled=True):
        self.emotion_model = None
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        if not enabled:
            return
        try:
            from transformers import pipeline

            self.emotion_model = pipeline("image-classification", model=model_name)
            print("[INFO] Emo0.1 Hugging Face model loaded")
        except Exception as e:
            print("[Warning] Emotion model unavailable; using OpenCV expression fallback:", e)

    def _opencv_expression(self, face_img):
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(cv2.resize(gray, (160, 160)))
        height, width = gray.shape
        upper = gray[: height // 2, :]
        lower = gray[height // 2 :, :]

        smiles = self.smile_cascade.detectMultiScale(
            lower,
            scaleFactor=1.6,
            minNeighbors=18,
            minSize=(28, 12),
        )
        if len(smiles) > 0:
            largest = max(smiles, key=lambda box: box[2] * box[3])
            smile_area = (largest[2] * largest[3]) / float(width * height)
            return "Happy", min(0.95, 0.68 + smile_area * 5.0)

        eyes = self.eye_cascade.detectMultiScale(
            upper,
            scaleFactor=1.12,
            minNeighbors=5,
            minSize=(18, 14),
        )

        lower_blur = cv2.GaussianBlur(lower, (5, 5), 0)
        dark_pixels = cv2.threshold(
            lower_blur,
            max(35, int(np.mean(lower_blur) * 0.72)),
            255,
            cv2.THRESH_BINARY_INV,
        )[1]
        contours, _ = cv2.findContours(dark_pixels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mouth_score = 0.0
        if contours:
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            aspect = h / float(w or 1)
            area = cv2.contourArea(contour) / float(width * height)
            mouth_score = area * (1.0 + aspect)
            if mouth_score > 0.035 and aspect > 0.28:
                return "Surprise", min(0.9, 0.58 + mouth_score * 4.0)

        brow_edges = cv2.Canny(upper, 80, 160)
        brow_intensity = float(np.mean(brow_edges > 0))
        contrast = float(np.std(gray))

        if brow_intensity > 0.18 and contrast > 48:
            return "Angry", min(0.86, 0.54 + brow_intensity)

        lower_mean = float(np.mean(lower))
        upper_mean = float(np.mean(upper))
        if lower_mean < upper_mean * 0.9 or mouth_score > 0.022:
            return "Sad", min(0.82, 0.52 + abs(upper_mean - lower_mean) / 255.0)

        if len(eyes) >= 2:
            return "Neutral", 0.56

        return "Neutral", 0.45

    @staticmethod
    def _normalize_label(label):
        if not label:
            return "Unknown"
        clean = str(label).strip()
        return EXPRESSION_LABELS.get(clean.lower(), clean.title())

    def predict(self, face_img):
        if face_img is None or face_img.size == 0:
            return "Unknown", 0.0

        if self.emotion_model is None:
            return self._opencv_expression(face_img)

        try:
            rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            results = self.emotion_model(pil_image)
            if results:
                top = results[0]
                return self._normalize_label(top["label"]), float(top["score"])
            return self._opencv_expression(face_img)
        except Exception as e:
            print("[Emotion prediction error]:", e)
            return self._opencv_expression(face_img)
