# src/embeddings/emotion_recognizer.py

from transformers import pipeline
import cv2
import numpy as np

class EmotionRecognizer:
    def __init__(self):
        try:
            # Load Emo0.1 Hugging Face pipeline
            self.emotion_model = pipeline("image-classification", model="arpanghoshal/Emo0.1")
            print("[INFO] Emo0.1 Hugging Face model loaded")
        except Exception as e:
            print("[Warning] Failed to load Emo0.1 pipeline:", e)
            self.emotion_model = None

    def predict(self, face_img):
        if self.emotion_model is None:
            return "Unknown", 0.0

        if face_img is None or face_img.size == 0:
            return "Unknown", 0.0

        try:
            # Convert BGR to RGB for transformers
            rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            results = self.emotion_model(rgb)
            if results:
                top = results[0]
                return top["label"], float(top["score"])
            return "Unknown", 0.0
        except Exception as e:
            print("[Emotion prediction error]:", e)
            return "Unknown", 0.0
