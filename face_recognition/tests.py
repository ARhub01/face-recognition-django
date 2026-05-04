import shutil
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from .models import UploadedImage


TEST_ROOT = Path(__file__).resolve().parents[1] / ".test-media"
TEMP_MEDIA_ROOT = TEST_ROOT / "media"
TEMP_MEDIA_ROOT.mkdir(parents=True, exist_ok=True)


@override_settings(MEDIA_ROOT=TEMP_MEDIA_ROOT)
class UploadImageTests(TestCase):
    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(TEST_ROOT, ignore_errors=True)

    def setUp(self):
        self.user = User.objects.create_user(username="tester", password="secret-pass")
        self.client.login(username="tester", password="secret-pass")

    @staticmethod
    def _jpg_upload():
        image = np.full((120, 120, 3), 255, dtype=np.uint8)
        ok, buffer = cv2.imencode(".jpg", image)
        assert ok
        return SimpleUploadedFile("face.jpg", buffer.tobytes(), content_type="image/jpeg")

    @patch("face_recognition.views.get_recognizer")
    def test_upload_saves_processed_image_and_results(self, get_recognizer):
        recognizer = Mock()
        recognizer.recognize.return_value = [
            {
                "box": [10, 10, 40, 40],
                "label": "person1",
                "confidence": 0.91,
                "emotion": "happy",
                "emotion_conf": 0.82,
            }
        ]
        get_recognizer.return_value = recognizer

        response = self.client.post(reverse("upload_image"), {"image": self._jpg_upload()})

        self.assertEqual(response.status_code, 200)
        uploaded = UploadedImage.objects.get()
        self.assertEqual(uploaded.predicted_label, "tester | User ID 1")
        self.assertEqual(uploaded.face_count, 1)
        self.assertTrue(uploaded.image.name.startswith("uploads/"))
        self.assertTrue(uploaded.processed_image.name.startswith("processed/"))
        self.assertEqual(uploaded.results_json[0]["emotion"], "happy")
        self.assertEqual(uploaded.results_json[0]["display_label"], "tester | User ID 1")
        self.assertEqual(uploaded.results_json[0]["registered_user_id"], self.user.id)

    @patch("face_recognition.views.get_recognizer")
    def test_upload_handles_no_faces(self, get_recognizer):
        recognizer = Mock()
        recognizer.recognize.return_value = []
        get_recognizer.return_value = recognizer

        self.client.post(reverse("upload_image"), {"image": self._jpg_upload()})

        uploaded = UploadedImage.objects.get()
        self.assertEqual(uploaded.predicted_label, "No face detected")
        self.assertEqual(uploaded.face_count, 0)

    @patch("face_recognition.views.get_recognizer")
    def test_user_can_delete_own_processed_image(self, get_recognizer):
        recognizer = Mock()
        recognizer.recognize.return_value = [
            {
                "box": [10, 10, 40, 40],
                "label": "Unknown",
                "confidence": 0.0,
                "emotion": "Neutral",
                "emotion_conf": 0.55,
            }
        ]
        get_recognizer.return_value = recognizer

        self.client.post(reverse("upload_image"), {"image": self._jpg_upload()})
        uploaded = UploadedImage.objects.get()

        response = self.client.post(reverse("delete_image", args=[uploaded.id]))

        self.assertEqual(response.status_code, 302)
        self.assertFalse(UploadedImage.objects.exists())
