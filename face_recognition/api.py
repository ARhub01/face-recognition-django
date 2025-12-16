from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser

import cv2
import numpy as np

from src.inference_v2.recognizer import FaceRecognizerV2


class FaceRecognitionAPI(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        image_file = request.FILES.get("image")
        if not image_file:
            return Response({"error": "No image provided"}, status=400)

        img_bytes = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        recognizer = FaceRecognizerV2()
        results = recognizer.recognize(image)

        return Response({"results": results})
