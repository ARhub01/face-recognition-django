from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import StreamingHttpResponse
import cv2
import base64
import os

from .forms import RegisterForm, ImageUploadForm
from .models import UploadedImage
from src.inference_v2.recognizer import FaceRecognizerV2

@login_required
def dashboard(request):
    images = UploadedImage.objects.filter(user=request.user).order_by("-created_at")
    return render(request, "dashboard.html", {"images": images})

def register(request):
    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("login")
    else:
        form = RegisterForm()
    return render(request, "registration/register.html", {"form": form})

@login_required
def upload_image(request):
    processed_image_url = None

    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            obj = form.save(commit=False)
            obj.user = request.user
            obj.save()

            image = cv2.imread(obj.image.path)
            recognizer = FaceRecognizerV2()  # Detect, embed, recognize + emotion
            results = recognizer.recognize(image)

            for res in results:
                x, y, w, h = res["box"]
                label = res["label"]
                score = res["confidence"]
                emotion = res.get("emotion", "Unknown")
                emotion_conf = res.get("emotion_conf", 0.0)

                # Draw on image
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, f"{label} ({score:.2f})", (x, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(image, f"{emotion} ({emotion_conf:.2f})", (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Save info in DB
                obj.predicted_label = label
                obj.confidence = score
                obj.emotion = emotion
                obj.emotion_conf = emotion_conf
                break  # Only first face

            # Save processed image in media/uploads/
            processed_name = f"processed_{os.path.basename(obj.image.name)}"
            processed_path = os.path.join(os.path.dirname(obj.image.path), processed_name)
            cv2.imwrite(processed_path, image)
            obj.image.name = os.path.join("uploads", processed_name)
            obj.save()

            _, buffer = cv2.imencode(".jpg", image)
            processed_image_url = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

    else:
        form = ImageUploadForm()

    return render(request, "upload.html", {"form": form, "processed_image_url": processed_image_url})

@login_required
def video_feed(request):
    camera = cv2.VideoCapture(0)
    recognizer = FaceRecognizerV2()

    def gen():
        try:
            while True:
                success, frame = camera.read()
                if not success:
                    break

                results = recognizer.recognize(frame)

                for res in results:
                    x, y, w, h = res["box"]
                    label = res["label"]
                    score = res["confidence"]
                    emotion = res.get("emotion", "Unknown")
                    emotion_conf = res.get("emotion_conf", 0.0)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ({score:.2f})", (x, y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"{emotion} ({emotion_conf:.2f})", (x, y + h + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                _, buffer = cv2.imencode(".jpg", frame)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                       buffer.tobytes() + b"\r\n")
        finally:
            camera.release()

    return StreamingHttpResponse(gen(), content_type="multipart/x-mixed-replace; boundary=frame")

@login_required
def webcam_view(request):
    return render(request, "webcam.html")
