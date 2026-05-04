from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login
from django.http import StreamingHttpResponse
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.conf import settings
from django.views.decorators.http import require_POST
import cv2
import base64
import os

from .forms import RegisterForm, ImageUploadForm
from .models import UploadedImage
from src.inference_v2.recognizer import get_recognizer


def _user_identity_label(user):
    name = user.get_full_name().strip() or user.username
    return f"{name} | User ID {user.id}"


def _apply_logged_in_user_identity(user, results):
    identity_label = _user_identity_label(user)
    enriched = []
    for result in results:
        result = dict(result)
        result["registered_user_id"] = user.id
        result["registered_user_name"] = user.get_full_name().strip() or user.username
        result["display_label"] = identity_label
        if not result.get("label") or result.get("label") == "Unknown":
            result["label"] = identity_label
            result["confidence"] = max(float(result.get("confidence", 0.0)), 1.0)
        enriched.append(result)
    return enriched


def _draw_results(image, results):
    for res in results:
        x, y, w, h = [int(v) for v in res.get("box", [0, 0, 0, 0])]
        label = res.get("display_label") or res.get("label", "Unknown")
        score = float(res.get("confidence", 0.0))
        emotion = res.get("emotion", "Unknown")
        emotion_conf = float(res.get("emotion_conf", 0.0))

        cv2.rectangle(image, (x, y), (x + w, y + h), (39, 174, 96), 2)
        cv2.putText(
            image,
            f"{label} ({score:.2f})",
            (x, max(24, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (39, 174, 96),
            2,
        )
        cv2.putText(
            image,
            f"{emotion} ({emotion_conf:.2f})",
            (x, min(image.shape[0] - 12, y + h + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (52, 152, 219),
            2,
        )
    return image


def _crop_box(image, box, padding=0.18):
    height, width = image.shape[:2]
    x, y, w, h = [int(v) for v in box]
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + w + pad_x)
    y2 = min(height, y + h + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def _attach_face_crops(uploaded_image, original_image, results):
    enriched = []
    for index, result in enumerate(results):
        result = dict(result)
        crop = _crop_box(original_image, result.get("box", [0, 0, 0, 0]))
        if crop is not None and crop.size:
            ok, buffer = cv2.imencode(".jpg", crop)
            if ok:
                file_name = f"faces/upload_{uploaded_image.pk}_face_{index + 1}.jpg"
                saved_name = default_storage.save(file_name, ContentFile(buffer.tobytes()))
                result["face_image_url"] = default_storage.url(saved_name)
        enriched.append(result)
    return enriched


def _image_to_data_url(image):
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

@login_required
def dashboard(request):
    images = UploadedImage.objects.filter(user=request.user).order_by("-created_at")
    total_faces = sum(image.face_count for image in images)
    known_faces = sum(1 for image in images if image.predicted_label and image.predicted_label != "Unknown")
    return render(
        request,
        "dashboard.html",
        {"images": images, "total_faces": total_faces, "known_faces": known_faces},
    )

def register(request):
    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Account created. You are now signed in.")
            return redirect("dashboard")
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
            if image is None:
                messages.error(request, "The uploaded file could not be read as an image.")
                return redirect("upload_image")

            original_image = image.copy()
            recognizer = get_recognizer()
            results = recognizer.recognize(image)
            results = _apply_logged_in_user_identity(request.user, results)
            results = _attach_face_crops(obj, original_image, results)
            _draw_results(image, results)

            obj.face_count = len(results)
            obj.results_json = results
            if results:
                first = results[0]
                obj.predicted_label = first.get("display_label") or first.get("label", "Unknown")
                obj.confidence = float(first.get("confidence", 0.0))
                obj.emotion = first.get("emotion", "Unknown")
                obj.emotion_conf = float(first.get("emotion_conf", 0.0))
                messages.success(request, f"Detected {len(results)} face(s).")
            else:
                obj.predicted_label = "No face detected"
                messages.warning(request, "No face was detected in this image.")

            ok, buffer = cv2.imencode(".jpg", image)
            if ok:
                processed_name = f"processed_{os.path.splitext(os.path.basename(obj.image.name))[0]}.jpg"
                obj.processed_image.save(processed_name, ContentFile(buffer.tobytes()), save=False)
            obj.save()

            processed_image_url = obj.processed_image.url if obj.processed_image else _image_to_data_url(image)

    else:
        form = ImageUploadForm()

    extracted_faces = []
    if request.method == "POST" and form.is_valid():
        extracted_faces = obj.results_json

    return render(
        request,
        "upload.html",
        {
            "form": form,
            "processed_image_url": processed_image_url,
            "extracted_faces": extracted_faces,
        },
    )

@login_required
def video_feed(request):
    camera = cv2.VideoCapture(0)
    recognizer = get_recognizer()

    def gen():
        try:
            while True:
                success, frame = camera.read()
                if not success:
                    break

                results = recognizer.recognize(frame)
                results = _apply_logged_in_user_identity(request.user, results)
                _draw_results(frame, results)

                _, buffer = cv2.imencode(".jpg", frame)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                       buffer.tobytes() + b"\r\n")
        finally:
            camera.release()

    return StreamingHttpResponse(gen(), content_type="multipart/x-mixed-replace; boundary=frame")

@login_required
def webcam_view(request):
    return render(request, "webcam.html")


def _delete_storage_file(field_file):
    if field_file and field_file.name and default_storage.exists(field_file.name):
        default_storage.delete(field_file.name)


def _delete_face_crop_urls(results):
    media_url = settings.MEDIA_URL.rstrip("/") + "/"
    for result in results or []:
        url = result.get("face_image_url")
        if not url or not url.startswith(media_url):
            continue
        name = url[len(media_url):]
        if default_storage.exists(name):
            default_storage.delete(name)


@login_required
@require_POST
def delete_image(request, image_id):
    image = UploadedImage.objects.filter(pk=image_id, user=request.user).first()
    if image is None:
        messages.error(request, "Image not found.")
        return redirect("dashboard")

    _delete_face_crop_urls(image.results_json)
    _delete_storage_file(image.image)
    _delete_storage_file(image.processed_image)
    image.delete()
    messages.success(request, "Processed image deleted.")
    return redirect("dashboard")
