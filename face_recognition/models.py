from django.db import models
from django.contrib.auth.models import User


class UploadedImage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to="uploads/")
    processed_image = models.ImageField(upload_to="processed/", blank=True, null=True)
    predicted_label = models.CharField(max_length=100, blank=True)
    confidence = models.FloatField(default=0.0)
    emotion = models.CharField(max_length=50, blank=True)
    emotion_conf = models.FloatField(default=0.0)
    face_count = models.PositiveIntegerField(default=0)
    results_json = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        label = self.predicted_label or "No face"
        emotion = self.emotion or "Unknown emotion"
        return f"{self.user.username} - {label} - {emotion}"
