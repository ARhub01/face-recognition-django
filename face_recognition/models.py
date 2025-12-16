from django.db import models
from django.contrib.auth.models import User

class UploadedImage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to="uploads/")  # stored on disk
    predicted_label = models.CharField(max_length=100, blank=True)
    confidence = models.FloatField(default=0.0)
    emotion = models.CharField(max_length=50, blank=True)
    emotion_conf = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.predicted_label} - {self.emotion}"
