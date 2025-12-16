from django.contrib import admin
from .models import UploadedImage

@admin.register(UploadedImage)
class UploadedImageAdmin(admin.ModelAdmin):
    list_display = ("user", "predicted_label", "confidence", "emotion", "emotion_conf", "created_at")
