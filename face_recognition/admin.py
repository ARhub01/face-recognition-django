from django.contrib import admin
from .models import UploadedImage

@admin.register(UploadedImage)
class UploadedImageAdmin(admin.ModelAdmin):
    list_display = (
        "user",
        "predicted_label",
        "confidence",
        "emotion",
        "emotion_conf",
        "face_count",
        "created_at",
    )
    list_filter = ("created_at", "emotion")
    search_fields = ("user__username", "predicted_label", "emotion")
    readonly_fields = ("created_at",)
