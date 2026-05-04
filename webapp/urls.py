from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

BASE_DIR = settings.BASE_DIR
urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("face_recognition.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
