from django.urls import path
from . import views
from django.contrib.auth import views as auth_views
from .api import FaceRecognitionAPI

urlpatterns = [
    path("register/", views.register, name="register"),
    path("login/", auth_views.LoginView.as_view(template_name="registration/login.html"), name="login"),
    path("logout/", auth_views.LogoutView.as_view(next_page="login"), name="logout"),
    path("", views.dashboard, name="dashboard"),
    path("upload/", views.upload_image, name="upload_image"),
    path("webcam/", views.webcam_view, name="webcam_view"),
    path("video_feed/", views.video_feed, name="video_feed"),
    path("api/recognize/", FaceRecognitionAPI.as_view(), name="api_recognize"),
]
