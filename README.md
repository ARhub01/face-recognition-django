# Face Recognition & Emotion Detection Web App

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![Django](https://img.shields.io/badge/django-5.2.9-green)](https://www.djangoproject.com/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.15-orange)](https://www.tensorflow.org/)

A **Django-based web application** for **face recognition** and **emotion detection** using deep learning and computer vision.  
Users can upload images or use a webcam to detect faces, identify known users, and recognize emotions.

---
> ⚠️ Project Status: Ongoing academic project  
> This repository represents an active research-oriented system development in computer vision and deep learning.

## Note on Models
Pretrained models (ArcFace, Emo0.1) are used to study system integration, inference pipelines, and deployment challenges rather than to claim novel model contributions.

## 🔹 Features
- User registration & login system
- Upload images for face recognition
- Detect multiple faces per image
- Predict identity using **ArcFace embeddings**
- Emotion recognition using **Emo0.1 (Hugging Face)**
- Real-time webcam face recognition and emotion detection
- Face indexing & search using **FAISS**
- Supports multiple face alignment and embedding storage

---

## 🛠 Installation

1. **Clone the repository**
```bash
git clone https://github.com/username/face-recognition-django.git
cd face-recognition-django
Create a virtual environment

bash
Copy code
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
Install dependencies

bash
Copy code
pip install -r requirements.txt
Run database migrations

bash
Copy code
python manage.py makemigrations
python manage.py migrate
Run the Django server

bash
Copy code
python manage.py runserver
Access the web app

Image upload: http://127.0.0.1:8000/upload/

Real-time webcam: http://127.0.0.1:8000/webcam/

📁 Project Structure
bash
Copy code
face_recognition-django/
├─ face_recognition/          # Django app (views, models, forms, templates)
├─ src/                       # ML/AI pipeline
│  ├─ detection/              # Face detectors (MTCNN, InsightFace)
│  ├─ alignment/              # Face alignment utilities
│  ├─ embeddings/             # ArcFace embeddings + Emo0.1
│  ├─ indexing/               # FAISS face index
├─ media/                      # Uploaded & processed images
├─ manage.py
├─ requirements.txt
└─ README.md
⚙ Usage
Upload Image: Detect faces, identify users, and predict emotions.

Webcam Feed: Real-time recognition with bounding boxes, labels, and emotion display.

Sample Output:

Detected Faces	Predicted Identity	Emotion
✅	Ahmad Raza	Happy
✅	Unknown	Neutral

🚀 Future Enhancements
Train on custom face datasets

Improve emotion recognition accuracy

Add multi-user embedding management

Deploy on cloud with HTTPS support