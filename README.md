# Face Recognition Django System v1.1

A Django web app for face recognition, emotion detection, image upload history, webcam streaming, and JSON API recognition.

## What is improved in v1.1

- Cleaner Django settings, URL routing, templates, and static styling.
- Uploads now keep the original image and save a separate processed result image.
- Recognition results are saved as structured JSON, including face count, labels, confidence, emotion, and bounding boxes.
- The recognizer is cached so heavy AI models are not rebuilt for every request.
- MTCNN, InsightFace, FAISS, and Hugging Face emotion detection now fail gracefully with local fallbacks instead of crashing the app.
- Dataset loading and embedding generation now support one folder per person.
- Added upload tests for result saving and no-face handling.

## Setup

Python 3.11 is recommended for the full ML stack. Newer Python versions can still run the Django app, but some optional AI packages may be skipped and the safe fallback recognizers will be used.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Open the app at:

- Dashboard: http://127.0.0.1:8000/
- Upload: http://127.0.0.1:8000/upload/
- Webcam: http://127.0.0.1:8000/webcam/
- API: http://127.0.0.1:8000/api/recognize/

## Training Data

Put known-person images in this format:

```text
data/lfw/
  person1/
    image1.jpg
    image2.jpg
  person2/
    image1.jpg
```

Then rebuild embeddings:

```powershell
python -m src.generate_embeddings_v2 --data-dir data/lfw --output data/processed/processed_embeddings.npz
```

## API Example

```powershell
curl -X POST -F "image=@sample.jpg" http://127.0.0.1:8000/api/recognize/
```

The response includes `face_count` and a `results` list with bounding boxes, predicted labels, confidence scores, emotions, and emotion confidence.
