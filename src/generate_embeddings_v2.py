import os
import numpy as np
import argparse
from src.embeddings.arcface_embedder import ArcFaceEmbedder
from src.detection.mtcnn_detector import MTCNNDetector
from src.alignment.face_alignment import align_face
from src.utils import config_path, load_config, PROJECT_ROOT
import cv2


def build_embeddings(data_dir=None, output_file=None):
    config = load_config()
    data_dir = data_dir or config_path(config, "paths.raw_data", os.path.join(PROJECT_ROOT, "data", "lfw"))
    output_file = output_file or config_path(
        config,
        "paths.embeddings",
        os.path.join(PROJECT_ROOT, "data", "processed", "processed_embeddings.npz"),
    )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    detector = MTCNNDetector()
    embedder = ArcFaceEmbedder()

    embeddings = []
    labels = []

    for root, _, files in os.walk(data_dir):
        label = os.path.basename(root)
        if label == os.path.basename(data_dir):
            label = None

        for img_file in files:
            img_path = os.path.join(root, img_file)
            current_label = label or os.path.splitext(img_file)[0]

            try:
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: cannot read {img_path}")
                    continue

                faces = detector.detect(image)
                if len(faces) == 0:
                    print(f"No face found in {img_path}")
                    continue

                face = faces[0]
                face_img = align_face(image, face.get("keypoints", {}), face.get("box"))

                emb = embedder.embed(face_img)
                if emb is not None:
                    embeddings.append(emb)
                    labels.append(current_label)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    np.savez(output_file, embeddings=embeddings, labels=labels)
    print(f"Saved embeddings: {output_file}, shape={embeddings.shape}")
    return output_file, embeddings.shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate face embeddings for recognition")
    parser.add_argument("--data-dir", default=None, help="Folder with one subfolder per person")
    parser.add_argument("--output", default=None, help="Output .npz path")
    args = parser.parse_args()
    build_embeddings(args.data_dir, args.output)
