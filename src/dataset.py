import os
import cv2
import numpy as np
from src.utils import load_config, setup_logger, PROJECT_ROOT
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

config = load_config()
logger = setup_logger("DATASET")


class FaceDataset:
    def __init__(self, data_dir=None, image_size=None):
        if data_dir is None:
            data_dir = os.path.join(PROJECT_ROOT, "data/processed")
        self.data_dir = data_dir

        if image_size is None:
            image_size = config["face_detection"]["image_size"]
        self.image_size = image_size

        self.images = []
        self.labels = []

    def load_data(self):
        logger.info(f"Loading dataset from {self.data_dir}")
        if not os.path.isdir(self.data_dir):
            logger.warning(f"Dataset directory does not exist: {self.data_dir}")
            return np.array([]), np.array([]), LabelEncoder()

        for person_name in sorted(os.listdir(self.data_dir)):
            person_folder = os.path.join(self.data_dir, person_name)
            if not os.path.isdir(person_folder):
                continue

            for img_file in sorted(os.listdir(person_folder)):
                img_path = os.path.join(person_folder, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    logger.warning(f"Cannot read image {img_path}")
                    continue
                img = cv2.resize(img, (self.image_size, self.image_size))
                img = img.astype("float32") / 255.0
                self.images.append(img)
                self.labels.append(person_name)

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

        self.label_encoder = LabelEncoder()
        if len(self.labels) == 0:
            logger.warning("No labeled images found")
            return self.images, np.array([]), self.label_encoder

        self.labels_enc = self.label_encoder.fit_transform(self.labels)
        self.labels_cat = to_categorical(self.labels_enc)

        logger.info(
            f"Loaded {len(self.images)} images with {len(self.label_encoder.classes_)} classes"
        )
        return self.images, self.labels_cat, self.label_encoder
