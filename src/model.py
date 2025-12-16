import tensorflow as tf
from tensorflow.keras import layers, models
from src.utils import load_config, setup_logger, PROJECT_ROOT
import os

config = load_config()
logger = setup_logger("MODEL")


def create_embedding_model(
    input_shape=(224, 224, 3),
    embedding_dim=128,
    backbone_name="MobileNetV2",
    trainable=False,
):
    """
    Creates a CNN embedding model using Transfer Learning
    """
    logger.info(f"Creating embedding model with backbone: {backbone_name}")

    if backbone_name.lower() == "mobilenetv2":
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
    elif backbone_name.lower() == "resnet50":
        base_model = tf.keras.applications.ResNet50(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
    else:
        raise ValueError("Unsupported backbone")

    base_model.trainable = trainable

    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(embedding_dim, activation=None, name="embedding"),
        ]
    )

    logger.info("Embedding model created successfully")
    return model


# Optional: test creation
if __name__ == "__main__":
    model = create_embedding_model(
        input_shape=(
            config["face_detection"]["image_size"],
            config["face_detection"]["image_size"],
            3,
        ),
        embedding_dim=config["model"]["embedding_dim"],
        backbone_name=config["model"]["backbone"],
        trainable=config["model"]["trainable"],
    )
    model.summary()
