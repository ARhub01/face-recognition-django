import os
from src.utils import load_config, setup_logger, PROJECT_ROOT
from src.model import create_embedding_model
from src.dataset import FaceDataset
import tensorflow as tf

# Load config
config = load_config()

# Setup logger (logs go to project root)
logger = setup_logger("TRAIN", log_dir=os.path.join(PROJECT_ROOT, "logs"))


def train_model():
    dataset = FaceDataset(data_dir=os.path.join(PROJECT_ROOT, "data/processed"))
    X, y, label_encoder = dataset.load_data()

    if len(X) == 0:
        logger.error("No images found in dataset. Check your processed images!")
        return

    num_classes = y.shape[1]
    input_shape = (
        config["face_detection"]["image_size"],
        config["face_detection"]["image_size"],
        3,
    )

    embedding_model = create_embedding_model(
        input_shape=input_shape,
        embedding_dim=config["model"]["embedding_dim"],
        backbone_name=config["model"]["backbone"],
        trainable=config["model"]["trainable"],
    )

    inputs = tf.keras.Input(shape=input_shape)
    embeddings = embedding_model(inputs)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(embeddings)
    full_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    full_model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config["training"]["learning_rate"]
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    full_model.summary()

    checkpoints_dir = os.path.join(PROJECT_ROOT, "models/checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoints_dir, "face_model_{epoch:02d}.h5")

    logger.info("Starting training...")
    history = full_model.fit(
        X,
        y,
        batch_size=config["training"]["batch_size"],
        epochs=config["training"]["epochs"],
        validation_split=config["training"]["validation_split"],
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path, save_weights_only=False, save_best_only=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, verbose=1
            ),
        ],
    )

    final_model_path = os.path.join(PROJECT_ROOT, "models/embedding_model.h5")
    full_model.save(final_model_path)
    logger.info(f"Training completed. Model saved at {final_model_path}")


if __name__ == "__main__":
    train_model()
