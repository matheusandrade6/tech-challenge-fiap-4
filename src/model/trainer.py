import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training, callbacks, and progress reporting."""

    def __init__(self, checkpoint_dir: str | Path = "models/checkpoints") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)

    def build_callbacks(self) -> list:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.checkpoint_dir / "best_model.keras"

        return [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=20,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                patience=10,
                factor=0.5,
                min_lr=1e-7,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
        ]

    def train(
        self,
        model: tf.keras.Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> tf.keras.callbacks.History:
        logger.info(
            "Training start — epochs=%d, batch_size=%d, train_samples=%d, val_samples=%d",
            epochs,
            batch_size,
            len(X_train),
            len(X_val),
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.build_callbacks(),
            verbose=1,
        )

        epochs_run = len(history.history["loss"])
        best_val_loss = min(history.history["val_loss"])
        logger.info(
            "Training complete — epochs_run=%d, best_val_loss=%.6f",
            epochs_run,
            best_val_loss,
        )
        return history
