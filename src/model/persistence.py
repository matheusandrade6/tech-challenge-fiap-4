import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.preprocessing.scaler import ClosingPriceScaler

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = Path("models/saved_models/lstm_model.keras")
_DEFAULT_SCALER_PATH = Path("models/scalers/scaler.pkl")
_DEFAULT_METADATA_PATH = Path("models/metadata.json")


@dataclass
class ModelArtifacts:
    model: tf.keras.Model
    scaler: ClosingPriceScaler
    metadata: dict


class ModelPersistence:
    """Saves and loads LSTM model artifacts: model weights, scaler, and metadata."""

    def __init__(
        self,
        model_path: str | Path = _DEFAULT_MODEL_PATH,
        scaler_path: str | Path = _DEFAULT_SCALER_PATH,
        metadata_path: str | Path = _DEFAULT_METADATA_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.metadata_path = Path(metadata_path)

    def save(
        self,
        model: tf.keras.Model,
        scaler: ClosingPriceScaler,
        test_metrics: dict,
        sequence_length: int = 60,
        feature_columns: list[str] | None = None,
        model_version: str = "v1.0",
    ) -> dict[str, Path]:
        """Persist model, scaler, and metadata; return mapping of artifact name -> path."""
        if feature_columns is None:
            feature_columns = ["Close"]

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(self.model_path))
        logger.info("Model saved -> %s  (version=%s)", self.model_path, model_version)

        scaler.save(self.scaler_path)

        training_date = datetime.now().isoformat()
        metadata = {
            "model_version": model_version,
            "training_date": training_date,
            "sequence_length": sequence_length,
            "feature_columns": feature_columns,
            "test_metrics": {
                "mae": test_metrics.get("mae") or test_metrics.get("MAE"),
                "rmse": test_metrics.get("rmse") or test_metrics.get("RMSE"),
                "mape": test_metrics.get("mape") or test_metrics.get("MAPE"),
            },
        }

        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(
            "Metadata saved -> %s  [version=%s, trained_at=%s]",
            self.metadata_path,
            model_version,
            training_date,
        )

        paths: dict[str, Path] = {
            "model": self.model_path,
            "scaler": self.scaler_path,
            "metadata": self.metadata_path,
        }
        print(
            f"\n[ModelPersistence] Artifacts saved:\n"
            f"  model    -> {self.model_path}\n"
            f"  scaler   -> {self.scaler_path}\n"
            f"  metadata -> {self.metadata_path}\n"
        )
        return paths

    def load(self) -> ModelArtifacts:
        """Load model, scaler, and metadata from disk and return as ModelArtifacts."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        model = tf.keras.models.load_model(str(self.model_path))
        logger.info("Model loaded from %s", self.model_path)

        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {self.scaler_path}")
        scaler = ClosingPriceScaler.load(self.scaler_path)

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        logger.info(
            "Metadata loaded — version=%s, trained_at=%s",
            metadata.get("model_version", "unknown"),
            metadata.get("training_date", "unknown"),
        )

        return ModelArtifacts(model=model, scaler=scaler, metadata=metadata)

    def save_versioned(
        self,
        model: tf.keras.Model,
        scaler: ClosingPriceScaler,
        test_metrics: dict,
        sequence_length: int = 60,
        feature_columns: list[str] | None = None,
    ) -> dict[str, Path]:
        """Save artifacts with a timestamp-based version tag alongside the canonical copy."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v_{timestamp}"

        versioned = ModelPersistence(
            model_path=self.model_path.parent / f"lstm_model_{timestamp}.keras",
            scaler_path=self.scaler_path.parent / f"scaler_{timestamp}.pkl",
            metadata_path=self.metadata_path.parent / f"metadata_{timestamp}.json",
        )
        versioned.save(model, scaler, test_metrics, sequence_length, feature_columns, version)

        # Also overwrite the canonical (latest) artifacts
        return self.save(model, scaler, test_metrics, sequence_length, feature_columns, version)


class ModelInference:
    """Single-step next-day price prediction using a loaded LSTM model."""

    def predict_next_price(
        self,
        last_60_days: np.ndarray,
        model: tf.keras.Model,
        scaler: ClosingPriceScaler,
    ) -> float:
        last_60_days = np.asarray(last_60_days, dtype=float)

        if last_60_days.ndim != 1 or len(last_60_days) != 60:
            raise ValueError(
                f"last_60_days must be a 1-D array of exactly 60 prices; "
                f"got shape {last_60_days.shape}"
            )

        normed = scaler.transform(last_60_days)
        reshaped = normed.reshape(1, 60, 1)
        pred_norm = model.predict(reshaped, verbose=0)
        pred_price = scaler.inverse_transform(pred_norm.flatten())

        result = float(pred_price[0])
        logger.info(
            "predict_next_price — input_shape=%s  predicted_price=%.4f",
            reshaped.shape,
            result,
        )
        return result


def get_model_version(metadata_path: str | Path = _DEFAULT_METADATA_PATH) -> dict:
    """Return version, training date, and test metrics from saved metadata JSON."""
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return {
        "model_version": metadata.get("model_version", "unknown"),
        "training_date": metadata.get("training_date", "unknown"),
        "test_metrics": metadata.get("test_metrics", {}),
    }
