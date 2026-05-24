import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing.scaler import ClosingPriceScaler

logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    mae: float
    rmse: float
    mape: float

    def as_dict(self) -> dict:
        return {"MAE": self.mae, "RMSE": self.rmse, "MAPE": self.mape}


class ModelEvaluator:
    """Computes evaluation metrics and generates diagnostic plots."""

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        scaler: ClosingPriceScaler,
    ) -> Metrics:
        y_true_inv = scaler.inverse_transform(y_true)
        y_pred_inv = scaler.inverse_transform(y_pred.flatten())

        mae = float(np.mean(np.abs(y_true_inv - y_pred_inv)))
        rmse = float(np.sqrt(np.mean((y_true_inv - y_pred_inv) ** 2)))

        # Exclude zeros to avoid division-by-zero; expected only at market open gaps
        nonzero = y_true_inv != 0
        mape = float(
            np.mean(np.abs((y_true_inv[nonzero] - y_pred_inv[nonzero]) / y_true_inv[nonzero])) * 100
        )

        return Metrics(mae=mae, rmse=rmse, mape=mape)

    def plot_training_history(
        self,
        history: "tf.keras.callbacks.History",  # noqa: F821
        output_dir: str | Path = "reports",
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history.history["loss"], label="Train Loss")
        ax.plot(history.history["val_loss"], label="Val Loss")
        ax.set_title("Training History — MSE Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.legend()
        ax.grid(alpha=0.3)

        path = output_dir / "training_history.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved training history plot -> %s", path)
        return path

    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        scaler: ClosingPriceScaler,
        split_name: str = "Test",
        output_dir: str | Path = "reports",
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        y_true_inv = scaler.inverse_transform(y_true)
        y_pred_inv = scaler.inverse_transform(y_pred.flatten())

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(y_true_inv, label="Actual Price", color="steelblue")
        ax.plot(y_pred_inv, label="Predicted Price", color="darkorange", alpha=0.85)
        ax.set_title(f"LSTM — Actual vs Predicted ({split_name} Set)")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Price (BRL)")
        ax.legend()
        ax.grid(alpha=0.3)

        path = output_dir / f"predictions_{split_name.lower()}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved predictions plot -> %s", path)
        return path
