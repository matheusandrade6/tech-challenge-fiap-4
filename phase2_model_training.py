import logging
import os
import random
import sys
from pathlib import Path

import numpy as np

# Reproducibility — must be set before importing TensorFlow
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

import tensorflow as tf  # noqa: E402

tf.random.set_seed(RANDOM_SEED)

sys.path.insert(0, str(Path(__file__).parent))

from src.model.evaluator import ModelEvaluator
from src.model.lstm import LSTMBuilder
from src.model.trainer import ModelTrainer
from src.preprocessing.scaler import ClosingPriceScaler

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/phase2_model_training.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
SEQUENCE_LENGTH = 60
EPOCHS = 100
BATCH_SIZE = 32

def load_data() -> dict[str, np.ndarray]:
    logger.info("Loading preprocessed data from %s", PROCESSED_DIR)
    arrays: dict[str, np.ndarray] = {}
    for name in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
        path = PROCESSED_DIR / f"{name}.npy"
        arrays[name] = np.load(path)
        logger.info("  Loaded %-10s shape=%s", name, arrays[name].shape)
    return arrays


def load_scaler() -> ClosingPriceScaler:
    path = PROCESSED_DIR / "scaler.pkl"
    scaler = ClosingPriceScaler.load(path)
    logger.info("Scaler loaded from %s  [%.4f, %.4f]", path, scaler.data_min, scaler.data_max)
    return scaler


def print_metrics_summary(split: str, metrics) -> None:
    sep = "-" * 40
    print(sep)
    print(f"Metrics — {split}")
    print(f"  MAE  : {metrics.mae:.4f} BRL")
    print(f"  RMSE : {metrics.rmse:.4f} BRL")
    print(f"  MAPE : {metrics.mape:.2f} %")
    print(sep)


def main() -> None:
    logger.info("=" * 60)
    logger.info("PHASE 2.1 — LSTM MODEL TRAINING — START")
    logger.info("=" * 60)

    data = load_data()
    scaler = load_scaler()

    builder = LSTMBuilder()
    model = builder.build(sequence_length=SEQUENCE_LENGTH, n_features=1)
    model.summary()

    trainer = ModelTrainer(checkpoint_dir=MODELS_DIR / "checkpoints")
    history = trainer.train(
        model,
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    evaluator = ModelEvaluator()
    val_pred = model.predict(data["X_val"], verbose=0)
    test_pred = model.predict(data["X_test"], verbose=0)

    val_metrics = evaluator.calculate_metrics(data["y_val"], val_pred, scaler)
    test_metrics = evaluator.calculate_metrics(data["y_test"], test_pred, scaler)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print_metrics_summary("Validation", val_metrics)
    print_metrics_summary("Test", test_metrics)
    print("=" * 60 + "\n")

    logger.info("Val  — MAE=%.4f  RMSE=%.4f  MAPE=%.2f%%", val_metrics.mae, val_metrics.rmse, val_metrics.mape)
    logger.info("Test — MAE=%.4f  RMSE=%.4f  MAPE=%.2f%%", test_metrics.mae, test_metrics.rmse, test_metrics.mape)

    evaluator.plot_training_history(history, output_dir=REPORTS_DIR)
    evaluator.plot_predictions(data["y_val"], val_pred, scaler, split_name="Validation", output_dir=REPORTS_DIR)
    evaluator.plot_predictions(data["y_test"], test_pred, scaler, split_name="Test", output_dir=REPORTS_DIR)

    saved_model_path = MODELS_DIR / "saved_models" / "lstm_model"
    saved_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(saved_model_path))
    logger.info("Model saved -> %s", saved_model_path)

    logger.info("=" * 60)
    logger.info("PHASE 2.1 — COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
