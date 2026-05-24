import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

CHECKPOINT = Path("models/checkpoints/best_model.keras")
SAVED_MODEL_PATH = Path("models/saved_models/lstm_model.keras")
SCALER_SRC = Path("data/processed/scaler.pkl")
SCALER_DST = Path("models/scalers/scaler.pkl")
METADATA_PATH = Path("models/metadata.json")
PROCESSED_DIR = Path("data/processed")


def main() -> None:
    import tensorflow as tf

    from src.model.evaluator import ModelEvaluator
    from src.model.persistence import ModelPersistence
    from src.preprocessing.scaler import ClosingPriceScaler

    if not CHECKPOINT.exists():
        logger.error("Checkpoint not found at %s", CHECKPOINT)
        sys.exit(1)
    logger.info("Loading checkpoint from %s", CHECKPOINT)
    model = tf.keras.models.load_model(str(CHECKPOINT))
    model.summary()

    SAVED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(SAVED_MODEL_PATH))
    logger.info("Keras model written to %s", SAVED_MODEL_PATH)

    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    scaler = ClosingPriceScaler.load(SCALER_SRC)

    evaluator = ModelEvaluator()
    test_pred = model.predict(X_test, verbose=0)
    metrics = evaluator.calculate_metrics(y_test, test_pred, scaler)
    test_metrics = metrics.as_dict()
    logger.info(
        "Test metrics — MAE=%.4f  RMSE=%.4f  MAPE=%.2f%%",
        test_metrics.get("mae") or test_metrics.get("MAE"),
        test_metrics.get("rmse") or test_metrics.get("RMSE"),
        test_metrics.get("mape") or test_metrics.get("MAPE"),
    )

    SCALER_DST.parent.mkdir(parents=True, exist_ok=True)
    persistence = ModelPersistence(
        model_path=SAVED_MODEL_PATH,
        scaler_path=SCALER_DST,
        metadata_path=METADATA_PATH,
    )
    persistence.save(
        model=model,
        scaler=scaler,
        test_metrics=test_metrics,
        sequence_length=60,
        feature_columns=["Close"],
        model_version="v1.0",
    )
    logger.info("All artifacts ready — API can now be started.")


if __name__ == "__main__":
    main()
