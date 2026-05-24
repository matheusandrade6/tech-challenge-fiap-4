import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.model.evaluator import ModelEvaluator
from src.model.lstm import LSTMBuilder
from src.model.persistence import ModelInference, ModelPersistence, get_model_version
from src.preprocessing.scaler import ClosingPriceScaler

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/phase3_model_persistence.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")


def load_test_data() -> tuple[np.ndarray, np.ndarray]:
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    logger.info("Test data loaded — X_test=%s  y_test=%s", X_test.shape, y_test.shape)
    return X_test, y_test


def load_preprocessing_scaler() -> ClosingPriceScaler:
    path = PROCESSED_DIR / "scaler.pkl"
    scaler = ClosingPriceScaler.load(path)
    logger.info("Preprocessing scaler loaded from %s", path)
    return scaler


def compute_test_metrics(scaler: ClosingPriceScaler) -> dict:
    X_test, y_test = load_test_data()

    import tensorflow as tf
    model_path = MODELS_DIR / "saved_models" / "lstm_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Run phase2_model_training.py first."
        )
    model = tf.keras.models.load_model(str(model_path))

    evaluator = ModelEvaluator()
    test_pred = model.predict(X_test, verbose=0)
    metrics = evaluator.calculate_metrics(y_test, test_pred, scaler)
    return metrics.as_dict(), model


def print_section(title: str) -> None:
    sep = "=" * 60
    print(f"\n{sep}\n{title}\n{sep}")


def step_save(model, scaler: ClosingPriceScaler, test_metrics: dict) -> None:
    print_section("STEP 1 — SAVING MODEL ARTIFACTS")
    persistence = ModelPersistence(
        model_path=MODELS_DIR / "saved_models" / "lstm_model.keras",
        scaler_path=MODELS_DIR / "scalers" / "scaler.pkl",
        metadata_path=MODELS_DIR / "metadata.json",
    )
    paths = persistence.save(
        model=model,
        scaler=scaler,
        test_metrics=test_metrics,
        sequence_length=60,
        feature_columns=["Close"],
        model_version="v1.0",
    )
    logger.info("Saved paths: %s", {k: str(v) for k, v in paths.items()})


def step_load_and_verify() -> None:
    print_section("STEP 2 — LOADING MODEL ARTIFACTS")
    persistence = ModelPersistence(
        model_path=MODELS_DIR / "saved_models" / "lstm_model.keras",
        scaler_path=MODELS_DIR / "scalers" / "scaler.pkl",
        metadata_path=MODELS_DIR / "metadata.json",
    )
    artifacts = persistence.load()

    print(f"  model_version  : {artifacts.metadata['model_version']}")
    print(f"  training_date  : {artifacts.metadata['training_date']}")
    print(f"  sequence_length: {artifacts.metadata['sequence_length']}")
    print(f"  feature_columns: {artifacts.metadata['feature_columns']}")
    metrics = artifacts.metadata["test_metrics"]
    print(f"  test_metrics   : MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  MAPE={metrics['mape']:.2f}%")
    return artifacts


def step_predict(artifacts) -> None:
    print_section("STEP 3 — INFERENCE VERIFICATION")
    X_test, _ = load_test_data()

    # The stored sequences are already normalized; inverse-transform to raw prices
    # so predict_next_price() receives un-normalized input (as the API will).
    last_sequence_norm = X_test[-1].flatten()
    last_sequence_raw = artifacts.scaler.inverse_transform(last_sequence_norm)

    inference = ModelInference()
    predicted = inference.predict_next_price(
        last_60_days=last_sequence_raw,
        model=artifacts.model,
        scaler=artifacts.scaler,
    )
    print(f"  Input  : last 60 days prices  min={last_sequence_raw.min():.2f}  max={last_sequence_raw.max():.2f}")
    print(f"  Output : predicted next-day price = {predicted:.4f} BRL")

    try:
        inference.predict_next_price(np.zeros(30), artifacts.model, artifacts.scaler)
    except ValueError as exc:
        print(f"  Edge-case guard: {exc}")


def step_version_info() -> None:
    print_section("STEP 4 — VERSION INFO")
    info = get_model_version(MODELS_DIR / "metadata.json")
    for key, value in info.items():
        print(f"  {key}: {value}")


def main() -> None:
    logger.info("=" * 60)
    logger.info("PHASE 3.1 — MODEL PERSISTENCE & UTILITIES — START")
    logger.info("=" * 60)

    scaler = load_preprocessing_scaler()
    test_metrics, model = compute_test_metrics(scaler)

    step_save(model, scaler, test_metrics)
    artifacts = step_load_and_verify()
    step_predict(artifacts)
    step_version_info()

    logger.info("=" * 60)
    logger.info("PHASE 3.1 — COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
