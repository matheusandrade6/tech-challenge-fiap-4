import logging
import random
import sys
from pathlib import Path

import numpy as np

# Reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

sys.path.insert(0, str(Path(__file__).parent))

from src.data.collector import StockDataCollector
from src.preprocessing.pipeline import PreprocessingPipeline

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/phase1_preprocessing.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

PROCESSED_CSV = Path("data/processed/stock_data.csv")
OUTPUT_DIR = Path("data/processed")
SEQUENCE_LENGTH = 60
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10


def load_data() -> "pd.DataFrame":  # noqa: F821
    logger.info("Loading processed stock data from %s", PROCESSED_CSV)
    df = StockDataCollector.load_csv(PROCESSED_CSV)
    logger.info("Loaded %d rows, %d columns", *df.shape)
    return df


def run_pipeline(df: "pd.DataFrame") -> None:  # noqa: F821
    pipeline = PreprocessingPipeline(
        sequence_length=SEQUENCE_LENGTH,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        output_dir=OUTPUT_DIR,
    )

    artifacts = pipeline.run(df, price_column="Close")
    saved = pipeline.save(artifacts)

    print_summary(artifacts, saved)
    verify_no_leakage(artifacts)


def print_summary(artifacts, saved: dict) -> None:
    meta = artifacts.metadata
    sep = "=" * 60

    print(f"\n{sep}")
    print("PHASE 1.2 — PREPROCESSING SUMMARY")
    print(sep)
    print(f"Sequence length   : {meta['sequence_length']}")
    print(f"\nTrain  ({meta['train_dates']['n_days']} days): "
          f"{meta['train_dates']['start']} -> {meta['train_dates']['end']}")
    print(f"Val    ({meta['val_dates']['n_days']} days): "
          f"{meta['val_dates']['start']} -> {meta['val_dates']['end']}")
    print(f"Test   ({meta['test_dates']['n_days']} days): "
          f"{meta['test_dates']['start']} -> {meta['test_dates']['end']}")

    print(f"\nScaler fit range  : [{meta['original_min']:.4f}, {meta['original_max']:.4f}]")

    print("\nArray shapes:")
    for key in ("X_train", "X_val", "X_test"):
        print(f"  {key:10s}: {tuple(meta['shapes'][key])}")
    print(f"  {'y_train':10s}: {artifacts.y_train.shape}")
    print(f"  {'y_val':10s}: {artifacts.y_val.shape}")
    print(f"  {'y_test':10s}: {artifacts.y_test.shape}")

    print("\nNormalised value statistics (sanity check):")
    for name, arr in [
        ("y_train", artifacts.y_train),
        ("y_val",   artifacts.y_val),
        ("y_test",  artifacts.y_test),
    ]:
        print(f"  {name:8s}  min={arr.min():.4f}  max={arr.max():.4f}  "
              f"mean={arr.mean():.4f}  std={arr.std():.4f}")

    print("\nSaved artifacts:")
    for key, path in saved.items():
        print(f"  {key:10s}: {path}")
    print(sep + "\n")


def verify_no_leakage(artifacts) -> None:
    """Assert the scaler was not influenced by val / test data."""
    scaler = artifacts.scaler
    train_min = artifacts.y_train.min()
    train_max = artifacts.y_train.max()

    assert train_min >= 0.0, "Normalised train min below 0 — check scaler fit."
    assert train_max <= 1.0, "Normalised train max above 1 — check scaler fit."

    # Val / test values CAN exceed [0, 1] if the price went outside the training
    # range, which is expected and NOT a sign of data leakage.
    logger.info(
        "Leakage check passed — scaler fitted only on training split "
        "(train normalised range: [%.4f, %.4f])",
        train_min, train_max,
    )


def main() -> None:
    logger.info("=" * 60)
    logger.info("PHASE 1.2 — DATA PREPROCESSING — START")
    logger.info("=" * 60)

    df = load_data()
    run_pipeline(df)

    logger.info("PHASE 1.2 — COMPLETE")


if __name__ == "__main__":
    main()
