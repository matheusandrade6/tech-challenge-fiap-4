import logging
import random
import sys
from pathlib import Path

import numpy as np

# Reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Make src importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.data.collector import StockDataCollector
from src.data.feature_engineering import FeatureEngineer
from src.eda.analyzer import StockAnalyzer
from src.visualization.plotter import StockPlotter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/phase1_eda.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

SYMBOL = "PETR4.SA"
START_DATE = "2018-01-01"
END_DATE = "2024-07-20"

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(exist_ok=True)


def collect_data() -> "pd.DataFrame":  # noqa: F821
    collector = StockDataCollector(SYMBOL, start=START_DATE, end=END_DATE)
    df = collector.download()
    collector.save_raw(RAW_DIR)
    return df


def run_eda(df: "pd.DataFrame") -> None:  # noqa: F821
    analyzer = StockAnalyzer(df, symbol=SYMBOL)
    analyzer.run_full_report()


def engineer_features(df: "pd.DataFrame") -> "pd.DataFrame":  # noqa: F821
    enriched = (
        FeatureEngineer(df)
        .add_all(vol_window=20, ma_windows=[20, 50, 200])
        .build()
    )
    logger.info("Feature engineering complete. Shape: %s", enriched.shape)
    return enriched


def visualize(df: "pd.DataFrame") -> None:  # noqa: F821
    plotter = StockPlotter(df, symbol=SYMBOL, output_dir=PROCESSED_DIR)
    plotter.plot_all(save=True)
    logger.info("All plots saved to %s", PROCESSED_DIR)


def save_processed(df: "pd.DataFrame") -> Path:  # noqa: F821
    out = PROCESSED_DIR / "stock_data.csv"
    df.to_csv(out)
    logger.info("Processed data saved to %s  (rows=%d, cols=%d)", out, *df.shape)
    return out


def print_summary(df: "pd.DataFrame") -> None:  # noqa: F821
    print("\n" + "=" * 50)
    print("FINAL SUMMARY — PROCESSED DATASET")
    print("=" * 50)
    print(f"Symbol    : {SYMBOL}")
    print(f"Period    : {df.index.min().date()} -> {df.index.max().date()}")
    print(f"Shape     : {df.shape}")
    print(f"Columns   : {list(df.columns)}")
    print(f"Null count: {df.isnull().sum().sum()}")

    if "Daily_Return" in df.columns:
        r = df["Daily_Return"].dropna()
        print(f"\nDaily Return stats:")
        print(f"  Mean  : {r.mean():.4%}")
        print(f"  Std   : {r.std():.4%}")
        print(f"  Min   : {r.min():.4%}")
        print(f"  Max   : {r.max():.4%}")


def main() -> None:
    logger.info("=" * 60)
    logger.info("PHASE 1.1 — DATA COLLECTION & EDA — START")
    logger.info("=" * 60)

    df_raw = collect_data()
    run_eda(df_raw)
    df_enriched = engineer_features(df_raw)
    visualize(df_enriched)
    save_processed(df_enriched)
    print_summary(df_enriched)

    logger.info("PHASE 1.1 — COMPLETE")


if __name__ == "__main__":
    main()
