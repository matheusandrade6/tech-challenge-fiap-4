import logging

import pandas as pd

logger = logging.getLogger(__name__)


class StockAnalyzer:
    """Performs exploratory data analysis on a stock DataFrame."""

    def __init__(self, df: pd.DataFrame, symbol: str = ""):
        self.df = df
        self.symbol = symbol

    def check_shape(self) -> None:
        print(f"\n{'='*50}")
        print(f"DATASET SHAPE — {self.symbol}")
        print(f"{'='*50}")
        print(f"Rows   : {self.df.shape[0]:,}")
        print(f"Columns: {self.df.shape[1]}")

    def check_dtypes(self) -> None:
        print(f"\n{'='*50}")
        print("DATA TYPES")
        print(f"{'='*50}")
        print(self.df.dtypes.to_string())

    def check_missing(self) -> None:
        missing = self.df.isnull().sum()
        pct = (missing / len(self.df) * 100).round(2)
        summary = pd.DataFrame({"Missing": missing, "Pct (%)": pct})
        print(f"\n{'='*50}")
        print("MISSING VALUES")
        print(f"{'='*50}")
        print(summary[summary["Missing"] > 0].to_string() if missing.any() else "No missing values.")

    def describe(self) -> pd.DataFrame:
        stats = self.df.describe().round(4)
        print(f"\n{'='*50}")
        print("DESCRIPTIVE STATISTICS")
        print(f"{'='*50}")
        print(stats.to_string())
        return stats

    def check_data_quality(self) -> None:
        print(f"\n{'='*50}")
        print("DATA QUALITY CHECKS")
        print(f"{'='*50}")

        if "Close" in self.df.columns:
            negatives = (self.df["Close"] <= 0).sum()
            print(f"Non-positive closing prices : {negatives}")

            gaps = self.df.index.to_series().diff().dt.days
            large_gaps = gaps[gaps > 5]
            print(f"Calendar gaps > 5 days      : {len(large_gaps)}")
            if not large_gaps.empty:
                print("  First 5 gaps:")
                print(large_gaps.head().to_string())

        duplicates = self.df.index.duplicated().sum()
        print(f"Duplicate dates             : {duplicates}")

    def run_full_report(self) -> pd.DataFrame:
        """Print all EDA sections and return the describe() DataFrame."""
        self.check_shape()
        self.check_dtypes()
        self.check_missing()
        self.check_data_quality()
        return self.describe()
