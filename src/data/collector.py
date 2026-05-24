import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class StockDataCollector:
    """Downloads and persists raw historical stock data via yfinance."""

    DEFAULT_START = "2018-01-01"
    DEFAULT_END = "2024-07-20"

    def __init__(self, symbol: str, start: str = DEFAULT_START, end: str = DEFAULT_END):
        self.symbol = symbol
        self.start = start
        self.end = end
        self._raw: pd.DataFrame | None = None

    def download(self) -> pd.DataFrame:
        """Download OHLCV data from Yahoo Finance and cache it locally."""
        logger.info("Downloading %s from %s to %s", self.symbol, self.start, self.end)
        df = yf.download(self.symbol, start=self.start, end=self.end, auto_adjust=True)

        if df.empty:
            raise ValueError(f"No data returned for {self.symbol}. Check the ticker symbol.")

        # Flatten MultiIndex columns that yfinance sometimes returns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"

        self._raw = df
        logger.info("Downloaded %d rows for %s", len(df), self.symbol)
        return df

    def save_raw(self, output_dir: str | Path = "data/raw") -> Path:
        """Save the raw dataframe to CSV."""
        if self._raw is None:
            raise RuntimeError("Call download() before save_raw().")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        path = output_dir / f"{self.symbol.replace('.', '_')}_raw.csv"
        self._raw.to_csv(path)
        logger.info("Raw data saved to %s", path)
        return path

    @staticmethod
    def load_csv(path: str | Path) -> pd.DataFrame:
        """Load a previously saved raw CSV back into a DataFrame."""
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        logger.info("Loaded %d rows from %s", len(df), path)
        return df

    @property
    def raw(self) -> pd.DataFrame:
        if self._raw is None:
            raise RuntimeError("No data available yet. Call download() first.")
        return self._raw
