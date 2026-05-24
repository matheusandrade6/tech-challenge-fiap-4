import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Adds derived financial features to a stock OHLCV DataFrame."""

    def __init__(self, df: pd.DataFrame):
        if "Close" not in df.columns:
            raise ValueError("DataFrame must contain a 'Close' column.")
        self.df = df.copy()

    def add_daily_returns(self) -> "FeatureEngineer":
        """Percentage change in closing price day over day."""
        self.df["Daily_Return"] = self.df["Close"].pct_change()
        logger.debug("Added Daily_Return column.")
        return self

    def add_rolling_volatility(self, window: int = 20) -> "FeatureEngineer":
        """Annualised rolling standard deviation of daily returns."""
        col = f"Volatility_{window}d"
        self.df[col] = (
            self.df["Daily_Return"].rolling(window=window).std() * np.sqrt(252)
        )
        logger.debug("Added %s column.", col)
        return self

    def add_moving_averages(self, windows: list[int] | None = None) -> "FeatureEngineer":
        """Simple moving averages of the closing price."""
        if windows is None:
            windows = [20, 50, 200]
        for w in windows:
            col = f"MA_{w}"
            self.df[col] = self.df["Close"].rolling(window=w).mean()
            logger.debug("Added %s column.", col)
        return self

    def add_all(self, vol_window: int = 20, ma_windows: list[int] | None = None) -> "FeatureEngineer":
        """Convenience method — adds returns, volatility, and MAs in one call."""
        return (
            self.add_daily_returns()
            .add_rolling_volatility(vol_window)
            .add_moving_averages(ma_windows)
        )

    def build(self) -> pd.DataFrame:
        """Return the enriched DataFrame."""
        return self.df
