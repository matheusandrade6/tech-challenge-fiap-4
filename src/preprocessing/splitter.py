import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitResult:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    train_dates: pd.DatetimeIndex
    val_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex


class TemporalSplitter:
    """Splits a 1-D price series into train / val / test preserving temporal order."""

    def __init__(
        self,
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
    ) -> None:
        if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split(self, series: pd.Series) -> SplitResult:
        """Return train / val / test arrays and their corresponding DatetimeIndex slices."""
        n = len(series)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        train = series.iloc[:train_end]
        val = series.iloc[train_end:val_end]
        test = series.iloc[val_end:]

        logger.info(
            "Split sizes — train: %d, val: %d, test: %d",
            len(train), len(val), len(test),
        )
        logger.info(
            "Date ranges -- train: %s->%s | val: %s->%s | test: %s->%s",
            train.index.min().date(), train.index.max().date(),
            val.index.min().date(), val.index.max().date(),
            test.index.min().date(), test.index.max().date(),
        )

        return SplitResult(
            train=train.values.astype(float),
            val=val.values.astype(float),
            test=test.values.astype(float),
            train_dates=train.index,
            val_dates=val.index,
            test_dates=test.index,
        )
