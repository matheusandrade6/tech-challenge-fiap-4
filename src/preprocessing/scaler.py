import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class ClosingPriceScaler:
    """MinMaxScaler wrapper that enforces fit-only-on-train discipline."""

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        self._scaler = MinMaxScaler(feature_range=feature_range)
        self._is_fitted = False

    def fit(self, train_data: np.ndarray) -> "ClosingPriceScaler":
        """Fit the scaler ONLY on training data to prevent data leakage."""
        self._scaler.fit(train_data.reshape(-1, 1))
        self._is_fitted = True
        logger.info(
            "Scaler fitted — data_min=%.4f  data_max=%.4f",
            self._scaler.data_min_[0],
            self._scaler.data_max_[0],
        )
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return self._scaler.transform(data.reshape(-1, 1)).flatten()

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return self._scaler.inverse_transform(data.reshape(-1, 1)).flatten()

    def fit_transform(self, train_data: np.ndarray) -> np.ndarray:
        return self.fit(train_data).transform(train_data)

    def save(self, path: str | Path) -> Path:
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._scaler, f)
        logger.info("Scaler saved to %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ClosingPriceScaler":
        path = Path(path)
        with open(path, "rb") as f:
            inner = pickle.load(f)
        instance = cls()
        instance._scaler = inner
        instance._is_fitted = True
        logger.info("Scaler loaded from %s", path)
        return instance

    @property
    def data_min(self) -> float:
        self._require_fitted()
        return float(self._scaler.data_min_[0])

    @property
    def data_max(self) -> float:
        self._require_fitted()
        return float(self._scaler.data_max_[0])

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Scaler has not been fitted. Call fit() first.")
