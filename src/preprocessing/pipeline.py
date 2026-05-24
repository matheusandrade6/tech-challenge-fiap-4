import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .scaler import ClosingPriceScaler
from .sequencer import SequenceBuilder
from .splitter import SplitResult, TemporalSplitter

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingArtifacts:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: ClosingPriceScaler
    metadata: dict


class PreprocessingPipeline:
    """End-to-end preprocessing pipeline for LSTM training."""

    def __init__(
        self,
        sequence_length: int = 60,
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        output_dir: str | Path = "data/processed",
    ) -> None:
        self.sequence_length = sequence_length
        self.output_dir = Path(output_dir)
        self._splitter = TemporalSplitter(train_ratio, val_ratio, test_ratio)
        self._scaler = ClosingPriceScaler()
        self._sequencer = SequenceBuilder(sequence_length)

    def run(self, df: pd.DataFrame, price_column: str = "Close") -> PreprocessingArtifacts:
        """Execute the full preprocessing pipeline and return all artifacts."""
        if price_column not in df.columns:
            raise ValueError(f"Column '{price_column}' not found in DataFrame.")

        logger.info("Starting preprocessing pipeline  (sequence_length=%d)", self.sequence_length)

        split = self._splitter.split(df[price_column])
        logger.info("Data split complete.")

        scaled_train = self._scaler.fit_transform(split.train)
        scaled_val = self._scaler.transform(split.val)
        scaled_test = self._scaler.transform(split.test)
        logger.info("Normalisation complete  (fit on training set only).")

        X_train, y_train = self._sequencer.create_sequences(scaled_train)
        X_val, y_val = self._sequencer.create_sequences(scaled_val)
        X_test, y_test = self._sequencer.create_sequences(scaled_test)

        metadata = self._build_metadata(split, X_train, X_val, X_test)

        artifacts = PreprocessingArtifacts(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            scaler=self._scaler,
            metadata=metadata,
        )

        return artifacts

    def save(self, artifacts: PreprocessingArtifacts) -> dict[str, Path]:
        """Persist all preprocessing artifacts to `output_dir`."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        saved: dict[str, Path] = {}

        array_map = {
            "X_train": artifacts.X_train,
            "y_train": artifacts.y_train,
            "X_val": artifacts.X_val,
            "y_val": artifacts.y_val,
            "X_test": artifacts.X_test,
            "y_test": artifacts.y_test,
        }
        for name, array in array_map.items():
            path = self.output_dir / f"{name}.npy"
            np.save(path, array)
            saved[name] = path
            logger.info("Saved %s -> %s  shape=%s", name, path, array.shape)

        scaler_path = self.output_dir / "scaler.pkl"
        artifacts.scaler.save(scaler_path)
        saved["scaler"] = scaler_path

        meta_path = self.output_dir / "preprocessing_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(artifacts.metadata, f, indent=2, default=str)
        saved["metadata"] = meta_path
        logger.info("Metadata saved -> %s", meta_path)

        return saved

    def _build_metadata(
        self,
        split: SplitResult,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
    ) -> dict:
        return {
            "sequence_length": self.sequence_length,
            "train_dates": {
                "start": str(split.train_dates.min().date()),
                "end": str(split.train_dates.max().date()),
                "n_days": len(split.train_dates),
            },
            "val_dates": {
                "start": str(split.val_dates.min().date()),
                "end": str(split.val_dates.max().date()),
                "n_days": len(split.val_dates),
            },
            "test_dates": {
                "start": str(split.test_dates.min().date()),
                "end": str(split.test_dates.max().date()),
                "n_days": len(split.test_dates),
            },
            "original_min": self._scaler.data_min,
            "original_max": self._scaler.data_max,
            "shapes": {
                "X_train": list(X_train.shape),
                "X_val": list(X_val.shape),
                "X_test": list(X_test.shape),
            },
        }
