import logging

import numpy as np

logger = logging.getLogger(__name__)


class SequenceBuilder:
    """Creates overlapping sliding-window sequences for LSTM training."""

    def __init__(self, sequence_length: int = 60) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be a positive integer.")
        self.sequence_length = sequence_length

    def create_sequences(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(data)
        if n <= self.sequence_length:
            raise ValueError(
                f"Data length ({n}) must be greater than sequence_length ({self.sequence_length})."
            )

        n_samples = n - self.sequence_length
        X = np.empty((n_samples, self.sequence_length), dtype=np.float32)
        y = np.empty(n_samples, dtype=np.float32)

        for i in range(n_samples):
            X[i] = data[i : i + self.sequence_length]
            y[i] = data[i + self.sequence_length]

        # Reshape to (samples, time_steps, features=1) as required by LSTM
        X = X.reshape(n_samples, self.sequence_length, 1)

        logger.info(
            "Sequences created — X: %s  y: %s", X.shape, y.shape
        )
        return X, y
