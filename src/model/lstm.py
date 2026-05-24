import logging

import tensorflow as tf
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


class LSTMBuilder:
    """Builds and compiles the LSTM model for stock price prediction."""

    def build(
        self,
        sequence_length: int = 60,
        n_features: int = 1,
        lstm_units: int = 50,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
    ) -> tf.keras.Model:
        model = tf.keras.Sequential(
            [
                layers.Input(shape=(sequence_length, n_features)),
                layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate),
                layers.LSTM(lstm_units, dropout=dropout_rate),
                layers.Dense(25, activation="relu"),
                layers.Dense(1),
            ],
            name="lstm_stock_predictor",
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=["mae", "mse"],
        )

        logger.info(
            "Model built — sequence_length=%d, lstm_units=%d, dropout=%.2f, lr=%.4f",
            sequence_length,
            lstm_units,
            dropout_rate,
            learning_rate,
        )
        return model
