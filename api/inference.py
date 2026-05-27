import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Ensure the project root is on sys.path so ``src.*`` imports resolve when the
# module is imported from any working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from api.config import settings  # noqa: E402

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Manages LSTM artifact lifecycle and price prediction."""

    def __init__(self) -> None:
        self._artifacts = None  # type: Optional[object]
        self._inference = None  # type: Optional[object]

    def load(self) -> None:
        # NOTE: TensorFlow is imported lazily here (inside load()) rather than at
        # module level.  A top-level TF import takes ~30 s on cold start, which
        # delays uvicorn's port binding long enough for Render to report
        # "No open ports detected" and fail the deploy.  By deferring the import
        # to this method — which runs inside the FastAPI lifespan, *after* the
        # socket is already bound — the app starts accepting connections immediately
        # while the model is loading in the background startup hook.
        from src.model.persistence import ModelInference, ModelPersistence  # noqa: E402

        logger.info(
            "Loading model artifacts — model=%s  scaler=%s  metadata=%s",
            settings.MODEL_PATH,
            settings.SCALER_PATH,
            settings.METADATA_PATH,
        )
        t0 = time.perf_counter()
        self._inference = ModelInference()
        persistence = ModelPersistence(
            model_path=settings.MODEL_PATH,
            scaler_path=settings.SCALER_PATH,
            metadata_path=settings.METADATA_PATH,
        )
        self._artifacts = persistence.load()
        elapsed = time.perf_counter() - t0
        meta = self._artifacts.metadata
        logger.info(
            "Model ready in %.2fs — version=%s  trained_at=%s  "
            "test_rmse=%.4f  test_mae=%.4f  test_mape=%.2f%%",
            elapsed,
            meta.get("model_version"),
            meta.get("training_date"),
            meta.get("test_metrics", {}).get("rmse", float("nan")),
            meta.get("test_metrics", {}).get("mae", float("nan")),
            meta.get("test_metrics", {}).get("mape", float("nan")),
        )

    @property
    def is_loaded(self) -> bool:
        """``True`` after a successful call to :meth:`load`."""
        return self._artifacts is not None

    @property
    def metadata(self) -> dict:
        """Metadata dict from the saved model (version, dates, metrics)."""
        self._require_loaded()
        return self._artifacts.metadata  # type: ignore[union-attr]

    def predict(self, prices: List[float]) -> Tuple[float, float, float]:
        self._require_loaded()

        prices_arr = np.asarray(prices, dtype=float)

        if prices_arr.ndim != 1 or len(prices_arr) != settings.SEQUENCE_LENGTH:
            raise ValueError(
                f"Expected exactly {settings.SEQUENCE_LENGTH} prices, got {len(prices_arr)}"
            )
        if np.any(prices_arr <= 0):
            raise ValueError("All prices must be positive numbers (> 0)")

        t0 = time.perf_counter()
        predicted = self._inference.predict_next_price(  # type: ignore[union-attr]
            last_60_days=prices_arr,
            model=self._artifacts.model,  # type: ignore[union-attr]
            scaler=self._artifacts.scaler,  # type: ignore[union-attr]
        )
        latency = time.perf_counter() - t0

        rmse: float = (
            self._artifacts.metadata.get("test_metrics", {}).get("rmse") or 0.0  # type: ignore[union-attr]
        )
        ci_margin = settings.CI_Z_SCORE * rmse
        ci_min = max(0.0, predicted - ci_margin)
        ci_max = predicted + ci_margin

        logger.info(
            "Inference — predicted=%.4f  ci=[%.4f, %.4f]  latency=%.4fs",
            predicted,
            ci_min,
            ci_max,
            latency,
        )
        return predicted, ci_min, ci_max

    def _require_loaded(self) -> None:
        if not self.is_loaded:
            raise RuntimeError(
                "InferenceEngine is not loaded. Call engine.load() before making predictions."
            )


engine = InferenceEngine()
