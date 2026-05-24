"""
Metrics tracking, drift detection, and alerting for the prediction API.
"""

import json
import logging
import math
import threading
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import List

logger = logging.getLogger(__name__)


def detect_drift(
    recent_predictions: List[float],
    recent_actuals: List[float],
    baseline_rmse: float,
    threshold: float = 0.15,
) -> bool:
    if baseline_rmse <= 0 or not recent_predictions or not recent_actuals:
        return False

    n = min(len(recent_predictions), len(recent_actuals))
    rmse = math.sqrt(
        sum((p - a) ** 2 for p, a in zip(recent_predictions[:n], recent_actuals[:n])) / n
    )

    if rmse > baseline_rmse * (1.0 + threshold):
        pct_increase = (rmse / baseline_rmse - 1.0) * 100.0
        logger.warning(
            "Model drift detected — current_rmse=%.4f  baseline_rmse=%.4f  increase=%.1f%%",
            rmse,
            baseline_rmse,
            pct_increase,
        )
        return True

    return False


def alert_critical(message: str, context: dict | None = None) -> None:
    """Log a CRITICAL-level alert with optional structured context."""
    logger.critical("ALERT: %s | context=%s", message, context or {})


class MetricsTracker:
    """Thread-safe in-memory tracker for API operational metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.start_time: datetime = datetime.now(timezone.utc)
        self.total_predictions: int = 0
        self.total_errors: int = 0
        self.inference_times: List[float] = []
        self._recent_errors: List[dict] = []

    def record_prediction(self, inference_time: float) -> None:
        """Increment prediction counter and record latency in seconds."""
        with self._lock:
            self.total_predictions += 1
            self.inference_times.append(inference_time)

    def record_error(self, endpoint: str, detail: str) -> None:
        """Increment error counter and append to the rolling error log (last 20)."""
        with self._lock:
            self.total_errors += 1
            self._recent_errors.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "endpoint": endpoint,
                    "detail": detail,
                }
            )
            if len(self._recent_errors) > 20:
                self._recent_errors.pop(0)

    def get_stats(self) -> dict:
        """Return a snapshot of current metrics as a plain dict."""
        with self._lock:
            uptime = datetime.now(timezone.utc) - self.start_time
            uptime_seconds = uptime.total_seconds()
            avg_inf_ms = (
                mean(self.inference_times) * 1000.0 if self.inference_times else 0.0
            )
            return {
                "uptime_seconds": uptime_seconds,
                "uptime_human": str(uptime).split(".")[0],
                "total_predictions": self.total_predictions,
                "predictions_per_hour": self.total_predictions
                / max(1.0, uptime_seconds / 3600.0),
                "total_errors": self.total_errors,
                "error_rate": self.total_errors / max(1, self.total_predictions),
                "avg_inference_time_ms": avg_inf_ms,
                "recent_errors": list(self._recent_errors[-10:]),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def save_to_file(self, path: Path) -> None:
        """Persist the current stats snapshot to a JSON file."""
        stats = self.get_stats()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(stats, fh, indent=2)
            logger.debug("Metrics saved to %s", path)
        except Exception as exc:
            logger.error("Failed to save metrics: %s", exc, exc_info=True)

    def check_error_rate_alert(self, threshold: float) -> None:
        """Emit a CRITICAL alert if the current error rate exceeds *threshold*."""
        with self._lock:
            rate = self.total_errors / max(1, self.total_predictions)
        if rate > threshold:
            alert_critical(
                f"Error rate {rate:.2%} exceeds alert threshold {threshold:.2%}",
                {"total_predictions": self.total_predictions, "total_errors": self.total_errors},
            )


metrics = MetricsTracker()
