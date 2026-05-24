import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings:
    """Centralised configuration.  One singleton instance is exported as ``settings``."""

    def __init__(self) -> None:
        self.MODEL_PATH = Path(
            os.getenv("MODEL_PATH", str(_PROJECT_ROOT / "models" / "saved_models" / "lstm_model.keras"))
        )
        self.SCALER_PATH = Path(
            os.getenv("SCALER_PATH", str(_PROJECT_ROOT / "models" / "scalers" / "scaler.pkl"))
        )
        self.METADATA_PATH = Path(
            os.getenv("METADATA_PATH", str(_PROJECT_ROOT / "models" / "metadata.json"))
        )

        self.HOST: str = os.getenv("API_HOST", "0.0.0.0")
        self.PORT: int = int(os.getenv("API_PORT", "8000"))
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

        cors_raw = os.getenv("CORS_ORIGINS", "*")
        self.CORS_ORIGINS: list[str] = [o.strip() for o in cors_raw.split(",")]

        self.SEQUENCE_LENGTH: int = 60
        # z-score for 95 % confidence interval
        self.CI_Z_SCORE: float = 1.96

        self.METRICS_PATH = Path(
            os.getenv("METRICS_PATH", str(_PROJECT_ROOT / "logs" / "metrics.json"))
        )
        self.DASHBOARD_PATH = Path(
            os.getenv("DASHBOARD_PATH", str(_PROJECT_ROOT / "logs" / "dashboard.html"))
        )
        # Alert threshold: error rate above this fraction triggers a critical log
        self.ERROR_RATE_ALERT_THRESHOLD: float = float(
            os.getenv("ERROR_RATE_ALERT_THRESHOLD", "0.05")
        )
        # Drift threshold: RMSE increase above this fraction triggers a warning
        self.DRIFT_THRESHOLD: float = float(os.getenv("DRIFT_THRESHOLD", "0.15"))


settings = Settings()


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a rotating file handler and a console handler."""
    logs_dir = _PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = TimedRotatingFileHandler(
        logs_dir / "api.log",
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # Avoid duplicate handlers if setup_logging is called more than once
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)
