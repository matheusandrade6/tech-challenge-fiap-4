"""
Pydantic request / response models for the prediction API.
"""

from datetime import datetime
from typing import List

from pydantic import BaseModel, field_validator


class PredictionRequest(BaseModel):
    """Input payload for a single next-day price prediction."""

    stock_symbol: str
    historical_prices: List[float]

    @field_validator("historical_prices")
    @classmethod
    def validate_prices(cls, v: List[float]) -> List[float]:
        if len(v) != 60:
            raise ValueError(
                f"historical_prices must contain exactly 60 values, got {len(v)}"
            )
        if any(p <= 0 for p in v):
            raise ValueError("All prices must be positive numbers (> 0)")
        return v


class ConfidenceInterval(BaseModel):
    """95 % prediction interval derived from the model's test-set RMSE."""

    min: float
    max: float


class PredictionResponse(BaseModel):
    """Prediction result returned by POST /predict."""

    predicted_price: float
    confidence_interval: ConfidenceInterval
    timestamp: datetime
    model_version: str


class BatchPredictionRequest(BaseModel):
    """Payload for POST /predict-batch."""

    requests: List[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    """Results for all items in a batch prediction request."""

    predictions: List[PredictionResponse]
    total: int


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str
    model: str
    timestamp: datetime


class ModelInfoResponse(BaseModel):
    """Response schema for GET /model-info."""

    version: str
    training_date: str
    sequence_length: int
    feature_columns: List[str]
    test_metrics: dict
