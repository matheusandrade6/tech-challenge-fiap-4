import asyncio
import time
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from typing import AsyncGenerator, List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from api.config import settings, setup_logging
from api.inference import engine
from api.metrics import metrics
from api.models import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ConfidenceInterval,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from api.monitoring import generate_dashboard

import logging

setup_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


async def _metrics_save_task() -> None:
    """Persist metrics JSON to disk every hour."""
    while True:
        await asyncio.sleep(3600)
        metrics.save_to_file(settings.METRICS_PATH)
        metrics.check_error_rate_alert(settings.ERROR_RATE_ALERT_THRESHOLD)


async def _dashboard_update_task() -> None:
    """Regenerate the HTML monitoring dashboard every 5 minutes."""
    while True:
        await asyncio.sleep(300)
        generate_dashboard(settings.DASHBOARD_PATH, metrics.get_stats())


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("API starting up — loading model artifacts...")
    try:
        engine.load()
        logger.info("Model artifacts loaded successfully.")
    except Exception as exc:
        logger.error("Failed to load model on startup: %s", exc, exc_info=True)
        raise

    # Write an initial dashboard so it exists immediately after startup
    generate_dashboard(settings.DASHBOARD_PATH, metrics.get_stats())

    task_metrics = asyncio.create_task(_metrics_save_task())
    task_dashboard = asyncio.create_task(_dashboard_update_task())

    yield

    task_metrics.cancel()
    task_dashboard.cancel()
    with suppress(asyncio.CancelledError):
        await task_metrics
        await task_dashboard

    metrics.save_to_file(settings.METRICS_PATH)
    logger.info("API shutting down.")


app = FastAPI(
    title="LSTM Stock Price Prediction API",
    description=(
        "RESTful API for next-day closing price prediction using an LSTM model "
        "trained on PETR4.SA historical data.\n\n"
        "Submit the last **60 daily closing prices** to `/predict` and receive "
        "a predicted next-day price together with a 95 % confidence interval."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Meta"],
    summary="API and model liveness check",
)
async def health() -> HealthResponse:
    """Return API status and whether the model is loaded.

    Always returns HTTP 200.  Consumers should check the ``model`` field for
    the string ``"loaded"`` to confirm predictions are available.
    """
    return HealthResponse(
        status="ok",
        model="loaded" if engine.is_loaded else "not_loaded",
        timestamp=datetime.now(tz=timezone.utc),
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["Meta"],
    summary="Model version and accuracy metrics",
)
async def model_info() -> ModelInfoResponse:
    """Return metadata about the currently loaded model.

    Includes version, training date, expected sequence length, and
    test-set accuracy metrics (MAE, RMSE, MAPE).

    Raises HTTP 503 if the model is not yet loaded.
    """
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    meta = engine.metadata
    return ModelInfoResponse(
        version=meta.get("model_version", "unknown"),
        training_date=meta.get("training_date", "unknown"),
        sequence_length=meta.get("sequence_length", settings.SEQUENCE_LENGTH),
        feature_columns=meta.get("feature_columns", ["Close"]),
        test_metrics=meta.get("test_metrics", {}),
    )


@app.get(
    "/metrics",
    tags=["Meta"],
    summary="Live operational metrics",
)
async def get_metrics() -> dict:
    """Return current API metrics: uptime, prediction counts, error rate, and inference latency."""
    return metrics.get_stats()


@app.get(
    "/dashboard",
    tags=["Meta"],
    summary="HTML monitoring dashboard",
)
async def dashboard() -> FileResponse:
    """Serve the static HTML monitoring dashboard.

    The dashboard is regenerated every 5 minutes in the background.
    """
    if not settings.DASHBOARD_PATH.exists():
        generate_dashboard(settings.DASHBOARD_PATH, metrics.get_stats())
    return FileResponse(settings.DASHBOARD_PATH, media_type="text/html")


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Single next-day price prediction",
    status_code=200,
)
async def predict(request: PredictionRequest) -> PredictionResponse:
    if not engine.is_loaded:
        metrics.record_error("/predict", "Model not loaded")
        raise HTTPException(status_code=503, detail="Model is not loaded")

    logger.info(
        "POST /predict — symbol=%s  n=%d  first=%.2f  last=%.2f",
        request.stock_symbol,
        len(request.historical_prices),
        request.historical_prices[0],
        request.historical_prices[-1],
    )

    t0 = time.perf_counter()
    try:
        predicted, ci_min, ci_max = engine.predict(request.historical_prices)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Inference error for %s: %s", request.stock_symbol, exc, exc_info=True
        )
        metrics.record_error("/predict", str(exc))
        raise HTTPException(status_code=500, detail="Internal inference error")

    latency = time.perf_counter() - t0
    metrics.record_prediction(latency)

    response = PredictionResponse(
        predicted_price=predicted,
        confidence_interval=ConfidenceInterval(min=ci_min, max=ci_max),
        timestamp=datetime.now(tz=timezone.utc),
        model_version=engine.metadata.get("model_version", "unknown"),
    )
    logger.info(
        "POST /predict — symbol=%s  predicted=%.4f  ci=[%.4f, %.4f]  latency=%.4fs",
        request.stock_symbol,
        predicted,
        ci_min,
        ci_max,
        latency,
    )
    return response


@app.post(
    "/predict-batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Batch next-day price predictions",
    status_code=200,
)
async def predict_batch(batch: BatchPredictionRequest) -> BatchPredictionResponse:
    if not engine.is_loaded:
        metrics.record_error("/predict-batch", "Model not loaded")
        raise HTTPException(status_code=503, detail="Model is not loaded")

    if not batch.requests:
        raise HTTPException(status_code=400, detail="requests list must not be empty")

    logger.info("POST /predict-batch — n_requests=%d", len(batch.requests))

    results: List[PredictionResponse] = []
    for req in batch.requests:
        t0 = time.perf_counter()
        try:
            predicted, ci_min, ci_max = engine.predict(req.historical_prices)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"symbol={req.stock_symbol}: {exc}",
            )
        except Exception as exc:
            logger.error(
                "Batch inference error for %s: %s", req.stock_symbol, exc, exc_info=True
            )
            metrics.record_error("/predict-batch", f"{req.stock_symbol}: {exc}")
            raise HTTPException(
                status_code=500,
                detail=f"Internal inference error for symbol={req.stock_symbol}",
            )

        latency = time.perf_counter() - t0
        metrics.record_prediction(latency)

        results.append(
            PredictionResponse(
                predicted_price=predicted,
                confidence_interval=ConfidenceInterval(min=ci_min, max=ci_max),
                timestamp=datetime.now(tz=timezone.utc),
                model_version=engine.metadata.get("model_version", "unknown"),
            )
        )

    logger.info("POST /predict-batch — completed %d predictions", len(results))
    return BatchPredictionResponse(predictions=results, total=len(results))


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
