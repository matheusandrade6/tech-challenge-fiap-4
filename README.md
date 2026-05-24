# LSTM Stock Price Prediction API

A RESTful API that predicts the next-day closing price of **PETR4.SA** (Petrobras) using a Long Short-Term Memory (LSTM) neural network. Submit the last 60 daily closing prices and receive a predicted price together with a 95% confidence interval.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Step 1 — Train the Model](#step-1--train-the-model)
- [Step 2 — Run the API Locally](#step-2--run-the-api-locally)
- [Step 3 — Deploy with Docker](#step-3--deploy-with-docker)
- [Step 4 — Deploy to Render](#step-4--deploy-to-render)
- [API Reference](#api-reference)
- [Environment Variables](#environment-variables)
- [Smoke Testing](#smoke-testing)

---

## Architecture Overview

```
Historical prices (yfinance)
        │
        ▼
  Phase 1a: EDA          ← phase1_eda.py
  Phase 1b: Preprocessing ← phase1_preprocessing.py
        │
        ▼
  Phase 2: LSTM Training  ← phase2_model_training.py
        │
        ▼
  Phase 3: Model Persistence ← phase3_model_persistence.py
        │
        ▼
  FastAPI REST API        ← api/main.py
  POST /predict → predicted_price + 95% CI
```

---

## Tech Stack

| Layer | Library |
|---|---|
| Model | TensorFlow 2.18 / Keras LSTM |
| API | FastAPI + Uvicorn |
| Data | yfinance, pandas, NumPy |
| Preprocessing | scikit-learn |
| Containerisation | Docker + Docker Compose |
| Cloud | Render (render.yaml included) |

---

## Project Structure

```
.
├── api/                    # FastAPI application
│   ├── main.py             # Routes and lifespan
│   ├── config.py           # Settings (env-var driven)
│   ├── inference.py        # Model loading & prediction engine
│   ├── models.py           # Pydantic request/response schemas
│   ├── metrics.py          # Operational metrics collector
│   └── monitoring.py       # HTML dashboard generator
├── src/
│   ├── data/               # Data collection & feature engineering
│   ├── eda/                # Exploratory data analysis
│   ├── preprocessing/      # Scaling, sequencing, splitting pipeline
│   ├── visualization/      # Plot helpers
│   └── model/              # LSTM builder, trainer, evaluator, persistence
├── models/                 # Saved model artifacts (tracked in git)
│   ├── saved_models/lstm_model.keras
│   ├── scalers/scaler.pkl
│   └── metadata.json
├── scripts/
│   └── test_deployment.py  # Smoke-test suite
├── phase1_eda.py
├── phase1_preprocessing.py
├── phase2_model_training.py
├── phase3_model_persistence.py
├── Dockerfile
├── docker-compose.yml
├── render.yaml
└── requirements.txt
```

---

## Prerequisites

- **Python 3.11** (or 3.10 for Render)
- **Docker & Docker Compose** (for containerised deployment)
- Internet access (data is fetched from Yahoo Finance during training)

---

## Step 1 — Train the Model

> Skip this step if the `models/` directory already contains trained artifacts (`lstm_model.keras`, `scaler.pkl`, `metadata.json`).

### 1.1 Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 1.2 Install dependencies

```bash
pip install -r requirements.txt
```

### 1.3 Run the training pipeline

Run the four phases in order from the project root:

```bash
# Phase 1a — Exploratory Data Analysis (downloads PETR4.SA data)
python phase1_eda.py

# Phase 1b — Data preprocessing (creates train/val/test splits + scaler)
python phase1_preprocessing.py

# Phase 2 — LSTM model training (100 epochs, early stopping)
python phase2_model_training.py

# Phase 3 — Save model artifacts to models/
python phase3_model_persistence.py
```

After Phase 3 the `models/` directory will contain:

```
models/
├── saved_models/lstm_model.keras
├── scalers/scaler.pkl
└── metadata.json
```

Training logs are written to `logs/`.

---

## Step 2 — Run the API Locally

```bash
# Make sure the virtual environment is active and model artifacts exist
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger UI.

### Using a `.env` file (optional)

Copy the example and adjust paths if needed:

```bash
cp .env.example .env
```

The defaults in `.env.example` work out of the box when running from the project root.

---

## Step 3 — Deploy with Docker

### 3.1 Build and start

```bash
docker compose up --build
```

The container exposes port `8000`. Docker Compose also configures:

- A health check that polls `GET /health` every 30 seconds
- Automatic restarts (`restart: unless-stopped`)
- A bind-mount of `./logs` so log files survive container restarts

### 3.2 Verify the container is healthy

```bash
docker compose ps
# STATUS column should show "healthy" after ~30 s
```

### 3.3 Stop

```bash
docker compose down
```

---

## Step 4 — Deploy to Render

A `render.yaml` file is included for one-click deployment on [Render](https://render.com).

### 4.1 Push your repository to GitHub (including the `models/` directory)

The `models/` directory is intentionally tracked in git so Render's free tier can access the artifacts without a paid persistent disk.

### 4.2 Create a new Web Service on Render

1. Go to [dashboard.render.com](https://dashboard.render.com) → **New → Web Service**
2. Connect your GitHub repository
3. Render will detect `render.yaml` automatically and pre-fill the settings:
   - **Runtime:** Python
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
   - **Health check path:** `/health`
4. Click **Create Web Service**

Render will build and deploy the service. The public URL will be shown in the dashboard once the health check passes.

### 4.3 Verify the deployment

```bash
python scripts/test_deployment.py --url https://your-service.onrender.com
```

---

## API Reference

All endpoints return JSON. The interactive docs are available at `GET /docs`.

### `GET /health`

Returns API liveness and model status.

```json
{
  "status": "ok",
  "model": "loaded",
  "timestamp": "2025-01-01T00:00:00Z"
}
```

### `GET /model-info`

Returns model version, training date, expected sequence length, and test-set accuracy metrics (MAE, RMSE, MAPE).

### `GET /metrics`

Returns live operational metrics: uptime, total predictions, error rate, and inference latency statistics.

### `GET /dashboard`

Serves an HTML monitoring dashboard that auto-refreshes every 5 minutes.

### `POST /predict`

Predict the next-day closing price for a single sequence.

**Request body:**

```json
{
  "stock_symbol": "PETR4.SA",
  "historical_prices": [38.5, 38.9, 39.1, ...]  // exactly 60 positive values
}
```

**Response:**

```json
{
  "predicted_price": 39.87,
  "confidence_interval": {
    "min": 38.94,
    "max": 40.80
  },
  "timestamp": "2025-01-01T12:00:00Z",
  "model_version": "v1.0"
}
```

**Validation rules:**

- `historical_prices` must contain **exactly 60** values
- All prices must be **positive** numbers (> 0)

### `POST /predict-batch`

Run multiple predictions in a single request.

**Request body:**

```json
{
  "requests": [
    { "stock_symbol": "PETR4.SA", "historical_prices": [...] },
    { "stock_symbol": "PETR4.SA", "historical_prices": [...] }
  ]
}
```

**Response:**

```json
{
  "predictions": [ ... ],
  "total": 2
}
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `models/saved_models/lstm_model.keras` | Path to the Keras model file |
| `SCALER_PATH` | `models/scalers/scaler.pkl` | Path to the fitted scaler |
| `METADATA_PATH` | `models/metadata.json` | Path to model metadata |
| `API_HOST` | `0.0.0.0` | Uvicorn bind host |
| `API_PORT` | `8000` | Uvicorn bind port |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `CORS_ORIGINS` | `*` | Comma-separated list of allowed CORS origins |
| `ERROR_RATE_ALERT_THRESHOLD` | `0.05` | Error rate fraction that triggers a critical log |
| `DRIFT_THRESHOLD` | `0.15` | RMSE increase fraction that triggers a drift warning |

---

## Smoke Testing

A deployment smoke-test suite is included in [scripts/test_deployment.py](scripts/test_deployment.py). It covers health check, model info, valid and invalid prediction payloads, batch predictions, latency, and docs availability.

```bash
# Against local instance
python scripts/test_deployment.py --url http://localhost:8000

# Against a remote deployment
python scripts/test_deployment.py --url https://your-service.onrender.com --timeout 60
```

Exit code is `0` if all tests pass, `1` otherwise.
