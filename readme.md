# Racing Podium Prediction

Predict the top 3 finishers (podium) in racing events using machine learning.

## Architecture

-   **FastAPI backend** - REST API for predictions
-   **Streamlit dashboard** - Interactive UI for visualization
-   **Docker** - Containerized deployment
-   **PaaS hosting** - Deploy on Render/Railway/Fly.io

## ML Approach

-   **Binary classification** to predict podium finish (top 3) vs non-podium
-   Train model on historical race data with driver features
-   Predict probability of podium finish for each driver
-   Select top 3 drivers with highest probabilities as predicted podium

## Models

-   Logistic Regression
-   Random Forest
-   Gradient Boosting/XGBoost

## Quick Start

```bash
# Run with Docker
docker-compose up

# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

## API Example

```python
# POST /predict
{
  "race_id": 123,
  "drivers": [
    {"driver_id": 1, "features": {...}},
    {"driver_id": 2, "features": {...}}
  ]
}

# Response
{
  "podium": [
    {"position": 1, "driver_id": 5, "probability": 0.763},
    {"position": 2, "driver_id": 12, "probability": 0.681},
    {"position": 3, "driver_id": 8, "probability": 0.547}
  ]
}
```

## Deployment

-   Render / Railway / Fly.io for FastAPI
-   Streamlit Cloud for dashboard
-   Environment variables for model config
