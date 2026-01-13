# Racing Podium Prediction

Predict the top 3 finishers (podium) in racing events using machine learning.

## Live Demo

🏎️ **Dashboard**: https://danthepanman-csfloat-skin-price-valu-streamlit-dashboard-rk6p0h.streamlit.app/

🔗 **API**: https://f1-race-podium-predictor.onrender.com

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

### Run API (Docker)

```bash
# Build Docker image
docker build -t podium-prediction .

# Run container
docker run -it --rm -p 9696:9696 podium-prediction

# API: http://localhost:9696
```

### Run Streamlit Dashboard

```bash
# Install dependencies
pip install streamlit pandas requests

# Run dashboard
streamlit run streamlit_dashboard.py

# Dashboard: http://localhost:8501
```

## API Example

```python
import requests

# Example driver data for prediction
driver_data = {
    'grid': 5,                          # Starting grid position
    'driverRef': 'leclerc',            # Driver reference ID
    'constructorRef': 'ferrari',       # Constructor/team reference
    'year': 2024,                      # Race year
    'round': 10,                       # Race round number
    'podium_rate_last_year': 0.45,    # Podium rate from previous season
    'podium_rate_curr_year': 0.50,    # Current season podium rate
    'podium_rate_all_time': 0.38      # Career podium rate
}

# Make prediction request
url = 'http://localhost:9696/predict'
response = requests.post(url, json=driver_data)
result = response.json()

print(f"Podium Probability: {result['podium_probability']:.2%}")
print(f"Prediction: {result['podium_prediction']}")
```

## Deployment

-   Render / Railway / Fly.io for FastAPI
-   Streamlit for dashboard
