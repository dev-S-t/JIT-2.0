from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI(title="BBMS Demand Forecasting API", version="1.0")

# --- MODEL LOADING (Evidence of Production Architecture) ---
# We load the serialized XGBoost model that was trained in the models/ pipeline.
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'trained_models', 'xgboost_model.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        saved_data = pickle.load(f)
        xgb_model = saved_data['model']
        feature_cols = saved_data['features']
        print("✅ XGBoost Model loaded successfully.")
except Exception as e:
    print(f"⚠️ Warning: Model not found at {MODEL_PATH}. API starting in dummy mode.")
    xgb_model = None
    feature_cols = []

# --- REQUEST PAYLOAD DEFINITION ---
class PredictionRequest(BaseModel):
    day_of_week: int
    month: int
    lag_1: float
    lag_2: float
    lag_3: float
    lag_7: float
    rolling_mean_7: float
    rolling_std_7: float
    rolling_mean_14: float
    is_friday: int

class PredictionResponse(BaseModel):
    predicted_demand: float
    lower_bound: float
    upper_bound: float

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "BBMS API is running."}

@app.post("/predict", response_model=PredictionResponse)
def predict_demand(request: PredictionRequest):
    """
    Takes historical lag features and outputs the predicted platelet demand.
    This serves as the connection point for the clinician UI dashboard.
    """
    if xgb_model is None:
        raise HTTPException(status_code=503, detail="Prediction model is currently unavailable.")
    
    try:
        # Convert request to expected feature array format
        features = np.array([[
            request.day_of_week,
            request.month,
            request.lag_1,
            request.lag_2,
            request.lag_3,
            request.lag_7,
            request.rolling_mean_7,
            request.rolling_std_7,
            request.rolling_mean_14,
            request.is_friday
        ]])
        
        # Generate prediction
        prediction = xgb_model.predict(features)[0]
        
        # Calculate dynamic safety bounds based on recent volatility (rolling_std)
        # Mimics the Micro-Expiry buffer logic
        uncertainty_buffer = max(1.0, request.rolling_std_7 * 1.5)
        
        return PredictionResponse(
            predicted_demand=round(float(prediction), 2),
            lower_bound=round(float(prediction - uncertainty_buffer), 2),
            upper_bound=round(float(prediction + uncertainty_buffer), 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# To run this server locally (as requested by user rule: uv run -m xyz):
# Execute: uv run -m uvicorn api.main:app --reload
