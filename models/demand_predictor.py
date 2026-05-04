"""
Demand Prediction Models for Platelet Inventory

Trains 3 models:
1. SARIMA - Seasonal ARIMA for time series
2. XGBoost - Gradient boosting with lag features
3. SMA - Simple Moving Average (baseline)

Intentionally imperfect (~75-85% accuracy) to simulate real-world uncertainty.
Models systematically under-predict spikes (realistic limitation).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import pickle
import os

warnings.filterwarnings('ignore')

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'platelet_demand_hamilton_medium_hospital.csv')
MODEL_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'trained_models')


def load_and_prepare_data():
    """Load data and create features for ML models."""
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Create lag features
    for lag in [1, 2, 3, 7, 14]:
        df[f'lag_{lag}'] = df['units_demanded'].shift(lag)
    
    # Rolling statistics
    df['rolling_mean_7'] = df['units_demanded'].shift(1).rolling(7).mean()
    df['rolling_std_7'] = df['units_demanded'].shift(1).rolling(7).std()
    df['rolling_mean_14'] = df['units_demanded'].shift(1).rolling(14).mean()
    
    # Drop NaN rows created by lags
    df = df.dropna().reset_index(drop=True)
    
    return df


def train_sma_model(train_df, window=7):
    """Simple Moving Average - baseline model."""
    return {'window': window, 'last_values': train_df['units_demanded'].tail(window).tolist()}


def predict_sma(model, n_days, history):
    """Predict using SMA."""
    predictions = []
    values = list(history)
    window = model['window']
    
    for _ in range(n_days):
        pred = np.mean(values[-window:])
        predictions.append(pred)
        values.append(pred)
    
    return np.array(predictions)


def train_xgboost_model(train_df):
    """Train XGBoost with lag features."""
    feature_cols = ['day_of_week', 'month', 'lag_1', 'lag_2', 'lag_3', 'lag_7', 
                    'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14']
    
    # Add binary features
    train_df = train_df.copy()
    train_df['is_friday'] = (train_df['day_of_week'] == 4).astype(int)
    feature_cols.append('is_friday')
    
    X = train_df[feature_cols]
    y = train_df['units_demanded']
    
    # Intentionally use suboptimal parameters for realistic imperfection
    model = xgb.XGBRegressor(
        n_estimators=50,  # Lower than optimal
        max_depth=3,      # Shallow trees
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    
    model.fit(X, y)
    return model, feature_cols


def train_sarima_model(train_series):
    """Train SARIMA model with weekly seasonality."""
    # SARIMA(1,0,1)(1,0,1,7) - simple seasonal model
    model = SARIMAX(
        train_series,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fitted = model.fit(disp=False, maxiter=100)
    return fitted


def add_prediction_noise(predictions, noise_factor=0.08, bias_factor=-0.02):
    """
    Add realistic noise to predictions.
    - Slight negative bias (under-prediction of spikes)
    - Random noise for variability
    Target: MAPE ~35-40% (accuracy ~60-65%)
    """
    noise = np.random.normal(bias_factor, noise_factor, len(predictions))
    noisy_preds = predictions * (1 + noise)
    return np.maximum(0, noisy_preds)  # Ensure non-negative


def evaluate_model(y_true, y_pred, model_name):
    """Calculate evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    # Calculate under-prediction rate (important for JIT analysis)
    under_predictions = np.sum(y_pred < y_true)
    under_pred_rate = under_predictions / len(y_true) * 100
    
    return {
        'model': model_name,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'under_prediction_rate': under_pred_rate
    }


def train_all_models(df, train_ratio=0.8):
    """Train all models and return results."""
    
    train_size = int(len(df) * train_ratio)
    train_df = df[:train_size].copy()
    test_df = df[train_size:].copy()
    
    print(f"Training data: {len(train_df)} days")
    print(f"Test data: {len(test_df)} days")
    print()
    
    results = []
    predictions_dict = {}
    
    # 1. Simple Moving Average
    print("Training SMA model...")
    sma_model = train_sma_model(train_df, window=7)
    sma_preds = predict_sma(sma_model, len(test_df), train_df['units_demanded'].tolist())
    sma_preds = add_prediction_noise(sma_preds, noise_factor=0.18)
    results.append(evaluate_model(test_df['units_demanded'].values, sma_preds, 'SMA'))
    predictions_dict['SMA'] = sma_preds
    
    # 2. XGBoost
    print("Training XGBoost model...")
    xgb_model, feature_cols = train_xgboost_model(train_df)
    
    # Prepare test features
    test_df['is_friday'] = (test_df['day_of_week'] == 4).astype(int)
    X_test = test_df[feature_cols]
    xgb_preds = xgb_model.predict(X_test)
    xgb_preds = add_prediction_noise(xgb_preds, noise_factor=0.12, bias_factor=-0.03)
    results.append(evaluate_model(test_df['units_demanded'].values, xgb_preds, 'XGBoost'))
    predictions_dict['XGBoost'] = xgb_preds
    
    # 3. SARIMA
    print("Training SARIMA model...")
    try:
        sarima_model = train_sarima_model(train_df['units_demanded'])
        sarima_preds = sarima_model.forecast(steps=len(test_df))
        sarima_preds = add_prediction_noise(sarima_preds.values, noise_factor=0.10, bias_factor=-0.02)
        results.append(evaluate_model(test_df['units_demanded'].values, sarima_preds, 'SARIMA'))
        predictions_dict['SARIMA'] = sarima_preds
    except Exception as e:
        print(f"SARIMA failed: {e}")
        results.append({'model': 'SARIMA', 'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'under_prediction_rate': np.nan})
    
    return results, predictions_dict, test_df, train_df, xgb_model, feature_cols


def select_best_model(results):
    """Select best model based on MAPE (lower is better)."""
    valid_results = [r for r in results if not np.isnan(r['mape'])]
    best = min(valid_results, key=lambda x: x['mape'])
    return best['model']


def main():
    """Main training pipeline."""
    print("="*60)
    print("PLATELET DEMAND PREDICTION MODEL TRAINING")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df = load_and_prepare_data()
    print(f"Total records: {len(df)}")
    
    # Train models
    print("\n" + "-"*40)
    results, predictions_dict, test_df, train_df, xgb_model, feature_cols = train_all_models(df)
    
    # Display results
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"\n{'Model':<12} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'Under-Pred%':>12}")
    print("-"*50)
    for r in results:
        if not np.isnan(r['mape']):
            print(f"{r['model']:<12} {r['mae']:>8.2f} {r['rmse']:>8.2f} {r['mape']:>7.1f}% {r['under_prediction_rate']:>11.1f}%")
    
    # Select best model
    best_model = select_best_model(results)
    print(f"\n✓ Best Model: {best_model}")
    
    # Save models and predictions
    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
    
    # Save XGBoost model (most practical for simulation)
    with open(os.path.join(MODEL_OUTPUT_PATH, 'xgboost_model.pkl'), 'wb') as f:
        pickle.dump({'model': xgb_model, 'features': feature_cols}, f)
    
    # Save predictions for simulation
    predictions_df = pd.DataFrame({
        'date': test_df['date'].values,
        'actual': test_df['units_demanded'].values,
        'pred_sma': predictions_dict['SMA'],
        'pred_xgboost': predictions_dict['XGBoost'],
        'pred_sarima': predictions_dict.get('SARIMA', np.nan)
    })
    predictions_df.to_csv(os.path.join(MODEL_OUTPUT_PATH, 'predictions.csv'), index=False)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(MODEL_OUTPUT_PATH, 'model_results.csv'), index=False)
    
    print(f"\n💾 Models and predictions saved to: {MODEL_OUTPUT_PATH}")
    
    return results, predictions_dict, test_df


if __name__ == "__main__":
    results, predictions, test_df = main()
