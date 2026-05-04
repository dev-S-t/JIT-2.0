# Project Defense & Technical Evidence Guide

This document is your definitive "cheat sheet" and technical evidence reference for presenting the Blood Bank Management System (BBMS) research. If evaluators ask how specific results, models, or datasets were generated, point directly to these sections of the codebase.

---

## Question 1: "How did you generate your dataset? Why didn't you use real hospital data?"

**Answer:** Due to HIPAA/patient privacy policies and strict institutional data confidentiality, raw hospital blood bank transaction logs were unavailable. Instead, we programmatically generated a synthetic dataset specifically engineered to mathematically mimic real-world hospital demand topologies. 

**Code Evidence (`data_generation/generate_platelet_data.py`):**
Our algorithm models the base demand along a Negative Binomial distribution (standard for clinical count data). We explicitly hardcoded real-world healthcare cycles, including:
1. **Weekend Tropes vs. Weekday Surgeries:** (Friday spikes for prophylactic transfusions, drops on Sat/Sun).
2. **Holiday Rebounds:** Simulating post-holiday elective surgery catch-ups.
3. **Random Trauma Spikes:** 5% probability of a 30-50% mass casualty/emergency trauma spike.

```python
# From generic_platelet_data.py
# Temporal multipliers based on medical research data
WEEKDAY_MULTIPLIERS = {
    0: 1.10,  # Monday - slightly elevated (catch-up)
    1: 1.15,  # Tuesday - spike (post-weekend surgeries)
    2: 1.05,  # Wednesday - normal
    3: 1.05,  # Thursday - normal
    4: 1.20,  # Friday - highest (prophylactic transfusions)
    5: 0.70,  # Saturday - low
    6: 0.70,  # Sunday - low
}

# Example of Stochastic Trauma Event Injection
trauma_event = random.random() < 0.05
trauma_mult = random.uniform(1.30, 1.50) if trauma_event else 1.0
final_demand = base_demand * weekday_mult * month_mult * holiday_mult * rebound_mult * trauma_mult
```

---

## Question 2: "How did you train and save the predictive models? How did you get the results for Table I?"

**Answer:** We established an 80/20 train/test split. We trained three baseline architectural models: Simple Moving Average (SMA), Seasonal ARIMA (SARIMA), and XGBoost (Gradient Boosting). For XGBoost, we engineered lag features (demand from 1, 2, 3, 7, and 14 days prior) and rolling averages. After training, the best model and the raw predictive limits were exported as a finalized `predictions.csv` and serialized as an `xgboost_model.pkl` file so the simulation engine could ingest it without re-training.

**Code Evidence (`models/demand_predictor.py`):**
```python
def train_xgboost_model(train_df):
    """Train XGBoost with lag features."""
    feature_cols = ['day_of_week', 'month', 'lag_1', 'lag_2', 'lag_3', 'lag_7', 
                    'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14', 'is_friday']
    
    X = train_df[feature_cols]
    y = train_df['units_demanded']
    
    # Model configuration
    model = xgb.XGBRegressor(
        n_estimators=50, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=42
    )
    model.fit(X, y)
    return model, feature_cols

# Code saving the model to disk for later use
with open(os.path.join('_OUTPUT_DIR_', 'xgboost_model.pkl'), 'wb') as f:
    pickle.dump({'model': xgb_model, 'features': feature_cols}, f)
predictions_df.to_csv(os.path.join('_OUTPUT_DIR_', 'predictions.csv'), index=False)
```

The error metrics (MAE, RMSE, MAPE) for Table I were calculated directly on the 20% holdout test set using `scikit-learn` algorithms:
```python
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = mean_absolute_percentage_error(y_true, y_pred) * 100
```

---

## Question 3: "How were the simulation results generated? (Table II: 11.2% Wastage vs 2.5% Wastage)"

**Answer:** We wrote a deterministic object-oriented engine (`InventorySimulator`) to test out three distinct supply chain workflows (Traditional, JIT-Only, JIT+Micro) over the unseen test timeline. The simulator mimics the physical reality of a blood bank cooler, maintaining an active array of `PlateletUnit` objects that process via rigorous First-In-First-Out (FIFO) queuing logic.

**Code Evidence (`simulation/inventory_sim.py`):**
The underlying logic explicitly pops units out of the queue either when they expire or when they are demanded for a patient:
```python
def use_units(self, quantity: int, current_day: int) -> Tuple[int, int]:
    # Sort by expiry (oldest first - FIFO)
    self.inventory.sort(key=lambda u: u.expiry_day)
    
    available = len(self.inventory)
    units_to_use = min(int(quantity), available)
    shortage = max(0, int(quantity) - available)
    
    # Remove used units physically from the virtual queue
    self.inventory = self.inventory[units_to_use:]
    
    return units_to_use, shortage
```

---

## Question 4: "How does the 'Micro-Expiry' functionality physically work in the code?"

**Answer:** Our `simulate_jit_micro` algorithm tracks the tracking error (standard deviation) over a 7-day rolling window. If the algorithm detects an upcoming risk (e.g., the forecasted 2-day demand exceeds current valid inventory, combined with a high degree of uncertainty or upcoming weekends), it selectively identifies platelets within 2 days of initial expiry. It alters their state to "extended" and augments their expiration timestamp from the base 5 days to 7 days, thereby saving them from the daily `remove_expired` sweep cycle.

**Code Evidence (`simulation/inventory_sim.py`):**
```python
# Identifying units eligible to be safely extended
eligible = [u for u in self.inventory 
            if 0 < u.expiry_day - current_day <= days_before_expiry and not u.extended]

for unit in eligible:
    days_until_expiry = unit.expiry_day - current_day
    
    # Mathematically extending the shelf-life representation 
    unit.expiry_day = current_day + (self.extended_shelf_life - self.base_shelf_life + days_until_expiry)
    unit.extended = True
```
