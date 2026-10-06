import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import os
import matplotlib.pyplot as plt

# Configuration
DATA_PATH = "data/processed_hourly/combined_hourly_with_regimes.csv"
REPORTS_PATH = "reports/sota_benchmarks"

def train_stacking_ensemble(X_train, y_train, X_test):
    """
    SOTA Stacking: Layer 1 (Diverse Regressors) -> Layer 2 (Meta-Learner)
    """
    # Base Models
    models = [
        LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
        CatBoostRegressor(n_estimators=100, random_state=42, verbose=0),
        RandomForestRegressor(n_estimators=50, random_state=42)
    ]
    
    # Generate Meta-features
    train_meta = np.zeros((X_train.shape[0], len(models)))
    test_meta = np.zeros((X_test.shape[0], len(models)))
    
    for i, model in enumerate(models):
        model.fit(X_train, y_train)
        train_meta[:, i] = model.predict(X_train)
        test_meta[:, i] = model.predict(X_test)
        
    # Meta-Learner (Ridge to prevent overfitting)
    meta_model = Ridge()
    meta_model.fit(train_meta, y_train)
    
    return meta_model.predict(test_meta)

def benchmark_sota_strategies():
    if not os.path.exists(REPORTS_PATH): os.makedirs(REPORTS_PATH)
    
    df = pd.read_csv(DATA_PATH)
    # Filter features for high-resolution prediction
    features = [c for c in df.columns if '_lag_' in c or '_roll_' in c] + ['temperature_x', 'humidity', 'wind_speed_y', 'pressure']
    target = 'pm25'
    
    results = []
    
    for city in df['city'].unique():
        print(f"SOTA Benchmarking for {city}...")
        city_df = df[df['city'] == city].dropna(subset=[target] + features)
        if len(city_df) < 500: continue
        
        X = city_df[features]
        y = city_df[target]
        
        tscv = TimeSeriesSplit(n_splits=3)
        stacking_mses, stacking_r2s = [], []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            y_pred_stack = train_stacking_ensemble(X_train, y_train, X_test)
            
            stacking_mses.append(mean_squared_error(y_test, y_pred_stack))
            stacking_r2s.append(r2_score(y_test, y_pred_stack))
            
        results.append({
            'city': city,
            'model': 'SOTA_Stacking_Ensemble',
            'rmse': np.sqrt(np.mean(stacking_mses)),
            'r2': np.mean(stacking_r2s)
        })
        print(f"  Ensemble: RMSE={np.sqrt(np.mean(stacking_mses)):.4f}, R2={np.mean(stacking_r2s):.4f}")

    pd.DataFrame(results).to_csv(f"{REPORTS_PATH}/ensemble_results.csv", index=False)

if __name__ == "__main__":
    benchmark_sota_strategies()
