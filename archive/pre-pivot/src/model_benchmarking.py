import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import os

# Configuration
DATA_PATH = "data/processed_hourly/combined_hourly_with_regimes.csv"

def benchmark_models():
    df = pd.read_csv(DATA_PATH)
    # Correcting feature names for merged v3 data
    features = [c for c in df.columns if '_lag_' in c or '_roll_' in c] + ['temperature_x', 'humidity', 'wind_speed_y']
    if 'regime_id' in df.columns: features.append('regime_id')
    
    target = 'pm25'
    
    cities = df['city'].unique()
    results = []
    
    for city in cities:
        print(f"Benchmarking for {city}...")
        city_df = df[df['city'] == city].dropna(subset=[target] + features)
        
        if len(city_df) < 500: continue
        
        X = city_df[features]
        y = city_df[target]
        
        # TimeSeries Split
        tscv = TimeSeriesSplit(n_splits=3)
        
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42),
            'LightGBM': LGBMRegressor(random_state=42, verbose=-1),
            'CatBoost': CatBoostRegressor(random_state=42, verbose=0, iterations=200)
        }
        
        for name, model in models.items():
            mses, r2s = [], []
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mses.append(mean_squared_error(y_test, y_pred))
                r2s.append(r2_score(y_test, y_pred))
            
            results.append({
                'city': city,
                'model': name,
                'rmse': np.sqrt(np.mean(mses)),
                'r2': np.mean(r2s)
            })
            print(f"  {name}: RMSE={np.sqrt(np.mean(mses)):.2f}, R2={np.mean(r2s):.2f}")

    res_df = pd.DataFrame(results)
    res_df.to_csv("reports/model_benchmarking_results.csv", index=False)
    print("\nBenchmarking complete. Results saved to reports/model_benchmarking_results.csv")

if __name__ == "__main__":
    benchmark_models()