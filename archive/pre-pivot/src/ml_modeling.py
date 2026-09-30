import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
PROCESSED_DATA_PATH = "data/processed/combined_data.csv"
MODEL_PATH = "models"

def prepare_data(df, city, target='pm25', lags=3):
    """Prepare data for modeling for a specific city."""
    city_df = df[df['city'] == city].copy()
    city_df.sort_values('date', inplace=True)
    
    # Drop rows where target is NaN
    city_df.dropna(subset=[target], inplace=True)
    
    # Feature Engineering: Lagging features
    for i in range(1, lags + 1):
        city_df[f'{target}_lag_{i}'] = city_df[target].shift(i)
        
    # Features (excluding date, city, season, and other pollutants)
    # We want to use weather and historical pollution
    features = ['temp_max', 'temp_min', 'precipitation', 'wind_speed_max', 'humidity_max', 
                'month', 'day_of_week', 'is_weekend'] + [f'{target}_lag_{i}' for i in range(1, lags + 1)]
    
    # Drop rows with NaN from lags and features
    before_drop = len(city_df)
    city_df.dropna(subset=features + [target], inplace=True)
    after_drop = len(city_df)
    print(f"City: {city}, Rows before drop: {before_drop}, After drop: {after_drop}")
    
    X = city_df[features]
    y = city_df[target]
    
    return X, y

def train_rf_model(X, y, city):
    """Train a Random Forest model."""
    print(f"Training Random Forest for {city}...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MSE: {mse:.2f}, R2: {r2:.2f}")
    
    # Feature Importance
    plt.figure()
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title(f'Feature Importance - Random Forest ({city})')
    plt.savefig(f"plots/feature_importance_{city}.png")
    plt.close()
    
    return model

def main():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
        
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    for city in df['city'].unique():
        X, y = prepare_data(df, city)
        if len(X) > 50: # Ensure enough data points
            train_rf_model(X, y, city)
        else:
            print(f"Not enough data for {city}")

if __name__ == "__main__":
    main()
