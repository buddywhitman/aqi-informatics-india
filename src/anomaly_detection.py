import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import os

# Configuration
PROCESSED_DATA_PATH = "data/processed_hourly/combined_hourly.csv"
PLOTS_PATH = "plots/v2_hourly"

def detect_anomalies(df, city):
    """Identify anthropogenic 'Super-Spreader' events using Isolation Forest."""
    print(f"Detecting anomalies for {city}...")
    city_df = df[df['city'] == city].copy()
    
    # Features for anomaly detection: Pollutants + Weather
    # We want to find cases where pollution is high but weather is NOT typically conducive
    features = ['pm25', 'pm10', 'temperature_x', 'humidity', 'wind_speed']
    city_df = city_df.dropna(subset=features)
    
    if len(city_df) < 100: return
    
    model = IsolationForest(contamination=0.01, random_state=42) # 1% anomalies
    city_df['is_anomaly'] = model.fit_predict(city_df[features])
    
    anomalies = city_df[city_df['is_anomaly'] == -1]
    
    # Plot
    plt.figure(figsize=(15, 6))
    plt.plot(city_df['timestamp'], city_df['pm25'], label='Normal', color='blue', alpha=0.5)
    plt.scatter(anomalies['timestamp'], anomalies['pm25'], color='red', label='Anomaly (Super-Spreader)')
    plt.title(f'Anthropogenic Super-Spreader Events (Anomalies) - {city}')
    plt.legend()
    plt.savefig(f"{PLOTS_PATH}/anomalies_{city}.png")
    plt.close()
    
    print(f"Found {len(anomalies)} anomalies for {city}")

def main():
    if not os.path.exists(PLOTS_PATH):
        os.makedirs(PLOTS_PATH)
        
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    for city in df['city'].unique():
        detect_anomalies(df, city)

if __name__ == "__main__":
    main()
