import pandas as pd
import numpy as np
import shap
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# Configuration
PROCESSED_DATA_PATH = "data/processed_hourly/combined_hourly_v3.csv"
MODEL_PATH = "models"
PLOTS_PATH = "plots/xai_exhaustive"

def identify_tipping_points(shap_values, features_data, feature_names, city):
    """Analyze SHAP dependence to find mathematical tipping points."""
    tipping_points = {}
    
    # 1. Wind Speed Tipping Point (Where SHAP goes from - to +)
    if 'wind_speed_y' in feature_names:
        idx = feature_names.index('wind_speed_y')
        ws = features_data[:, idx]
        sv = shap_values[:, idx]
        
        # Find the value where sv cross 0 (approximate threshold for stagnation)
        zero_crossings = ws[np.where(np.diff(np.sign(sv)))[0]]
        if len(zero_crossings) > 0:
            tipping_points['Wind Speed Stagnation Threshold'] = np.mean(zero_crossings)
            
    # 2. Humidity Threshold (Where risk accelerates)
    if 'humidity' in feature_names:
        idx = feature_names.index('humidity')
        h = features_data[:, idx]
        sv = shap_values[:, idx]
        # Look for inflection points or sharp increases
        # Using a simple gradient check
        grad = np.gradient(sv, h)
        tipping_points['Critical Humidity Window'] = h[np.argmax(grad)]

    return tipping_points

def main():
    if not os.path.exists(PLOTS_PATH): os.makedirs(PLOTS_PATH)
    df = pd.read_csv(PROCESSED_DATA_PATH)
    features = ['pm25', 'temp_max', 'temp_min', 'precipitation', 'wind_speed_max', 'humidity_max']
    # Map to actual column names in v3 (if different)
    actual_cols = ['pm25', 'temperature_x', 'temperature_y', 'precipitation', 'wind_speed_y', 'humidity']
    
    all_tipping_points = []
    
    for city in df['city'].unique():
        print(f"Exhaustive XAI for {city}...")
        model_file = f"{MODEL_PATH}/hybrid_model_{city}.keras"
        if not os.path.exists(model_file): continue
        model = tf.keras.models.load_model(model_file)
        
        city_df = df[df['city'] == city][actual_cols].dropna()
        if len(city_df) < 200: continue
        
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(city_df)
        window_size = 7
        X = []
        for i in range(len(scaled_data) - window_size):
            X.append(scaled_data[i:(i + window_size), :])
        X = np.array(X)
        
        background = X[:100]
        test_sample = X[100:200]
        
        explainer = shap.GradientExplainer(model, background)
        # GradientExplainer for Keras expects list of inputs if multiple inputs, but here it's 1
        shap_values = explainer.shap_values(test_sample)
        
        # Squeeze out the extra dimension (samples, window, features, 1) -> (samples, window, features)
        shap_values = np.squeeze(shap_values)
        
        recent_sv = shap_values[:, -1, :]
        recent_data = test_sample[:, -1, :]
        
        original_units = scaler.inverse_transform(recent_data)
        tp = identify_tipping_points(recent_sv, original_units, features, city)
        tp['city'] = city
        all_tipping_points.append(tp)
        
        # Plot every possible dependence plot
        for i, feat in enumerate(features):
            plt.figure(figsize=(8, 5))
            # Get original units for plotting
            shap.dependence_plot(i, recent_sv, original_units, feature_names=features, show=False)
            plt.title(f'SHAP Dependence: {feat} in {city}')
            plt.savefig(f"{PLOTS_PATH}/{city}_{feat}_dependence.png")
            plt.close()

    pd.DataFrame(all_tipping_points).to_csv("reports/tipping_points_analysis.csv", index=False)
    print("Exhaustive XAI completed.")

if __name__ == "__main__":
    main()
