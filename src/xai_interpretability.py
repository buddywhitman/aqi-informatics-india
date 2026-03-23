import pandas as pd
import numpy as np
import shap
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# Configuration
PROCESSED_DATA_PATH = "data/processed/combined_data.csv"
MODEL_PATH = "models"
PLOTS_PATH = "plots/xai"

def main():
    if not os.path.exists(PLOTS_PATH):
        os.makedirs(PLOTS_PATH)
        
    df = pd.read_csv(PROCESSED_DATA_PATH)
    features = ['pm25', 'temp_max', 'temp_min', 'precipitation', 'wind_speed_max', 'humidity_max']
    
    for city in df['city'].unique():
        print(f"Generating SHAP explanations for {city}...")
        
        # Load model
        model_file = f"{MODEL_PATH}/hybrid_model_{city}.keras"
        if not os.path.exists(model_file):
            continue
            
        model = tf.keras.models.load_model(model_file)
        
        # Prepare background data for SHAP
        city_df = df[df['city'] == city][features].dropna()
        if len(city_df) < 50:
            continue
            
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(city_df)
        
        # We need to create sequences again to match model input
        window_size = 7
        X = []
        for i in range(len(scaled_data) - window_size):
            X.append(scaled_data[i:(i + window_size), :])
        X = np.array(X)
        
        # Explain the model's predictions using SHAP
        background = X[:50]
        test_sample = X[50:100]
        
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(test_sample)
        
        # Based on inspection, shap_values is a numpy array (samples, window, features, 1)
        # Squeeze out the last dimension
        shap_values = np.squeeze(shap_values)
        
        # Now shap_values shape: (samples, window_size, features)
        # We'll take the SHAP values for the most recent timestep (-1) 
        recent_shap_values = shap_values[:, -1, :]
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(recent_shap_values, test_sample[:, -1, :], feature_names=features, show=False)
        plt.title(f'SHAP Summary (Most Recent Timestep) - {city}')
        plt.tight_layout()
        plt.savefig(f"{PLOTS_PATH}/shap_summary_{city}.png")
        plt.close()
        
        print(f"SHAP plots saved for {city}")

if __name__ == "__main__":
    main()
