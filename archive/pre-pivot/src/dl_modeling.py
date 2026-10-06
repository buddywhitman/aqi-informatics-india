import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# Configuration
PROCESSED_DATA_PATH = "data/processed/combined_data.csv"
MODEL_PATH = "models"

def create_sequences(data, target, window_size=7):
    """Create sequences for LSTM."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size), :])
        y.append(target[i + window_size])
    return np.array(X), np.array(y)

def train_lstm_model(df, city, target_col='pm25', window_size=7):
    """Train an LSTM model for a specific city."""
    print(f"Training LSTM for {city}...")
    
    city_df = df[df['city'] == city].copy()
    city_df.sort_values('date', inplace=True)
    
    # Use weather and target
    features = ['pm25', 'temp_max', 'temp_min', 'precipitation', 'wind_speed_max', 'humidity_max']
    
    # Drop rows with NaNs in these features
    city_df = city_df[features].dropna()
    
    if len(city_df) < 100:
        print(f"Not enough data for {city} LSTM")
        return None
        
    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(city_df)
    
    # Target is PM2.5 (first column)
    X, y = create_sequences(scaled_data, scaled_data[:, 0], window_size)
    
    # Split into train/test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build Model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(window_size, len(features)), return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Train
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=0)
    
    # Evaluate
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE): {loss:.4f}")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Plot results
    plt.figure()
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(f'LSTM Prediction for PM2.5 - {city}')
    plt.legend()
    plt.savefig(f"plots/lstm_prediction_{city}.png")
    plt.close()
    
    return model

def main():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
        
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    for city in df['city'].unique():
        train_lstm_model(df, city)

if __name__ == "__main__":
    main()
