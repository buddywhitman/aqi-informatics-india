import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# Configuration
PROCESSED_DATA_PATH = "data/processed/combined_data.csv"
MODEL_PATH = "models"

def prepare_spatiotemporal_data(df, city, window_size=7):
    """Prepare data for CNN-LSTM."""
    city_df = df[df['city'] == city].copy()
    city_df.sort_values('date', inplace=True)
    
    features = ['pm25', 'temp_max', 'temp_min', 'precipitation', 'wind_speed_max', 'humidity_max']
    city_df = city_df[features].dropna()
    
    if len(city_df) < 100:
        return None, None, None
        
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(city_df)
    
    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:(i + window_size), :])
        y.append(scaled_data[i + window_size, 0]) # Target is PM2.5
        
    return np.array(X), np.array(y), scaler

def build_hybrid_model(input_shape):
    """Hybrid CNN-LSTM Architecture."""
    model = Sequential([
        # CNN for local feature extraction
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        # Repeat to match LSTM input if needed, but here we'll just transition
        # Alternatively, use TimeDistributed(Conv1D)
    ])
    
    # Correcting architecture to be truly hybrid temporal
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(50, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
        
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    for city in df['city'].unique():
        print(f"Training Hybrid CNN-LSTM for {city}...")
        X, y, scaler = prepare_spatiotemporal_data(df, city)
        
        if X is not None:
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            model = build_hybrid_model((X.shape[1], X.shape[2]))
            model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, verbose=0)
            
            loss = model.evaluate(X_test, y_test, verbose=0)
            print(f"{city} Hybrid Model Test Loss: {loss:.4f}")
            
            # Save model
            model.save(f"{MODEL_PATH}/hybrid_model_{city}.keras")
            
            # Save predictions plot
            y_pred = model.predict(X_test)
            plt.figure(figsize=(12, 6))
            plt.plot(y_test, label='Actual')
            plt.plot(y_pred, label='Hybrid Prediction')
            plt.title(f'CNN-LSTM Hybrid Prediction - {city}')
            plt.legend()
            plt.savefig(f"plots/hybrid_prediction_{city}.png")
            plt.close()

if __name__ == "__main__":
    main()
