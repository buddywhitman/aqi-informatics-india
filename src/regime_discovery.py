import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
DATA_PATH = "data/processed_hourly/combined_hourly_v3.csv"
PLOTS_PATH = "plots/regimes"

def perform_regime_discovery():
    if not os.path.exists(PLOTS_PATH): os.makedirs(PLOTS_PATH)
    
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. Feature Selection for Clustering
    # Dynamically select available columns
    potential_cols = ['pm25', 'pm10', 'no2', 'o3', 'co', 'so2', 'temperature_x', 'humidity', 'wind_speed']
    cluster_cols = [c for col in potential_cols for c in df.columns if col == c]
    
    print(f"Clustering using features: {cluster_cols}")
    
    # Drop rows with NaNs in these essential columns for clustering
    cluster_df = df.dropna(subset=cluster_cols).copy()
    
    X = cluster_df[cluster_cols].values
    
    # 2. Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. PCA (95% variance)
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA reduced dimensions from {len(cluster_cols)} to {X_pca.shape[1]}")
    
    # 4. Clustering (GMM as it's more stable for regime labeling than HDBSCAN sometimes)
    # We'll use GMM with 5 components as suggested by framework
    gmm = GaussianMixture(n_components=5, random_state=42)
    labels = gmm.fit_predict(X_pca)
    
    cluster_df['regime'] = labels
    
    # 5. Characterize Regimes
    regime_means = cluster_df.groupby('regime')[cluster_cols].mean()
    print("\n--- Regime Characteristics ---")
    print(regime_means)
    
    # 6. Markov Transition Analysis (Phase 3)
    for city in cluster_df['city'].unique():
        print(f"\n--- Transition Matrix for {city} ---")
        city_data = cluster_df[cluster_df['city'] == city].sort_values('timestamp')
        
        # Calculate transitions
        transitions = city_data['regime'].values
        n_regimes = 5
        matrix = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(transitions)-1):
            matrix[transitions[i], transitions[i+1]] += 1
            
        # Normalize to probabilities
        row_sums = matrix.sum(axis=1)
        matrix_prob = np.divide(matrix, row_sums[:, np.newaxis], where=row_sums[:, np.newaxis]!=0)
        
        # Plot Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix_prob, annot=True, cmap='Blues', fmt=".2f")
        plt.title(f'Regime Transition Probabilities - {city}')
        plt.xlabel('To Regime')
        plt.ylabel('From Regime')
        plt.savefig(f"{PLOTS_PATH}/transitions_{city}.png")
        plt.close()

    # Save data with regimes
    cluster_df.to_csv("data/processed_hourly/combined_hourly_with_regimes.csv", index=False)
    print("\nRegime discovery and transition analysis complete.")

if __name__ == "__main__":
    perform_regime_discovery()