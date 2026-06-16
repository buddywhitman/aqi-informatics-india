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

from sklearn.metrics import silhouette_score, davies_bouldin_score

def label_regime(row):
    """Assign qualitative labels based on pollutant profile."""
    # Logic: High PM but low wind = Stagnation
    # High NO2/CO = Traffic
    # High SO2 = Industrial
    if row['pm25'] > 100 and row['wind_speed_y'] < 10: return 'Stagnation-Driven'
    if row['no2'] > 50 or row.get('co', 0) > 1000: return 'Traffic-Dominated'
    if row['so2'] > 30: return 'Industrial-Bypass'
    if row['pm10'] > 150 and row['pm25'] < 50: return 'Dust-Event'
    return 'Low-Pollution/Clearance'

def perform_regime_discovery():
    if not os.path.exists(PLOTS_PATH): os.makedirs(PLOTS_PATH)
    
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. Feature Selection for Clustering
    # Dynamically select available columns
    potential_cols = ['pm25', 'pm10', 'no2', 'o3', 'co', 'so2', 'temperature_x', 'humidity', 'wind_speed_y', 'pressure']
    cluster_cols = [c for c in potential_cols if c in df.columns]
    
    # Drop rows with NaNs
    cluster_df = df.dropna(subset=cluster_cols).copy()
    X = cluster_df[cluster_cols].values
    
    # 2. Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. PCA (95% variance)
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    
    # 4. Clustering (GMM)
    gmm = GaussianMixture(n_components=5, random_state=42)
    labels = gmm.fit_predict(X_pca)
    cluster_df['regime_id'] = labels
    
    # 5. Validation Metrics
    sil = silhouette_score(X_pca, labels, sample_size=2000)
    db = davies_bouldin_score(X_pca, labels)
    print(f"\n--- Cluster Validation ---")
    print(f"Silhouette Score: {sil:.4f}")
    print(f"Davies-Bouldin Index: {db:.4f}")

    # 6. Characterize and Map Labels
    # Use mean-based mapping for consistent across-city labeling
    regime_stats = cluster_df.groupby('regime_id')[cluster_cols].mean()
    label_map = {}
    for rid in range(5):
        stats = regime_stats.loc[rid]
        label_map[rid] = label_regime(stats)
    
    cluster_df['regime_label'] = cluster_df['regime_id'].map(label_map)
    print("\n--- Regime Mapping ---")
    print(cluster_df.groupby(['regime_id', 'regime_label']).size())
    
    # 6. Markov Transition Analysis (Phase 3)
    for city in cluster_df['city'].unique():
        print(f"\n--- Transition Matrix for {city} ---")
        city_data = cluster_df[cluster_df['city'] == city].sort_values('timestamp')
        
        # Calculate transitions
        transitions = city_data['regime_id'].values
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