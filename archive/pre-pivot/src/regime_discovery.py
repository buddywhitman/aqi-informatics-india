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

from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.utils import resample

def perform_bootstrap_stability(X, n_iter=5):
    """Assess cluster stability using Bootstrap resampling and Adjusted Rand Index."""
    print("\n--- Bootstrap Stability Analysis ---")
    scores = []
    for i in range(n_iter):
        X_resample = resample(X, random_state=i)
        gmm1 = GaussianMixture(n_components=5, random_state=42).fit(X)
        gmm2 = GaussianMixture(n_components=5, random_state=i).fit(X_resample)
        
        l1 = gmm1.predict(X)
        l2 = gmm2.predict(X)
        scores.append(adjusted_rand_score(l1, l2))
    
    avg_stability = np.mean(scores)
    print(f"Mean Bootstrap Adjusted Rand Index (Stability): {avg_stability:.4f}")
    return avg_stability

def calculate_persistence(city_df):
    """Calculate the average duration (in hours) a city stays in each regime."""
    persistence = {}
    for rid in range(5):
        # Create a series of booleans for being in this regime
        in_regime = (city_df['regime_id'] == rid).astype(int)
        # Identify blocks of consecutive matches
        blocks = (in_regime.diff() != 0).cumsum()
        regime_blocks = blocks[in_regime == 1]
        if not regime_blocks.empty:
            durations = regime_blocks.value_counts()
            persistence[rid] = durations.mean()
        else:
            persistence[rid] = 0
    return persistence

def perform_regime_discovery():
    if not os.path.exists(PLOTS_PATH): os.makedirs(PLOTS_PATH)
    
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. Feature Selection for Clustering
    potential_cols = ['pm25', 'pm10', 'no2', 'o3', 'co', 'so2', 'temperature_x', 'humidity', 'wind_speed_y', 'pressure']
    cluster_cols = [c for potential in potential_cols for c in df.columns if potential == c]
    
    cluster_df = df.dropna(subset=cluster_cols).copy()
    X = cluster_df[cluster_cols].values
    
    # 2. Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. PCA
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    
    # 4. Clustering (GMM)
    gmm = GaussianMixture(n_components=5, random_state=42)
    labels = gmm.fit_predict(X_pca)
    cluster_df['regime_id'] = labels
    
    # 5. SOTA Validation
    sil = silhouette_score(X_pca, labels, sample_size=2000)
    db = davies_bouldin_score(X_pca, labels)
    stability = perform_bootstrap_stability(X_pca)
    
    # 6. Characterize and Map Labels
    regime_stats = cluster_df.groupby('regime_id')[cluster_cols].mean()
    label_map = {rid: label_regime(regime_stats.loc[rid]) for rid in range(5)}
    cluster_df['regime_label'] = cluster_df['regime_id'].map(label_map)

    # 7. Persistence and Transitions
    persistence_results = []
    for city in cluster_df['city'].unique():
        print(f"\n--- Transition and Persistence for {city} ---")
        city_data = cluster_df[cluster_df['city'] == city].sort_values('timestamp')
        
        # Persistence
        p = calculate_persistence(city_data)
        for rid, dur in p.items():
            persistence_results.append({'city': city, 'regime': label_map[rid], 'avg_hours': dur})
        
        # Transitions
        transitions = city_data['regime_id'].values
        matrix = np.zeros((5, 5))
        for i in range(len(transitions)-1):
            matrix[transitions[i], transitions[i+1]] += 1
        
        row_sums = matrix.sum(axis=1)
        matrix_prob = np.divide(matrix, row_sums[:, np.newaxis], where=row_sums[:, np.newaxis]!=0)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix_prob, annot=True, cmap='Blues', fmt=".2f", 
                    xticklabels=[label_map[i] for i in range(5)],
                    yticklabels=[label_map[i] for i in range(5)])
        plt.title(f'SOTA Transition Matrix - {city}')
        plt.savefig(f"{PLOTS_PATH}/transitions_{city}_sota.png")
        plt.close()

    pd.DataFrame(persistence_results).to_csv("reports/regime_persistence_sota.csv", index=False)
    cluster_df.to_csv("data/processed_hourly/combined_hourly_with_regimes.csv", index=False)
    print("\nRegime analysis v3.2 (SOTA) complete.")

if __name__ == "__main__":
    perform_regime_discovery()