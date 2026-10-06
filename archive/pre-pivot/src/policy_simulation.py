import pandas as pd
import numpy as np
from econml.dml import LinearDML
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import os

# Configuration
DATA_PATH = "data/processed_hourly/combined_hourly_with_regimes.csv"
PLOTS_PATH = "plots/causal_v2"

def run_policy_simulation():
    if not os.path.exists(PLOTS_PATH): os.makedirs(PLOTS_PATH)
    
    df = pd.read_csv(DATA_PATH)
    
    # Policy Simulation 1: Traffic Emission Reduction (Proxy: NO2)
    # Target: PM2.5, Treatment: NO2, Confounders: Weather
    features = ['temperature_x', 'humidity', 'wind_speed_y', 'precipitation']
    
    cities = df['city'].unique()
    
    results = []
    
    for city in cities:
        print(f"Simulating Policy for {city}...")
        city_df = df[df['city'] == city].dropna(subset=['pm25', 'no2'] + features)
        
        if len(city_df) < 200: continue
        
        X = city_df[features].values
        T = city_df['no2'].values
        Y = city_df['pm25'].values
        
        # Double Machine Learning (DML)
        # We estimate the causal effect of NO2 on PM2.5
        est = LinearDML(model_y=LGBMRegressor(), model_t=LGBMRegressor(), discrete_treatment=False)
        est.fit(Y, T, X=X)
        
        # Calculate Average Treatment Effect (ATE)
        # This is the expected change in PM2.5 for a 1 unit change in NO2
        ate = est.ate(X)
        
        # Simulate -20% NO2
        avg_no2 = city_df['no2'].mean()
        reduction = 0.20 * avg_no2
        predicted_pm25_reduction = ate * reduction
        
        print(f"  ATE (dPM25/dNO2): {ate:.4f}")
        print(f"  Predicted PM2.5 reduction for -20% Traffic emissions: {predicted_pm25_reduction:.2f} ug/m3")
        
        results.append({
            'city': city,
            'ate': ate,
            'no2_mean': avg_no2,
            'pm25_reduction': predicted_pm25_reduction
        })
        
        # Plot Treatment Effects
        te = est.effect(X)
        plt.figure()
        plt.hist(te, bins=30)
        plt.title(f'Causal Effect Distribution (dPM25/dNO2) - {city}')
        plt.xlabel('Marginal Effect')
        plt.savefig(f"{PLOTS_PATH}/dml_effect_{city}.png")
        plt.close()

    # Save simulation results
    res_df = pd.DataFrame(results)
    res_df.to_csv("reports/policy_simulation_results.csv", index=False)
    print("\nPolicy simulation complete. Results saved to reports/policy_simulation_results.csv")

if __name__ == "__main__":
    run_policy_simulation()