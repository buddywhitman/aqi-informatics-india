import pandas as pd
import numpy as np
from causalml.inference.meta import BaseXRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import os

# Configuration
PROCESSED_DATA_PATH = "data/processed/combined_data.csv"
PLOTS_PATH = "plots/causal"

def evaluate_policy_impact(df, city, treatment_period=('2023-11-01', '2023-11-15')):
    """Estimate the effect of a specific period (e.g. Diwali peaks) vs counterfactual."""
    print(f"Evaluating causal impact for {city}...")
    
    city_df = df[df['city'] == city].copy()
    city_df['date'] = pd.to_datetime(city_df['date'])
    city_df.sort_values('date', inplace=True)
    
    # Define treatment: 1 if date is in treatment_period, 0 otherwise
    start_t, end_t = pd.to_datetime(treatment_period[0], utc=True), pd.to_datetime(treatment_period[1], utc=True)
    city_df['treatment'] = ((city_df['date'] >= start_t) & (city_df['date'] <= end_t)).astype(int)
    
    # Features for the learner (Weather + time)
    features = ['temp_max', 'temp_min', 'precipitation', 'wind_speed_max', 'humidity_max', 'month', 'day_of_week']
    
    # Drop NaNs
    city_df = city_df.dropna(subset=features + ['pm25'])
    
    if len(city_df[city_df['treatment'] == 1]) < 5:
        print(f"Not enough treatment samples for {city}")
        return
        
    X = city_df[features].values
    y = city_df['pm25'].values
    treatment = city_df['treatment'].values
    
    # Use BaseXRegressor to estimate Average Treatment Effect (ATE)
    learner = BaseXRegressor(learner=XGBRegressor())
    # Estimate the individual treatment effect (ITE)
    ite = learner.fit_predict(X, treatment, y)
    
    # Calculate Average Treatment Effect (ATE)
    ate = np.mean(ite)
    
    print(f"{city} ATE: {ate:.2f} µg/m³")
    
    # Plot ITE distribution
    plt.figure()
    plt.hist(ite, bins=20)
    plt.title(f'Distribution of Treatment Effects (ITE) - {city}\nPeriod: {treatment_period}')
    plt.xlabel('Estimated Impact on PM2.5 (µg/m³)')
    plt.savefig(f"{PLOTS_PATH}/causal_impact_{city}.png")
    plt.close()

def main():
    if not os.path.exists(PLOTS_PATH):
        os.makedirs(PLOTS_PATH)
        
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    # Evaluate November 2025 impact (Peak Smog Season)
    for city in df['city'].unique():
        evaluate_policy_impact(df, city, treatment_period=('2025-11-01', '2025-11-30'))

if __name__ == "__main__":
    main()
