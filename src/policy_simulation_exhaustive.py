import pandas as pd
import numpy as np
from econml.dml import LinearDML
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import os

# Configuration
DATA_PATH = "data/processed_hourly/combined_hourly_with_regimes.csv"
PLOTS_PATH = "plots/causal_exhaustive"

SCENARIOS = {
    "Baseline vehicular reduction": {"proxy": "no2", "reduction": 0.20},
    "Aggressive vehicular ban": {"proxy": "no2", "reduction": 0.50},
    "Industrial throttling": {"proxy": "so2", "reduction": 0.30},
    "Deep industrial shutdown": {"proxy": "so2", "reduction": 0.60},
    "Ammonia/Agricultural control": {"proxy": "no", "reduction": 0.30} # Proxy for NH3/NO sources
}

def simulate_scenarios():
    if not os.path.exists(PLOTS_PATH): os.makedirs(PLOTS_PATH)
    df = pd.read_csv(DATA_PATH)
    features = ['temperature_x', 'humidity', 'wind_speed_y', 'pressure']
    
    results = []
    
    for city in df['city'].unique():
        print(f"Running scenario analysis for {city}...")
        for name, params in SCENARIOS.items():
            proxy = params['proxy']
            if proxy not in df.columns: continue
            
            city_df = df[df['city'] == city].dropna(subset=['pm25', proxy] + features)
            if len(city_df) < 300: continue
            
            X = city_df[features].values
            T = city_df[proxy].values
            Y = city_df['pm25'].values
            
            # Use DML for causality
            est = LinearDML(model_y=LGBMRegressor(n_estimators=100), 
                           model_t=LGBMRegressor(n_estimators=100), 
                           discrete_treatment=False)
            est.fit(Y, T, X=X)
            
            ate = est.ate(X)
            avg_proxy = city_df[proxy].mean()
            reduction_val = params['reduction'] * avg_proxy
            pm25_reduction = ate * reduction_val
            
            # Health impact link (approximate VSL mapping)
            # VSL for India ~ $0.5M to $1M. Using 0.5M USD as conservative.
            # Pop estimated from previous model
            pop_map = {"Delhi": 30e6, "Mumbai": 21e6, "Bengaluru": 13e6, "Kolkata": 15e6}
            pop = pop_map.get(city, 10e6)
            
            # Attributable deaths avoided (very simplified CRF)
            # Assume 1 ug/m3 reduction saves ~200 lives per year in a megacity (GBD proxy)
            deaths_saved = abs(pm25_reduction) * (pop / 1e6) * 15 # Scaling factor
            econ_benefit_m_usd = (deaths_saved * 0.5) # In Million USD
            
            results.append({
                'city': city,
                'scenario': name,
                'ate_marginal': ate,
                'pm25_reduction_ugm3': pm25_reduction,
                'est_lives_saved_annual': int(deaths_saved),
                'econ_benefit_m_usd': econ_benefit_m_usd
            })
            
    res_df = pd.DataFrame(results)
    res_df.to_csv("reports/exhaustive_policy_scenarios.csv", index=False)
    print("Scenario simulation completed.")

if __name__ == "__main__":
    simulate_scenarios()
