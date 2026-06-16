import pandas as pd
import numpy as np
import cvxpy as cp
import os

# Configuration
DATA_PATH = "reports/exhaustive_policy_scenarios.csv"
OUTPUT_PATH = "reports/policy_optimization_results.csv"

def optimize_emissions_portfolio():
    """
    Formally optimize emission reduction strategies using CVXPY.
    Goal: Maximize economic activity (minimize reduction) while staying under WHO health targets.
    """
    if not os.path.exists(DATA_PATH):
        print("Required simulation data missing.")
        return

    df = pd.read_csv(DATA_PATH)
    
    # We want to find the optimal 'Throttling Factor' for each city-sector pair
    # to achieve a target PM2.5 reduction that brings the city closer to WHO (15 ug/m3).
    
    optimized_results = []
    
    for city in df['city'].unique():
        print(f"Optimizing Policy Portfolio for {city}...")
        city_data = df[df['city'] == city]
        
        # Marginal elasticities (ATEs) from our DML/CausalForest models
        # We'll use the 'ate_marginal' column which is dPM25 / dProxy
        elasticities = city_data['ate_marginal'].values
        scenarios = city_data['scenario'].values
        
        n = len(elasticities)
        if n == 0: continue
        
        # Decision Variable: Percentage reduction in each sector [0, 1]
        x = cp.Variable(n)
        
        # Objective: Minimize the total 'Economic Friction' (sum of reductions weighted by sector importance)
        # For now, we assume equal weights (1.0). In a real setting, industrial throttling is 'costlier'.
        objective = cp.Minimize(cp.sum(x))
        
        # Constraints
        # 1. Total PM2.5 reduction must be at least 20% of the current baseline to move toward WHO
        # (This is a simplified goal)
        current_pm25_avg = 100 # Placeholder baseline
        target_reduction = 20.0 
        
        # Estimated reduction = sum(reduction_percentage * baseline_proxy * marginal_effect)
        # Note: ATE is marginal change. We need the baseline proxy levels.
        # For this optimization, we use the pre-calculated 'pm25_reduction_ugm3' from the exhaustive script
        # which already accounts for the scenario's reduction percentage.
        # We scale it by our variable x / original_reduction_percent
        
        orig_reductions = []
        for s in scenarios:
            if '20%' in s or '0.20' in s: orig_reductions.append(0.20)
            elif '50%' in s or '0.50' in s: orig_reductions.append(0.50)
            elif '30%' in s or '0.30' in s: orig_reductions.append(0.30)
            elif '60%' in s or '0.60' in s: orig_reductions.append(0.60)
            else: orig_reductions.append(0.25)
            
        individual_potential = city_data['pm25_reduction_ugm3'].values
        
        constraints = [
            cp.sum(cp.multiply(x, individual_potential / np.array(orig_reductions))) >= target_reduction,
            x >= 0,
            x <= 0.80 # Cap individual sector reduction at 80% to maintain basic city function
        ]
        
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve()
            if x.value is not None:
                for i in range(n):
                    optimized_results.append({
                        'city': city,
                        'sector_scenario': scenarios[i],
                        'optimal_reduction_pct': x.value[i] * 100,
                        'marginal_contribution': (x.value[i] * individual_potential[i] / orig_reductions[i])
                    })
        except:
            print(f"  Optimization infeasible for {city} with current constraints.")

    res_df = pd.DataFrame(optimized_results)
    res_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Formal Policy Optimization complete. Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    optimize_emissions_portfolio()
