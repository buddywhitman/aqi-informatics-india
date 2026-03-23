import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
PROCESSED_DATA_PATH = "data/processed_hourly/combined_hourly.csv"
PLOTS_PATH = "plots/high_impact"

def main():
    if not os.path.exists(PLOTS_PATH):
        os.makedirs(PLOTS_PATH)
        
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    # 1. Multi-City PM2.5 Heatmap (Month vs City)
    plt.figure(figsize=(12, 8))
    pivot_df = df.groupby(['city', 'month'])['pm25'].mean().unstack()
    sns.heatmap(pivot_df, annot=True, cmap="YlOrRd", fmt=".1f")
    plt.title("Figure 1: Seasonal Toxicity Gradient Across Indian Metropolises (Monthly Mean PM2.5)")
    plt.savefig(f"{PLOTS_PATH}/city_month_heatmap.png")
    plt.close()
    
    # 2. Health Risk Gradient (PM2.5 vs Estimated Deaths)
    # Using Kolkata as an example for the "Super-Spreader" story
    plt.figure(figsize=(10, 6))
    k_df = df[df['city'] == 'Kolkata'].copy()
    # Mocking deaths for plot if not already in csv (from previous model)
    baseline = 15.0
    beta = np.log(1.06) / 10.0
    k_df['excess_risk'] = 1 - np.exp(-beta * (k_df['pm25'] - baseline).clip(lower=0))
    
    sns.scatterplot(data=k_df, x='pm25', y='excess_risk', alpha=0.5, hue='hour', palette='viridis')
    plt.axvline(300, color='red', linestyle='--', label='Super-Spreader Threshold')
    plt.title("Figure 2: Concentration-Response Gradient and Super-Spreader Risk (Kolkata)")
    plt.ylabel("Relative Risk of Mortality")
    plt.xlabel("PM2.5 Concentration (ug/m3)")
    plt.legend()
    plt.savefig(f"{PLOTS_PATH}/health_risk_gradient.png")
    plt.close()
    
    print(f"High-impact plots generated in {PLOTS_PATH}")

if __name__ == "__main__":
    main()
