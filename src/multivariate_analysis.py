import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
PROCESSED_DATA_PATH = "data/processed_hourly/combined_hourly.csv"
PLOTS_PATH = "plots/v2_hourly"

def main():
    if not os.path.exists(PLOTS_PATH):
        os.makedirs(PLOTS_PATH)
        
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. Diurnal Cycle (Hour of Day vs PM2.5)
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='hour', y='pm25', hue='city', errorbar='sd')
    plt.title('Average Diurnal Cycle of PM2.5 (High Resolution)')
    plt.ylabel('PM2.5 (µg/m³)')
    plt.xlabel('Hour of Day')
    plt.xticks(range(0, 24))
    plt.savefig(f"{PLOTS_PATH}/diurnal_cycle_pm25.png")
    plt.close()
    
    # 2. NOx-O3 Titration Analysis (Multi-pollutant)
    # Check if o3 and no2 exist
    if 'no2' in df.columns and 'o3' in df.columns:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df, x='no2', y='o3', hue='city', alpha=0.3)
        plt.title('NO2 vs O3 Interaction (Urban Titration)')
        plt.savefig(f"{PLOTS_PATH}/no2_o3_titration.png")
        plt.close()
        
    # 3. Wind Rose Proxy (Wind Speed vs PM2.5)
    plt.figure(figsize=(10, 6))
    sns.jointplot(data=df, x='wind_speed', y='pm25', kind='hex', cmap='viridis')
    plt.savefig(f"{PLOTS_PATH}/wind_pm25_joint.png")
    plt.close()
    
    print(f"v2.0 Hourly Analysis plots saved to {PLOTS_PATH}/")

if __name__ == "__main__":
    main()
