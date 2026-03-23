import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
PROCESSED_DATA_PATH = "data/processed/combined_data.csv"
PLOTS_PATH = "plots"

def main():
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"File not found: {PROCESSED_DATA_PATH}")
        return

    df = pd.read_csv(PROCESSED_DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    # Set aesthetics
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (15, 8)
    
    # 1. PM2.5 Trends over Time
    plt.figure()
    sns.lineplot(data=df, x='date', y='pm25', hue='city', alpha=0.7)
    plt.title('Daily PM2.5 Trends in Indian Metropolises (2021-2026)')
    plt.ylabel('PM2.5 (µg/m³)')
    plt.xlabel('Date')
    plt.savefig(f"{PLOTS_PATH}/pm25_trends.png")
    plt.close()
    
    # 2. Seasonal Variation in PM2.5 (Boxplot)
    plt.figure()
    sns.boxplot(data=df, x='season', y='pm25', hue='city', 
                order=['Summer', 'Monsoon', 'Post-Monsoon', 'Winter'])
    plt.title('Seasonal PM2.5 Variation')
    plt.ylabel('PM2.5 (µg/m³)')
    plt.savefig(f"{PLOTS_PATH}/seasonal_pm25.png")
    plt.close()
    
    # 3. Correlation Heatmap (Pollutants vs Weather)
    # Select columns for correlation
    corr_cols = ['pm25', 'pm10', 'no2', 'o3', 'co', 'so2', 
                 'temp_max', 'temp_min', 'precipitation', 'wind_speed_max', 'humidity_max']
    
    for city in df['city'].unique():
        plt.figure(figsize=(12, 10))
        city_df = df[df['city'] == city]
        corr_matrix = city_df[corr_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(f'Correlation Matrix for {city}')
        plt.tight_layout()
        plt.savefig(f"{PLOTS_PATH}/correlation_{city}.png")
        plt.close()
        
    # 4. Distribution of Pollutants
    plt.figure()
    for city in df['city'].unique():
        sns.kdeplot(data=df[df['city'] == city], x='pm25', label=city, fill=True)
    plt.title('Distribution of PM2.5 Concentration')
    plt.xlabel('PM2.5 (µg/m³)')
    plt.legend()
    plt.savefig(f"{PLOTS_PATH}/pm25_distribution.png")
    plt.close()
    
    print(f"EDA plots saved to {PLOTS_PATH}/")

if __name__ == "__main__":
    main()
