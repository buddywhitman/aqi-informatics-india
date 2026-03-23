import pandas as pd
import numpy as np
import os

# Configuration
PROCESSED_DATA_PATH = "data/processed/combined_data.csv"
PLOTS_PATH = "plots"

def generate_strategies(city, df):
    """Derive strategies based on data analysis."""
    print(f"Generating strategy for {city}...")
    
    city_df = df[df['city'] == city].copy()
    
    # Correlation analysis for specific insights
    corr = city_df[['pm25', 'temp_max', 'precipitation', 'wind_speed_max', 'humidity_max']].corr()['pm25']
    
    strategies = []
    
    # Insight 1: Wind Speed
    if corr['wind_speed_max'] < -0.3:
        strategies.append("- Strong negative correlation with wind speed: Implement 'Stagnant Air Alerts' when wind speed forecasts are low (< 10 km/h). Restrict heavy vehicle movement during these periods.")
        
    # Insight 2: Precipitation
    if corr['precipitation'] < -0.2:
        strategies.append("- Rain significantly washes away pollutants: Focus on artificial cloud seeding or water sprinkling in hotspots during dry winter months.")
        
    # Insight 3: Seasonality
    winter_pm = city_df[city_df['season'] == 'Winter']['pm25'].mean()
    summer_pm = city_df[city_df['season'] == 'Summer']['pm25'].mean()
    if winter_pm > summer_pm * 1.5:
        strategies.append(f"- Severe Winter Spikes (Winter mean: {winter_pm:.1f} vs Summer mean: {summer_pm:.1f}): Enforcement of 'Winter Action Plan' (GRAP) should start earlier in November.")
        
    # Insight 4: Humidity
    if corr['humidity_max'] > 0.3:
        strategies.append("- High humidity correlates with pollution: Secondary aerosol formation might be high. Target SO2 and NO2 emissions from nearby industries.")

    return strategies

def main():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    report = "# Urban Environmental Informatics: Strategy Report\n\n"
    
    for city in df['city'].unique():
        strategies = generate_strategies(city, df)
        report += f"## {city}\n"
        if strategies:
            for s in strategies:
                report += f"{s}\n"
        else:
            report += "No significant correlations found for automated strategy generation.\n"
        report += "\n"
        
    with open("strategy_report.md", "w") as f:
        f.write(report)
        
    print("Strategy report saved to strategy_report.md")

if __name__ == "__main__":
    main()
