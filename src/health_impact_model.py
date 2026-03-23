import pandas as pd
import numpy as np

# Configuration
PROCESSED_DATA_PATH = "data/processed_hourly/combined_hourly.csv"

def calculate_health_impact(df, city):
    """
    Calculate excess mortality attributable to PM2.5 using WHO CRF.
    Formula: delta_M = Y0 * (1 - e^(-beta * delta_X)) * Pop
    """
    city_df = df[df['city'] == city].copy()
    
    # WHO 2021 Air Quality Guideline for PM2.5 (24-hour mean) is 15 µg/m³
    # For hourly, we'll use a conservative baseline of 15 as well.
    baseline_pm25 = 15.0
    
    # Relative Risk (RR) for All-Cause Mortality per 10 µg/m³ increase in PM2.5
    # Standard literature value (e.g., GBD): ~1.06 to 1.08. We use 1.06.
    RR_per_10 = 1.06
    beta = np.log(RR_per_10) / 10.0
    
    # Rough population estimates for the cities (in millions)
    populations = {
        "Delhi": 30.0, "Mumbai": 21.0, "Bengaluru": 13.0, 
        "Kolkata": 15.0, "Chennai": 11.0, "Hyderabad": 10.0, "Ahmedabad": 8.0
    }
    pop = populations.get(city, 10.0) * 1_000_000
    
    # Baseline daily mortality rate (India avg ~ 7 per 1000 per year)
    # Hourly baseline mortality rate:
    Y0_hourly = (7 / 1000) / (365 * 24)
    
    # Calculate excess PM2.5
    city_df['delta_x'] = city_df['pm25'] - baseline_pm25
    city_df.loc[city_df['delta_x'] < 0, 'delta_x'] = 0 # No health benefit below WHO guideline
    
    # Calculate attributable mortality for each hour
    city_df['excess_deaths'] = Y0_hourly * (1 - np.exp(-beta * city_df['delta_x'])) * pop
    
    total_excess_deaths = city_df['excess_deaths'].sum()
    print(f"{city}: Estimated Total Excess Deaths over period: {int(total_excess_deaths)}")
    
    # Calculate deaths during 'Super-Spreader' anomaly hours if available
    # For simplicity, we just look at the top 1% of polluted hours
    threshold = city_df['pm25'].quantile(0.99)
    peak_hours = city_df[city_df['pm25'] >= threshold]
    peak_deaths = peak_hours['excess_deaths'].sum()
    print(f"  Deaths during top 1% peak hours: {int(peak_deaths)} ({(peak_deaths/total_excess_deaths)*100:.1f}%)")
    
    return city_df

def main():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    for city in df['city'].unique():
        calculate_health_impact(df, city)

if __name__ == "__main__":
    main()
