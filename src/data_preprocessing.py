import pandas as pd
import numpy as np
import re
import os

# Configuration
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

def parse_openaq_timestamp(ts_str):
    """Extract UTC timestamp from the complex OpenAQ string."""
    match = re.search(r"utc='(.*?)'", ts_str)
    if match:
        return pd.to_datetime(match.group(1))
    return pd.to_datetime(ts_str, errors='coerce')

def preprocess_pollution(df):
    """Clean and pivot pollution data."""
    # Parse timestamps
    df['timestamp'] = df['timestamp'].apply(parse_openaq_timestamp)
    # Ensure UTC
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    # Pivot
    df_pivoted = df.pivot_table(index='timestamp', columns='parameter', values='value', aggfunc='mean')
    # Resample to daily (if not already)
    df_resampled = df_pivoted.resample('D').mean()
    return df_resampled

def preprocess_weather(df):
    """Clean weather data."""
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df.set_index('date', inplace=True)
    return df

def combine_city_data(city):
    """Combine pollution and weather data for a specific city."""
    print(f"Combining data for {city}...")
    
    poll_file = f"{RAW_DATA_PATH}/{city}_pollution_raw.csv"
    weath_file = f"{RAW_DATA_PATH}/{city}_weather_raw.csv"
    
    if not os.path.exists(poll_file) or not os.path.exists(weath_file):
        print(f"Missing files for {city}")
        return None
        
    poll_df = pd.read_csv(poll_file)
    weath_df = pd.read_csv(weath_file)
    
    poll_clean = preprocess_pollution(poll_df)
    weath_clean = preprocess_weather(weath_df)
    
    # Merge on index (date)
    combined = pd.merge(poll_clean, weath_clean, left_index=True, right_index=True, how='outer')
    combined['city'] = city
    
    return combined

def main():
    cities = ["Delhi", "Mumbai", "Bengaluru"]
    all_combined = []
    
    for city in cities:
        city_df = combine_city_data(city)
        if city_df is not None:
            all_combined.append(city_df)
            
    if all_combined:
        final_df = pd.concat(all_combined)
        
        # The index is the date. Let's name it 'date' if it's not already.
        final_df.index.name = 'date'
        final_df.reset_index(inplace=True)
        
        # Feature Engineering
        final_df['date'] = pd.to_datetime(final_df['date'], utc=True)
        final_df['month'] = final_df['date'].dt.month
        final_df['day_of_week'] = final_df['date'].dt.dayofweek
        final_df['is_weekend'] = final_df['day_of_week'].isin([5, 6]).astype(int)
        
        # Indian Seasons (simplified)
        # Winter: Dec-Feb, Summer: Mar-Jun, Monsoon: Jul-Sep, Post-monsoon: Oct-Nov
        def get_season(m):
            if m in [12, 1, 2]: return 'Winter'
            if m in [3, 4, 5, 6]: return 'Summer'
            if m in [7, 8, 9]: return 'Monsoon'
            return 'Post-Monsoon'
            
        final_df['season'] = final_df['month'].apply(get_season)
        
        # Sort by city and date
        final_df.sort_values(['city', 'date'], inplace=True)
        
        # Save processed data
        final_df.to_csv(f"{PROCESSED_DATA_PATH}/combined_data.csv", index=False)
        print(f"Saved combined processed data to {PROCESSED_DATA_PATH}/combined_data.csv")
        
        # Brief Summary
        print("\nSummary Statistics:")
        print(final_df.groupby('city')[['pm25', 'pm10', 'no2', 'o3', 'co', 'so2']].mean())

if __name__ == "__main__":
    main()
