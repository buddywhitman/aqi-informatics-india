import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import os
import glob

# Configuration
RAW_DATA_PATH = "data/raw_hourly"
PROCESSED_DATA_PATH = "data/processed_hourly"

import re

def parse_openaq_timestamp(ts_str):
    """Extract UTC timestamp from the complex OpenAQ string."""
    if not isinstance(ts_str, str): return ts_str
    match = re.search(r"utc='(.*?)'", ts_str)
    if match:
        return pd.to_datetime(match.group(1))
    return pd.to_datetime(ts_str, errors='coerce')

def preprocess_city_hourly(city):
    """Clean, pivot, and merge hourly data for a city."""
    poll_file = f"{RAW_DATA_PATH}/{city}_pollution_hourly.csv"
    weath_file = f"{RAW_DATA_PATH}/{city}_weather_hourly.csv"
    
    if not os.path.exists(poll_file) or not os.path.exists(weath_file):
        return None
        
    poll_df = pd.read_csv(poll_file)
    weath_df = pd.read_csv(weath_file)
    
    # 1. Pollution Preprocessing
    poll_df['timestamp'] = poll_df['timestamp'].apply(parse_openaq_timestamp)
    poll_df['timestamp'] = pd.to_datetime(poll_df['timestamp'], utc=True)
    # Round to nearest hour for alignment (using 'h' for Pandas 3.0)
    poll_df['timestamp'] = poll_df['timestamp'].dt.round('h')
    poll_pivoted = poll_df.pivot_table(index='timestamp', columns='parameter', values='value', aggfunc='mean')
    
    # 2. Weather Preprocessing
    weath_df['timestamp'] = pd.to_datetime(weath_df['timestamp'], utc=True)
    # Round to nearest hour for alignment
    weath_df['timestamp'] = weath_df['timestamp'].dt.round('h')
    weath_df.set_index('timestamp', inplace=True)
    
    # 3. Merge
    combined = pd.merge(poll_pivoted, weath_df, left_index=True, right_index=True, how='outer')
    combined['city'] = city
    
    # 4. Fill gaps (Linear Interpolation for small gaps)
    # Only interpolate numeric columns
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    combined[numeric_cols] = combined[numeric_cols].interpolate(method='linear', limit=3)
    
    return combined

def main():
    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)
        
    cities = ["Delhi", "Mumbai", "Bengaluru", "Kolkata", "Chennai", "Hyderabad", "Ahmedabad"]
    all_combined = []
    
    for city in cities:
        print(f"Processing {city}...")
        df = preprocess_city_hourly(city)
        if df is not None:
            all_combined.append(df)
            
    if not all_combined:
        print("No data to process.")
        return
        
    final_df = pd.concat(all_combined)
    final_df.index.name = 'timestamp'
    final_df.reset_index(inplace=True)
    
    # 5. Multivariate Imputation (IterativeImputer)
    # We only impute numerical columns
    num_cols = final_df.select_dtypes(include=[np.number]).columns
    print(f"Imputing columns: {num_cols.tolist()}")
    
    # To save time and memory, we'll do a simple KNN or Iterative imputer on a subset if needed
    # For now, let's use IterativeImputer with a limit on max_iter
    imputer = IterativeImputer(max_iter=5, random_state=42)
    final_df[num_cols] = imputer.fit_transform(final_df[num_cols])
    
    # 6. Feature Engineering
    final_df['hour'] = final_df['timestamp'].dt.hour
    final_df['day_of_week'] = final_df['timestamp'].dt.dayofweek
    final_df['month'] = final_df['timestamp'].dt.month
    
    # Save
    final_df.to_csv(f"{PROCESSED_DATA_PATH}/combined_hourly.csv", index=False)
    print(f"Saved processed hourly data to {PROCESSED_DATA_PATH}/combined_hourly.csv")
    print(final_df.head())

if __name__ == "__main__":
    main()
