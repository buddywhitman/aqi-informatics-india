import pandas as pd
import numpy as np
import os
import re

# Configuration
RAW_DATA_PATH = "data/raw_hourly"
PROCESSED_DATA_PATH = "data/processed_hourly"

def parse_openaq_timestamp(ts_str):
    """Extract UTC timestamp from the complex OpenAQ string."""
    if not isinstance(ts_str, str): return ts_str
    match = re.search(r"utc='(.*?)'", ts_str)
    if match:
        return pd.to_datetime(match.group(1))
    return pd.to_datetime(ts_str, errors='coerce')

def preprocess_city_hourly_v3(city):
    """Clean, pivot, and merge hourly data with advanced features."""
    poll_file = f"{RAW_DATA_PATH}/{city}_pollution_hourly.csv"
    weath_file = f"{RAW_DATA_PATH}/{city}_weather_hourly.csv"
    
    if not os.path.exists(poll_file) or not os.path.exists(weath_file):
        return None
        
    poll_df = pd.read_csv(poll_file)
    weath_df = pd.read_csv(weath_file)
    
    # 1. Pollution Preprocessing
    poll_df['timestamp'] = poll_df['timestamp'].apply(parse_openaq_timestamp)
    poll_df['timestamp'] = pd.to_datetime(poll_df['timestamp'], utc=True).dt.round('h')
    poll_pivoted = poll_df.pivot_table(index='timestamp', columns='parameter', values='value', aggfunc='mean')
    
    # 2. Weather Preprocessing
    weath_df['timestamp'] = pd.to_datetime(weath_df['timestamp'], utc=True).dt.round('h')
    weath_df.set_index('timestamp', inplace=True)
    
    # 3. Merge
    combined = pd.merge(poll_pivoted, weath_df, left_index=True, right_index=True, how='outer')
    combined['city'] = city
    
    # 4. Interpolation
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    combined[numeric_cols] = combined[numeric_cols].interpolate(method='linear', limit=3)
    
    # 5. Advanced Feature Engineering
    if 'pm25' in combined.columns and 'pm10' in combined.columns:
        combined['pm25_pm10_ratio'] = combined['pm25'] / (combined['pm10'] + 1e-6)
    
    if 'no2' in combined.columns and 'co' in combined.columns:
        combined['no2_co_ratio'] = combined['no2'] / (combined['co'] + 1e-6)
        
    if 'o3' in combined.columns and 'no2' in combined.columns:
        combined['o3_no2_ratio'] = combined['o3'] / (combined['no2'] + 1e-6)

    # Lags and Rolling Stats
    for col in ['pm25', 'no2', 'temperature', 'wind_speed']:
        if col in combined.columns:
            # Lags
            for l in [1, 3, 6, 24]:
                combined[f'{col}_lag_{l}h'] = combined[col].shift(l)
            # Rolling (3h and 24h)
            combined[f'{col}_roll_3h_mean'] = combined[col].rolling(window=3).mean()
            combined[f'{col}_roll_24h_max'] = combined[col].rolling(window=24).max()
            combined[f'{col}_roll_24h_std'] = combined[col].rolling(window=24).std()

    return combined

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer

def perform_advanced_imputation(df, method='mice'):
    """Handle missing values using SOTA multivariate strategies."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    print(f"  Performing {method.upper()} imputation on: {len(num_cols)} columns")
    
    if method == 'mice':
        imputer = IterativeImputer(max_iter=10, random_state=42)
    else:
        imputer = KNNImputer(n_neighbors=5)
        
    df[num_cols] = imputer.fit_transform(df[num_cols])
    return df

def main():
    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)
        
    cities = ["Delhi", "Mumbai", "Bengaluru", "Kolkata", "Chennai", "Hyderabad", "Ahmedabad"]
    all_combined = []
    
    for city in cities:
        print(f"Processing {city} (v3.1 SOTA)...")
        df = preprocess_city_hourly_v3(city)
        if df is not None:
            # Apply imputation per city to preserve local correlations
            df = perform_advanced_imputation(df, method='mice')
            all_combined.append(df)
            
    if not all_combined:
        print("No data.")
        return
        
    final_df = pd.concat(all_combined)
    final_df.index.name = 'timestamp'
    final_df.reset_index(inplace=True)
    
    # Final cleanup
    final_df.to_csv(f"{PROCESSED_DATA_PATH}/combined_hourly_v3.csv", index=False)
    print(f"Saved v3.0 processed hourly data.")

if __name__ == "__main__":
    main()
