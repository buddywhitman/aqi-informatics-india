import pandas as pd
import numpy as np
import re
import os

RAW_PATH = 'data/raw_hourly'

def parse_ts(ts_str):
    if not isinstance(ts_str, str):
        return ts_str
    # Extract timestamp inside utc='...'
    idx = ts_str.find("utc='")
    if idx != -1:
        end_idx = ts_str.find("'", idx + 5)
        if end_idx != -1:
            return ts_str[idx + 5:end_idx]
    return ts_str

pollutant_keys = ['pm25', 'pm10', 'no2', 'no', 'nox', 'so2', 'co', 'o3']

for city in ['Delhi', 'Mumbai', 'Bengaluru', 'Kolkata']:
    p_file = f'{RAW_PATH}/{city}_pollution_hourly.csv'
    w_file = f'{RAW_PATH}/{city}_weather_hourly.csv'
    if not os.path.exists(p_file):
        continue
    
    pdf = pd.read_csv(p_file)
    wdf = pd.read_csv(w_file)
    
    # Fast timestamp parsing
    pdf['timestamp'] = pd.to_datetime(pdf['timestamp'].apply(parse_ts), utc=True).dt.round('h')
    
    # Filter physical ranges: pollutants must be >= 0 and within realistic upper bounds
    pdf = pdf[(pdf['value'] >= 0) & (pdf['value'] < 2500)]
    
    # Filter only genuine pollutant parameters (leave meteorological variables to Open-Meteo)
    pdf = pdf[pdf['parameter'].isin(pollutant_keys)]
    
    p_piv = pdf.pivot_table(index='timestamp', columns='parameter', values='value', aggfunc='mean')
    
    wdf['timestamp'] = pd.to_datetime(wdf['timestamp'], utc=True).dt.round('h')
    wdf.set_index('timestamp', inplace=True)
    
    merged = pd.merge(p_piv, wdf, left_index=True, right_index=True, how='inner')
    merged['city'] = city
    
    print(f"*** {city} ***")
    print(f"  Merged hourly rows = {len(merged)} ({merged.index.min()} to {merged.index.max()})")
    print(f"  Columns: {list(merged.columns)}")
    valid_core = merged.dropna(subset=['pm25', 'no2', 'temperature', 'wind_speed', 'humidity', 'pressure'])
    print(f"  Complete valid core cases: {len(valid_core)}")
    print(f"  PM2.5 range: [{merged['pm25'].min():.1f}, {merged['pm25'].max():.1f}]")
    print(f"  NO2 range:   [{merged['no2'].min():.1f}, {merged['no2'].max():.1f}]")
    print(f"  Temp range:  [{merged['temperature'].min():.1f}, {merged['temperature'].max():.1f}] C")
    print(f"  Wind range:  [{merged['wind_speed'].min():.1f}, {merged['wind_speed'].max():.1f}] m/s")
