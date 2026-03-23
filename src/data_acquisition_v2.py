import os
import pandas as pd
from openaq import OpenAQ
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta

# Configuration
OPENAQ_API_KEY = "853598fc71cb58e85940ebe98ff95ff83d7a1e89361b7e777d8196624716d2f2"
RAW_DATA_PATH = "data/raw_hourly"

# Expanded Stations (Key Hubs for 7 cities)
STATIONS = {
    "Delhi": {"id": 17, "coords": (28.56, 77.18)},
    "Mumbai": {"id": 6927, "coords": (18.90, 72.82)},
    "Bengaluru": {"id": 5548, "coords": (12.91, 77.59)},
    "Kolkata": {"id": 2596, "coords": (22.57, 88.36)},
    "Chennai": {"id": 2588, "coords": (13.08, 80.27)},
    "Hyderabad": {"id": 2587, "coords": (17.38, 78.48)},
    "Ahmedabad": {"id": 2505, "coords": (23.02, 72.57)}
}

import time
from openaq.shared.exceptions import RateLimitError

def fetch_pollution_hourly(client, station_id, start_date, end_date):
    """Fetch hourly pollution data with proper pagination and rate limit handling."""
    def call_with_retry(func, *args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except RateLimitError:
                print("    Rate limit hit, sleeping 35 seconds...")
                time.sleep(35)
            except Exception as e:
                print(f"    Error: {e}")
                return None

    sensors_response = call_with_retry(client.locations.sensors, locations_id=station_id)
    if not sensors_response: return pd.DataFrame()
    sensor_ids = [s.id for s in sensors_response.results]
    
    all_data = []
    for s_id in sensor_ids:
        print(f"  - Sensor {s_id}")
        current_start = start_date
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=30), end_date)
            
            measurements = call_with_retry(
                client.measurements.list,
                sensors_id=s_id,
                data="hours",
                datetime_from=current_start.isoformat(),
                datetime_to=current_end.isoformat(),
                limit=1000 
            )
            
            if measurements and measurements.results:
                for res in measurements.results:
                    all_data.append({
                        "timestamp": res.period.datetime_from,
                        "parameter": res.parameter.name,
                        "value": res.value
                    })
            current_start = current_end
            
    return pd.DataFrame(all_data)

def fetch_weather_hourly(lat, lon, start_date, end_date):
    """Fetch hourly weather data from Open-Meteo."""
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m", "wind_direction_10m"],
        "timezone": "GMT"
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    
    hourly = response.Hourly()
    hourly_data = {"timestamp": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )}
    hourly_data["temperature"] = hourly.Variables(0).ValuesAsNumpy()
    hourly_data["humidity"] = hourly.Variables(1).ValuesAsNumpy()
    hourly_data["precipitation"] = hourly.Variables(2).ValuesAsNumpy()
    hourly_data["wind_speed"] = hourly.Variables(3).ValuesAsNumpy()
    hourly_data["wind_direction"] = hourly.Variables(4).ValuesAsNumpy()

    return pd.DataFrame(hourly_data)

def main():
    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH)
        
    client = OpenAQ(api_key=OPENAQ_API_KEY)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)
    
    for city, info in STATIONS.items():
        print(f"--- Fetching Hourly Data for {city} ---")
        
        # Pollution
        poll_df = fetch_pollution_hourly(client, info["id"], start_date, end_date)
        if not poll_df.empty:
            poll_df.to_csv(f"{RAW_DATA_PATH}/{city}_pollution_hourly.csv", index=False)
            
        # Weather
        weath_df = fetch_weather_hourly(info["coords"][0], info["coords"][1], start_date, end_date)
        if not weath_df.empty:
            weath_df.to_csv(f"{RAW_DATA_PATH}/{city}_weather_hourly.csv", index=False)
            
    client.close()

if __name__ == "__main__":
    main()
