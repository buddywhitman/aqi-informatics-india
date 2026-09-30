import os
import pandas as pd
from openaq import OpenAQ
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta

# Configuration
OPENAQ_API_KEY = "853598fc71cb58e85940ebe98ff95ff83d7a1e89361b7e777d8196624716d2f2"
RAW_DATA_PATH = "data/raw"

# Selected Stations
STATIONS = {
    "Delhi": {"id": 17, "coords": (28.56, 77.18)},  # R K Puram
    "Mumbai": {"id": 6927, "coords": (18.90, 72.82)}, # Colaba
    "Bengaluru": {"id": 5548, "coords": (12.91, 77.59)} # BTM Layout
}

def fetch_pollution_data(client, station_id, start_date, end_date):
    """Fetch daily pollution data for a specific station."""
    print(f"Fetching pollution data for station {station_id} from {start_date} to {end_date}...")
    
    # In OpenAQ v3, measurements are tied to sensors. 
    # We first need to get the sensors for this location.
    sensors_response = client.locations.sensors(locations_id=station_id)
    sensor_ids = [s.id for s in sensors_response.results]
    
    all_data = []
    for s_id in sensor_ids:
        # Fetch daily averages for each sensor
        # Note: OpenAQ API might have a limit on how many results it returns per call.
        # We might need to paginate or loop through smaller time chunks.
        try:
            measurements = client.measurements.list(
                sensors_id=s_id,
                data="days",
                datetime_from=start_date.isoformat(),
                datetime_to=end_date.isoformat(),
                limit=1000
            )
            for res in measurements.results:
                all_data.append({
                    "timestamp": res.period.datetime_from,
                    "parameter": res.parameter.name,
                    "value": res.value,
                    "units": res.parameter.units
                })
        except Exception as e:
            print(f"Error fetching sensor {s_id}: {e}")
            
    return pd.DataFrame(all_data)

def fetch_weather_data(lat, lon, start_date, end_date):
    """Fetch historical weather data from Open-Meteo."""
    print(f"Fetching weather data for ({lat}, {lon}) from {start_date} to {end_date}...")
    
    # Setup Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max", "relative_humidity_2m_max"],
        "timezone": "GMT"
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    daily = response.Daily()
    daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
    daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(2).ValuesAsNumpy()
    daily_wind_speed_10m_max = daily.Variables(3).ValuesAsNumpy()
    daily_relative_humidity_2m_max = daily.Variables(4).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    )}
    daily_data["temp_max"] = daily_temperature_2m_max
    daily_data["temp_min"] = daily_temperature_2m_min
    daily_data["precipitation"] = daily_precipitation_sum
    daily_data["wind_speed_max"] = daily_wind_speed_10m_max
    daily_data["humidity_max"] = daily_relative_humidity_2m_max

    return pd.DataFrame(daily_data)

def main():
    client = OpenAQ(api_key=OPENAQ_API_KEY)
    
    # Define timeframe (last 5 years for now to avoid massive data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    for city, info in STATIONS.items():
        print(f"=== Processing {city} ===")
        
        # Fetch Pollution
        pollution_df = fetch_pollution_data(client, info["id"], start_date, end_date)
        if not pollution_df.empty:
            pollution_df.to_csv(f"{RAW_DATA_PATH}/{city}_pollution_raw.csv", index=False)
            print(f"Saved {city} pollution data.")
        
        # Fetch Weather
        weather_df = fetch_weather_data(info["coords"][0], info["coords"][1], start_date, end_date)
        if not weather_df.empty:
            weather_df.to_csv(f"{RAW_DATA_PATH}/{city}_weather_raw.csv", index=False)
            print(f"Saved {city} weather data.")
            
    client.close()

if __name__ == "__main__":
    main()
