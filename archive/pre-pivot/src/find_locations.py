import os
from openaq import OpenAQ
import pandas as pd

# Use the API key provided by the user
OPENAQ_API_KEY = "853598fc71cb58e85940ebe98ff95ff83d7a1e89361b7e777d8196624716d2f2"

def get_location_ids(client, city_name, lat, lon):
    try:
        # Search for locations within 20km of the city center
        response = client.locations.list(coordinates=(lat, lon), radius=20000, limit=20)
        locations = []
        for res in response.results:
            locations.append({
                "id": res.id,
                "name": res.name,
                "locality": res.locality,
                "sensors": [s.parameter.name for s in res.sensors]
            })
        return locations
    except Exception as e:
        print(f"Error fetching for {city_name}: {e}")
        return []

if __name__ == "__main__":
    client = OpenAQ(api_key=OPENAQ_API_KEY)
    cities = {
        "Delhi": (28.61, 77.20),
        "Mumbai": (19.07, 72.87),
        "Bengaluru": (12.97, 77.59)
    }
    try:
        for city, coords in cities.items():
            print(f"--- Locations in {city} ---")
            locs = get_location_ids(client, city, coords[0], coords[1])
            for loc in locs:
                print(f"ID: {loc['id']}, Name: {loc['name']}, Locality: {loc['locality']}, Sensors: {loc['sensors']}")
            print("\n")
    finally:
        client.close()
