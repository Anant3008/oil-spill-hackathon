import requests
import pandas as pd
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

MARINE_API_URL = "https://marine-api.open-meteo.com/v1/marine"
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"

def get_environmental_data(lat: float, lon: float) -> pd.DataFrame:
    """
    Fetches hourly ocean current and wind data for a given latitude and longitude.
    Combines Open-Meteo's Marine API (currents) and Weather API (wind).
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        
    Returns:
        pd.DataFrame: A clean DataFrame with hourly time, wind, and current data.
    """
    logger.info(f"Fetching environmental data for Lat: {lat}, Lon: {lon}")
    
    # Parameters for both APIs
    marine_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "ocean_current_velocity,ocean_current_direction",
        "timezone": "UTC"
    }
    
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wind_speed_10m,wind_direction_10m",
        "timezone": "UTC"
    }
    
    try:
        # Fetch Marine Data (Ocean Currents)
        marine_resp = requests.get(MARINE_API_URL, params=marine_params)
        marine_resp.raise_for_status()
        marine_data = marine_resp.json()
        
        # Fetch Weather Data (Wind)
        weather_resp = requests.get(WEATHER_API_URL, params=weather_params)
        weather_resp.raise_for_status()
        weather_data = weather_resp.json()
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        raise

    # Extract 'hourly' sections and convert to Pandas DataFrames
    df_marine = pd.DataFrame(marine_data.get("hourly", {}))
    df_weather = pd.DataFrame(weather_data.get("hourly", {}))
    
    if df_marine.empty or df_weather.empty:
        logger.error("No hourly data returned from the APIs.")
        raise ValueError("Empty hourly data")

    # Merge on the 'time' column
    df_combined = pd.merge(df_marine, df_weather, on="time", how="inner")
    
    # Convert 'time' to proper datetime objects
    df_combined["time"] = pd.to_datetime(df_combined["time"])
    
    # Drop rows with NaN values (e.g., if one API has less data)
    df_combined.dropna(inplace=True)
    df_combined.reset_index(drop=True, inplace=True)
    
    # Log the verification details
    marine_units = marine_data.get("hourly_units", {})
    weather_units = weather_data.get("hourly_units", {})
    
    logger.info("Successfully fetched and merged environmental data.")
    logger.info("--- Data Units ---")
    logger.info(f"Current Velocity:  {marine_units.get('ocean_current_velocity')}")
    logger.info(f"Current Direction: {marine_units.get('ocean_current_direction')}")
    logger.info(f"Wind Speed:        {weather_units.get('wind_speed_10m')}")
    logger.info(f"Wind Direction:    {weather_units.get('wind_direction_10m')}")
    logger.info("--- Sample Data (First 3 hours) ---")
    
    # Print the sample cleanly
    print(df_combined.head(3).to_string(index=False))
    
    return df_combined

if __name__ == "__main__":
    # Test execution for a location (e.g., off the coast of Mumbai)
    test_lat, test_lon = 19.07, 72.88
    df = get_environmental_data(test_lat, test_lon)
