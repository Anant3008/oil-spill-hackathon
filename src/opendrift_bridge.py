import xarray as xr
import pandas as pd
import numpy as np
import requests
import os
import logging
from datetime import datetime

from src.vector_math import speed_dir_to_uv
from opendrift.readers import reader_netCDF_CF_generic

logger = logging.getLogger(__name__)

MARINE_API_URL = "https://marine-api.open-meteo.com/v1/marine"
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"

def create_env_reader_from_api(center_lat, center_lon, grid_size=3, step_deg=0.08, filename="output/env_forcing.nc"):
    """
    Builds a true spatial environmental grid using Open-Meteo's multi-coordinate API.
    Makes exactly one request per API to avoid rate-limiting and performance bottlenecks.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    logger.info(f"Generating true {grid_size}x{grid_size} spatial grid via single multi-coordinate API request...")

    # 1. Calculate 1D bounds
    half_size = grid_size // 2
    lats_1d = np.linspace(center_lat - half_size * step_deg, center_lat + half_size * step_deg, grid_size)
    lons_1d = np.linspace(center_lon - half_size * step_deg, center_lon + half_size * step_deg, grid_size)

    # 2. Flatten grid into coordinate pairs
    lat_list, lon_list = [], []
    for lat in lats_1d:
        for lon in lons_1d:
            lat_list.append(lat)
            lon_list.append(lon)

    # 3. Fetch from Marine API (One bulk request)
    # Added 'wave_height' to fetch significant wave height for OpenDrift
    marine_params = {
        "latitude": ",".join(map(str, lat_list)),
        "longitude": ",".join(map(str, lon_list)),
        "hourly": "ocean_current_velocity,ocean_current_direction,wave_height",
        "timezone": "UTC"
    }
    logger.info("Fetching multi-coordinate Marine data...")
    marine_resp = requests.get(MARINE_API_URL, params=marine_params).json()

    # 4. Fetch from Weather API (One bulk request)
    weather_params = {
        "latitude": ",".join(map(str, lat_list)),
        "longitude": ",".join(map(str, lon_list)),
        "hourly": "wind_speed_10m,wind_direction_10m",
        "timezone": "UTC"
    }
    logger.info("Fetching multi-coordinate Weather data...")
    weather_resp = requests.get(WEATHER_API_URL, params=weather_params).json()

    # Ensure uniform list structure even if grid_size=1
    if isinstance(marine_resp, dict) and "hourly" in marine_resp:
        marine_resp = [marine_resp]
        weather_resp = [weather_resp]

    # Extract global time array from the first coordinate's response
    time_array = pd.to_datetime(marine_resp[0]["hourly"]["time"])
    num_times = len(time_array)

    # 5. Pre-allocate the 3D grid arrays (time, lat, lon)
    u_curr_grid = np.zeros((num_times, grid_size, grid_size))
    v_curr_grid = np.zeros((num_times, grid_size, grid_size))
    u_wind_grid = np.zeros((num_times, grid_size, grid_size))
    v_wind_grid = np.zeros((num_times, grid_size, grid_size))
    wave_grid = np.zeros((num_times, grid_size, grid_size))

    # 6. Map the flat JSON response array back into the 2D geographic grid
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            m_hourly = marine_resp[idx]["hourly"]
            w_hourly = weather_resp[idx]["hourly"]

            # API returns km/h; physics engines require m/s
            spd_c = np.array(m_hourly["ocean_current_velocity"]) / 3.6
            dir_c = np.array(m_hourly["ocean_current_direction"])
            spd_w = np.array(w_hourly["wind_speed_10m"]) / 3.6
            dir_w = np.array(w_hourly["wind_direction_10m"])

            # Extract wave height (already in meters from Open-Meteo)
            wave_h = np.array(m_hourly["wave_height"])

            # Convert to U/V vectors
            u_c, v_c = speed_dir_to_uv(spd_c, dir_c, is_wind=False)
            u_w, v_w = speed_dir_to_uv(spd_w, dir_w, is_wind=True)

            # Place in spatial grid
            u_curr_grid[:, i, j] = u_c
            v_curr_grid[:, i, j] = v_c
            u_wind_grid[:, i, j] = u_w
            v_wind_grid[:, i, j] = v_w

            # If wave_height is None/NaN from API, fallback to 0.0
            wave_grid[:, i, j] = np.nan_to_num(wave_h, nan=0.0)

            idx += 1

    # 7. Package into a CF-compliant XArray Dataset using EXACT OpenDrift internal variable names
    ds = xr.Dataset(
        {
            "x_wind": (["time", "lat", "lon"], u_wind_grid),
            "y_wind": (["time", "lat", "lon"], v_wind_grid),
            "x_sea_water_velocity": (["time", "lat", "lon"], u_curr_grid),
            "y_sea_water_velocity": (["time", "lat", "lon"], v_curr_grid),
            "sea_surface_wave_significant_height": (["time", "lat", "lon"], wave_grid)
        },
        coords={
            "time": time_array,
            "lat": lats_1d,
            "lon": lons_1d
        }
    )

    # Apply exact standard names and units
    ds.x_wind.attrs['standard_name'] = 'x_wind'
    ds.x_wind.attrs['units'] = 'm/s'
    ds.y_wind.attrs['standard_name'] = 'y_wind'
    ds.y_wind.attrs['units'] = 'm/s'

    ds.x_sea_water_velocity.attrs['standard_name'] = 'x_sea_water_velocity'
    ds.x_sea_water_velocity.attrs['units'] = 'm/s'
    ds.y_sea_water_velocity.attrs['standard_name'] = 'y_sea_water_velocity'
    ds.y_sea_water_velocity.attrs['units'] = 'm/s'

    ds.sea_surface_wave_significant_height.attrs['standard_name'] = 'sea_surface_wave_significant_height'
    ds.sea_surface_wave_significant_height.attrs['units'] = 'm'

    ds.lon.attrs['standard_name'] = 'longitude'
    ds.lon.attrs['units'] = 'degrees_east'
    ds.lat.attrs['standard_name'] = 'latitude'
    ds.lat.attrs['units'] = 'degrees_north'

    ds.to_netcdf(filename)
    logger.info(f"Spatial Environmental NetCDF successfully saved to {filename}")

    reader = reader_netCDF_CF_generic.Reader(filename)
    return reader, time_array[0]
