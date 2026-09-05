import os
import logging
import requests
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from opendrift.readers import reader_netCDF_CF_generic
from vector_math import speed_dir_to_uv

logger = logging.getLogger(__name__)

MARINE_API_URL = "https://marine-api.open-meteo.com/v1/marine"
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"

class EnvironmentManager:
    """
    Data interface for OpenDrift. 
    Allows hot-swapping between Open-Meteo API data and professional NetCDF/OPeNDAP sources
    without changing the simulation code.
    """
    def __init__(self, cache_dir="output"):
        self.cache_dir = cache_dir
        self.readers = []
        os.makedirs(self.cache_dir, exist_ok=True)

    def add_netcdf(self, filepath_or_url):
        """Mounts a local NetCDF file or remote OPeNDAP URL."""
        logger.info(f"Mounting NetCDF reader: {filepath_or_url}")
        reader = reader_netCDF_CF_generic.Reader(filepath_or_url)
        self.readers.append(reader)
        return reader

    def add_openmeteo_grid(self, center_lat, center_lon, start_date=None, end_date=None, grid_size=5, step_deg=0.25):
        """
        Fetches a geographic grid from Open-Meteo APIs, handling time dependence for hindcasting/forecasting.
        Dates should be 'YYYY-MM-DD' strings.
        """
        filename = os.path.join(self.cache_dir, f"openmeteo_{center_lat}_{center_lon}.nc")
        logger.info(f"Generating spatial grid via Open-Meteo bulk APIs...")

        half_size = grid_size // 2
        lats_1d = np.linspace(center_lat - half_size * step_deg, center_lat + half_size * step_deg, grid_size)
        lons_1d = np.linspace(center_lon - half_size * step_deg, center_lon + half_size * step_deg, grid_size)

        lat_list, lon_list = [], []
        for lat in lats_1d:
            for lon in lons_1d:
                lat_list.append(lat)
                lon_list.append(lon)

        marine_params = {
            "latitude": ",".join(map(str, lat_list)),
            "longitude": ",".join(map(str, lon_list)),
            "hourly": "ocean_current_velocity,ocean_current_direction,wave_height",
            "timezone": "UTC"
        }
        weather_params = {
            "latitude": ",".join(map(str, lat_list)),
            "longitude": ",".join(map(str, lon_list)),
            "hourly": "wind_speed_10m,wind_direction_10m",
            "timezone": "UTC"
        }

        # Apply time constraints if provided
        if start_date and end_date:
            marine_params["start_date"] = start_date
            marine_params["end_date"] = end_date
            weather_params["start_date"] = start_date
            weather_params["end_date"] = end_date
        
        logger.info("Fetching multi-coordinate data...")
        marine_resp = requests.get(MARINE_API_URL, params=marine_params).json()
        weather_resp = requests.get(WEATHER_API_URL, params=weather_params).json()

        if isinstance(marine_resp, dict) and "hourly" in marine_resp:
            marine_resp = [marine_resp]
            weather_resp = [weather_resp]

        if "error" in marine_resp[0] or "error" in weather_resp[0]:
            raise ValueError(f"API Error. Marine: {marine_resp[0].get('reason')}, Weather: {weather_resp[0].get('reason')}")

        time_array = pd.to_datetime(marine_resp[0]["hourly"]["time"])
        num_times = len(time_array)

        u_curr_grid = np.zeros((num_times, grid_size, grid_size))
        v_curr_grid = np.zeros((num_times, grid_size, grid_size))
        u_wind_grid = np.zeros((num_times, grid_size, grid_size))
        v_wind_grid = np.zeros((num_times, grid_size, grid_size))
        wave_grid = np.zeros((num_times, grid_size, grid_size))

        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                m_hourly = marine_resp[idx]["hourly"]
                w_hourly = weather_resp[idx]["hourly"]

                spd_c = np.array(m_hourly["ocean_current_velocity"]) / 3.6
                dir_c = np.array(m_hourly["ocean_current_direction"])
                spd_w = np.array(w_hourly["wind_speed_10m"]) / 3.6
                dir_w = np.array(w_hourly["wind_direction_10m"])
                wave_h = np.array(m_hourly["wave_height"])

                u_c, v_c = speed_dir_to_uv(spd_c, dir_c, is_wind=False)
                u_w, v_w = speed_dir_to_uv(spd_w, dir_w, is_wind=True)

                u_curr_grid[:, i, j] = u_c
                v_curr_grid[:, i, j] = v_c
                u_wind_grid[:, i, j] = u_w
                v_wind_grid[:, i, j] = v_w
                wave_grid[:, i, j] = np.nan_to_num(wave_h, nan=0.0)

                idx += 1

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
        logger.info(f"Open-Meteo NetCDF cached to {filename}")

        reader = reader_netCDF_CF_generic.Reader(filename)
        self.readers.append(reader)
        
        # Return reader and the dataset's first time step as a fallback start_time
        return reader, time_array[0]

    def get_readers(self):
        return self.readers
