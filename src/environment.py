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

    def calculate_shifted_grid(self, start_lat, start_lon, start_date, end_date, duration_hours, is_backward=False):
        """
        Pre-flight check: Downloads 1 point to find the average wind/current drift vector.
        Shifts the grid center to provide adaptive trajectory-focused spatial coverage.
        Returns (new_lat, new_lon, optimal_step_deg).
        """
        self.start_date = start_date
        self.end_date = end_date

        logger.info(f"Running pre-flight check at ({start_lat}, {start_lon}) for adaptive coverage...")

        m_params = {
            "latitude": start_lat, "longitude": start_lon,
            "hourly": "ocean_current_velocity,ocean_current_direction",
            "timezone": "UTC", "start_date": start_date, "end_date": end_date
        }
        w_params = {
            "latitude": start_lat, "longitude": start_lon,
            "hourly": "wind_speed_10m,wind_direction_10m",
            "timezone": "UTC", "start_date": start_date, "end_date": end_date
        }

        try:
            m_resp = requests.get(MARINE_API_URL, params=m_params)
            m_resp.raise_for_status()
            m_data = m_resp.json()

            w_resp = requests.get(WEATHER_API_URL, params=w_params)
            w_resp.raise_for_status()
            w_data = w_resp.json()

            m_hourly = m_data["hourly"]
            w_hourly = w_data["hourly"]

            spd_c = np.array(m_hourly["ocean_current_velocity"]) / 3.6  # km/h to m/s
            dir_c = np.array(m_hourly["ocean_current_direction"])
            spd_w = np.array(w_hourly["wind_speed_10m"]) / 3.6
            dir_w = np.array(w_hourly["wind_direction_10m"])

            # Clean NaNs
            spd_c = np.nan_to_num(spd_c, nan=0.0)
            spd_w = np.nan_to_num(spd_w, nan=0.0)

            u_c, v_c = speed_dir_to_uv(spd_c, dir_c, is_wind=False)
            u_w, v_w = speed_dir_to_uv(spd_w, dir_w, is_wind=True)

            avg_u_c, avg_v_c = np.mean(u_c), np.mean(v_c)
            avg_u_w, avg_v_w = np.mean(u_w), np.mean(v_w)

            # Oil drift rule of thumb: 100% current + 3% wind
            u_drift = avg_u_c + (0.03 * avg_u_w)
            v_drift = avg_v_c + (0.03 * avg_v_w)

            if is_backward:
                u_drift = -u_drift
                v_drift = -v_drift

            # Displacement in meters
            duration_sec = duration_hours * 3600
            dx_m = u_drift * duration_sec
            dy_m = v_drift * duration_sec

            # Convert to degrees (approx 111km per latitude degree)
            lat_diff = dy_m / 111000.0
            lon_diff = dx_m / (111000.0 * np.cos(np.radians(start_lat)))

            # Shift center to midpoint
            new_lat = start_lat + (lat_diff / 2)
            new_lon = start_lon + (lon_diff / 2)

            # Dynamically calculate grid step size based on expected travel distance
            box_size_deg = max(abs(lat_diff), abs(lon_diff)) + 0.15
            optimal_step = max(0.02, box_size_deg / 21)
            optimal_step = min(0.08, optimal_step)      # Cap at 0.08 to retain high resolution

            logger.info(f"Avg drift vector: U={u_drift:.3f}m/s, V={v_drift:.3f}m/s.")
            logger.info(f"Adaptive shift to ({new_lat:.3f}, {new_lon:.3f}) with step {optimal_step:.3f} deg.")

            return float(new_lat), float(new_lon), float(optimal_step)

        except Exception as e:
            logger.warning(f"Pre-flight check failed, defaulting to original center. Error: {e}")
            return start_lat, start_lon, 0.05

    def add_openmeteo_grid(self, center_lat, center_lon, start_date=None, end_date=None, grid_size=21, step_deg=0.05):
        """
        Fetches a geographic grid from Open-Meteo APIs, handling time dependence for hindcasting/forecasting.
        Dates should be 'YYYY-MM-DD' strings.
        """
        start_date = start_date or getattr(self, 'start_date', None)
        end_date = end_date or getattr(self, 'end_date', None)

        filename = os.path.join(self.cache_dir, f"openmeteo_{center_lat}_{center_lon}.nc")
        logger.info(f"Generating spatial grid via Open-Meteo bulk APIs...")

        half_size = grid_size // 2
        lats_1d = np.linspace(center_lat - half_size * step_deg, center_lat + half_size * step_deg, grid_size)
        lons_1d = np.linspace(center_lon - half_size * step_deg, center_lon + half_size * step_deg, grid_size)

        lat_list, lon_list = [], []
        for lat in lats_1d:
            for lon in lons_1d:
                # Round to 4 decimal places to prevent 'Request-URI Too Large' HTTP 414 errors
                lat_list.append(round(lat, 4))
                lon_list.append(round(lon, 4))

        num_points = len(lat_list)

        marine_payload = {
            "latitude": lat_list,
            "longitude": lon_list,
            "hourly": ["ocean_current_velocity", "ocean_current_direction", "wave_height"],
            "timezone": ["UTC"] * num_points
        }
        weather_payload = {
            "latitude": lat_list,
            "longitude": lon_list,
            "hourly": ["wind_speed_10m", "wind_direction_10m"],
            "timezone": ["UTC"] * num_points
        }

        marine_params = {}
        weather_params = {}

        # Apply time constraints if provided
        if start_date and end_date:
            marine_payload["start_date"] = [start_date] * num_points
            marine_payload["end_date"] = [end_date] * num_points
            weather_payload["start_date"] = [start_date] * num_points
            weather_payload["end_date"] = [end_date] * num_points

        logger.info("Fetching multi-coordinate data using POST to avoid URL limits...")
        marine_resp_raw = requests.post(MARINE_API_URL, json=marine_payload)
        try:
            marine_resp_raw.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error(f"Marine API Error: {marine_resp_raw.text}")
            raise e
        marine_resp = marine_resp_raw.json()

        weather_resp_raw = requests.post(WEATHER_API_URL, json=weather_payload)
        try:
            weather_resp_raw.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error(f"Weather API Error: {weather_resp_raw.text}")
            raise e
        weather_resp = weather_resp_raw.json()

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
