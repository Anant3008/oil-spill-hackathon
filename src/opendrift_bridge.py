import xarray as xr
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

from src.data_fetcher import get_environmental_data
from src.vector_math import speed_dir_to_uv
from opendrift.readers import reader_netCDF_CF_generic

logger = logging.getLogger(__name__)

def create_env_reader_from_api(lat, lon, filename="output/env_forcing.nc"):
    """
    Fetches live API data using our data_fetcher, calculates U/V vectors,
    and packages it into a standard CF-compliant NetCDF file that OpenDrift can read natively.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 1. Fetch live data
    logger.info("Fetching Open-Meteo data for OpenDrift integration...")
    df = get_environmental_data(lat, lon)
    
    # 2. Convert speeds to m/s and directions to U/V vectors
    u_wind, v_wind = speed_dir_to_uv(df['wind_speed_10m']/3.6, df['wind_direction_10m'], is_wind=True)
    u_curr, v_curr = speed_dir_to_uv(df['ocean_current_velocity']/3.6, df['ocean_current_direction'], is_wind=False)
    
    # 3. Create an xarray Dataset representing a 3D grid (time, lat, lon)
    # Even though it's a single point, OpenDrift requires explicit spatial dimensions.
    # We create a tiny spatial box around the point to ensure gradients work without errors.
    lats = [lat - 0.5, lat, lat + 0.5]
    lons = [lon - 0.5, lon, lon + 0.5]
    
    # Broadcast our single-point temporal data across this 3x3 grid
    shape = (len(df), 3, 3)
    u_wind_grid = np.broadcast_to(u_wind.values[:, None, None], shape)
    v_wind_grid = np.broadcast_to(v_wind.values[:, None, None], shape)
    u_curr_grid = np.broadcast_to(u_curr.values[:, None, None], shape)
    v_curr_grid = np.broadcast_to(v_curr.values[:, None, None], shape)
    
    ds = xr.Dataset(
        {
            "x_wind": (["time", "lat", "lon"], u_wind_grid),
            "y_wind": (["time", "lat", "lon"], v_wind_grid),
            "x_sea_water_velocity": (["time", "lat", "lon"], u_curr_grid),
            "y_sea_water_velocity": (["time", "lat", "lon"], v_curr_grid),
        },
        coords={
            "time": df['time'].values,
            "lat": lats,
            "lon": lons
        }
    )
    
    # 4. Attach strict CF-compliant metadata (Critical for OpenDrift to recognize variables)
    ds.x_wind.attrs['standard_name'] = 'x_wind'
    ds.x_wind.attrs['units'] = 'm/s'
    ds.y_wind.attrs['standard_name'] = 'y_wind'
    ds.y_wind.attrs['units'] = 'm/s'
    ds.x_sea_water_velocity.attrs['standard_name'] = 'x_sea_water_velocity'
    ds.x_sea_water_velocity.attrs['units'] = 'm/s'
    ds.y_sea_water_velocity.attrs['standard_name'] = 'y_sea_water_velocity'
    ds.y_sea_water_velocity.attrs['units'] = 'm/s'
    
    ds.lon.attrs['standard_name'] = 'longitude'
    ds.lon.attrs['units'] = 'degrees_east'
    ds.lat.attrs['standard_name'] = 'latitude'
    ds.lat.attrs['units'] = 'degrees_north'
    
    # 5. Save to disk and load as an OpenDrift reader
    ds.to_netcdf(filename)
    logger.info(f"Environmental NetCDF saved to {filename}")
    
    reader = reader_netCDF_CF_generic.Reader(filename)
    return reader, df['time'].iloc[0]

if __name__ == "__main__":
    reader, start_time = create_env_reader_from_api(19.07, 72.88)
    print("\n--- Reader Created Successfully ---")
    print(reader)
