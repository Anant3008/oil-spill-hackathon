import xarray as xr
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

from data_fetcher import get_environmental_data
from vector_math import speed_dir_to_uv
from opendrift.readers import reader_netCDF_CF_generic

logger = logging.getLogger(__name__)

def create_env_reader_from_api(center_lat, center_lon, grid_size=3, step_deg=0.5, filename="output/env_forcing.nc"):
    """
    Builds a true spatial environmental grid by calling the existing single-point 
    data fetcher multiple times across a bounding box.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    logger.info(f"Generating true {grid_size}x{grid_size} spatial grid using existing data_fetcher...")
    
    # Calculate grid bounds
    half_size = grid_size // 2
    lats_1d = np.linspace(center_lat - half_size * step_deg, center_lat + half_size * step_deg, grid_size)
    lons_1d = np.linspace(center_lon - half_size * step_deg, center_lon + half_size * step_deg, grid_size)
    
    # We don't know the time array length until we fetch the first point
    time_array = None
    num_times = 0
    
    # 3D Grids
    u_curr_grid = None
    v_curr_grid = None
    u_wind_grid = None
    v_wind_grid = None
    
    # Suppress excessive logging from the fetcher since we are calling it 9 times
    logging.getLogger('data_fetcher').setLevel(logging.WARNING)

    # Loop over every point in the grid and call the existing fetcher
    for i, lat in enumerate(lats_1d):
        for j, lon in enumerate(lons_1d):
            # USE THE EXISTING SINGLE-POINT FETCHER!
            df = get_environmental_data(lat, lon)
            
            # Initialize arrays on the first successful fetch
            if time_array is None:
                time_array = df['time'].values
                num_times = len(time_array)
                u_curr_grid = np.zeros((num_times, grid_size, grid_size))
                v_curr_grid = np.zeros((num_times, grid_size, grid_size))
                u_wind_grid = np.zeros((num_times, grid_size, grid_size))
                v_wind_grid = np.zeros((num_times, grid_size, grid_size))
                
            # Convert scalar speeds/dirs to UV vectors
            u_wind, v_wind = speed_dir_to_uv(df['wind_speed_10m']/3.6, df['wind_direction_10m'], is_wind=True)
            u_curr, v_curr = speed_dir_to_uv(df['ocean_current_velocity']/3.6, df['ocean_current_direction'], is_wind=False)
            
            # Place the timeseries into the correct spatial pixel
            u_curr_grid[:, i, j] = u_curr
            v_curr_grid[:, i, j] = v_curr
            u_wind_grid[:, i, j] = u_wind
            v_wind_grid[:, i, j] = v_wind
            
    # Package into XArray Dataset with strict CF-compliant variable names
    ds = xr.Dataset(
        {
            "eastward_wind": (["time", "lat", "lon"], u_wind_grid),
            "northward_wind": (["time", "lat", "lon"], v_wind_grid),
            "eastward_sea_water_velocity": (["time", "lat", "lon"], u_curr_grid),
            "northward_sea_water_velocity": (["time", "lat", "lon"], v_curr_grid),
        },
        coords={
            "time": time_array,
            "lat": lats_1d,
            "lon": lons_1d
        }
    )
    
    # CF standard names mapped to OpenDrift internal names
    ds.eastward_wind.attrs['standard_name'] = 'eastward_wind'
    ds.eastward_wind.attrs['units'] = 'm/s'
    ds.northward_wind.attrs['standard_name'] = 'northward_wind'
    ds.northward_wind.attrs['units'] = 'm/s'
    
    ds.eastward_sea_water_velocity.attrs['standard_name'] = 'eastward_sea_water_velocity'
    ds.eastward_sea_water_velocity.attrs['units'] = 'm/s'
    ds.northward_sea_water_velocity.attrs['standard_name'] = 'northward_sea_water_velocity'
    ds.northward_sea_water_velocity.attrs['units'] = 'm/s'
    
    ds.lon.attrs['standard_name'] = 'longitude'
    ds.lon.attrs['units'] = 'degrees_east'
    ds.lat.attrs['standard_name'] = 'latitude'
    ds.lat.attrs['units'] = 'degrees_north'
    
    ds.to_netcdf(filename)
    logger.info(f"Spatial Environmental NetCDF successfully saved to {filename}")
    
    # Load and return the OpenDrift reader
    reader = reader_netCDF_CF_generic.Reader(filename)
    return reader, time_array[0]
