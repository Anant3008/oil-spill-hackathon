import logging
from datetime import timedelta
import numpy as np
import xarray as xr
from opendrift.models.openoil import OpenOil
from src.opendrift_bridge import create_env_reader_from_api

o = OpenOil()
start_lat, start_lon = 18.5, 71.5
reader_env, start_time = create_env_reader_from_api(start_lat, start_lon, grid_size=3, step_deg=0.2)
o.add_reader(reader_env)

# Seed exactly one particle exactly at center
o.seed_elements(lon=start_lon, lat=start_lat, number=1, time=start_time, oil_type='GENERIC HEAVY CRUDE')

# Set a debug hook to check environment values
o.run(duration=timedelta(hours=24), time_step=timedelta(hours=1))
