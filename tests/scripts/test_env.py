from src.opendrift_bridge import create_env_reader_from_api
from opendrift.models.openoil import OpenOil
import numpy as np

start_lat, start_lon = 18.5, 71.5
reader_env, start_time = create_env_reader_from_api(start_lat, start_lon, grid_size=3, step_deg=0.2)

o = OpenOil()
o.add_reader(reader_env)
o.seed_elements(lon=start_lon, lat=start_lat, number=1, time=start_time)

env_arrays = o.get_environment(list(o.required_variables), start_time, np.array([start_lon]), np.array([start_lat]), np.array([0]), np.array([0]))

for var, val in env_arrays.items():
    if val is not None and np.any(np.isnan(val)):
        print(f"{var} is NaN for active elements! {val}")
    else:
        print(f"{var} is valid: {val}")

