from src.environment import EnvironmentManager
from opendrift.models.openoil import OpenOil
import numpy as np
from datetime import timedelta

start_lat, start_lon = 18.5, 71.5
env_manager = EnvironmentManager()
reader_env, start_time = env_manager.add_openmeteo_grid(start_lat, start_lon, grid_size=3, step_deg=0.2)

# Print bounds of the reader
print(f"Reader bounds: lon {reader_env.xmin} to {reader_env.xmax}, lat {reader_env.ymin} to {reader_env.ymax}")

o = OpenOil(loglevel=30)
o.add_reader(reader_env)
o.seed_elements(lon=start_lon, lat=start_lat, number=1, time=start_time)

# Run step by step and monitor position
for i in range(24):
    o.run(steps=1, time_step=timedelta(hours=1))
    lon = o.elements.lon[0]
    lat = o.elements.lat[0]
    status = o.elements.status[0]
    print(f"Hour {i+1}: Particle at lon={lon:.4f}, lat={lat:.4f}, status={status}")
    
    # Check if particle left bounds
    if lon < reader_env.xmin or lon > reader_env.xmax or lat < reader_env.ymin or lat > reader_env.ymax:
        print(">>> Particle moved OUTSIDE reader bounds! This causes NaN interpolation!")
    if status != 0:
        break
