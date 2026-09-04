import numpy as np
import xarray as xr
import json
from datetime import timedelta
import logging

logging.basicConfig(level=logging.WARNING)

start_lat = 18.5
start_lon = 71.5
num_particles = 1000
duration_hours = 24

print("="*60)
print("1. RUNNING CUSTOM EULER MODEL (Phase 1-8)")
print("="*60)
# Add src to sys.path so drift_model can import its local files
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.drift_model import run_cloud_simulation
# Run the custom model with a slight diffusion to simulate realistic spread
cloud_custom = run_cloud_simulation(start_lat, start_lon, num_particles=num_particles, hours=duration_hours, diffusion_k=5.0)

print("\n" + "="*60)
print("2. RUNNING OPENDRIFT MODEL (Phase 10)")
print("="*60)
from opendrift.models.openoil import OpenOil
from src.opendrift_bridge import create_env_reader_from_api

reader_env, start_time = create_env_reader_from_api(start_lat, start_lon, grid_size=5, step_deg=0.25)
o = OpenOil(loglevel=50) # suppress verbose opendrift output
o.add_reader(reader_env)

# Seed particles with 500m radius (~0.005 spread) to match custom model's initial Gaussian spread
o.seed_elements(lon=start_lon, lat=start_lat,
                radius=500,
                number=num_particles,
                time=start_time,
                oil_type='GENERIC HEAVY CRUDE')

outfile = 'output/comparison_opendrift.nc'
o.run(duration=timedelta(hours=duration_hours),
      time_step=timedelta(hours=1),
      outfile=outfile)

print("\n" + "="*60)
print("3. COMPARISON RESULTS (T+24 Hours)")
print("="*60)

# Load Custom Data
final_lats_custom = cloud_custom.lats
final_lons_custom = cloud_custom.lons

# Load OpenDrift Data
ds = xr.open_dataset(outfile)
lons_od = ds.lon.values
lats_od = ds.lat.values

# Filter out missing_data (NaNs) in OpenDrift
final_lats_od = [lats_od[i, ~np.isnan(lats_od[i])][-1] for i in range(len(lats_od))]
final_lons_od = [lons_od[i, ~np.isnan(lons_od[i])][-1] for i in range(len(lons_od))]

print(f"STARTING POSITION: Lat {start_lat:.4f}, Lon {start_lon:.4f}")

print(f"\n[ CUSTOM MODEL ]")
print(f"  Final Centroid: Lat {np.mean(final_lats_custom):.4f}, Lon {np.mean(final_lons_custom):.4f}")
print(f"  Spread (StdDev): Lat {np.std(final_lats_custom):.5f}°, Lon {np.std(final_lons_custom):.5f}°")
print(f"  Net Distance Traveled: {np.mean(final_lats_custom) - start_lat:.4f}° Lat, {np.mean(final_lons_custom) - start_lon:.4f}° Lon")

print(f"\n[ OPENDRIFT MODEL ]")
print(f"  Final Centroid: Lat {np.mean(final_lats_od):.4f}, Lon {np.mean(final_lons_od):.4f}")
print(f"  Spread (StdDev): Lat {np.std(final_lats_od):.5f}°, Lon {np.std(final_lons_od):.5f}°")
print(f"  Net Distance Traveled: {np.mean(final_lats_od) - start_lat:.4f}° Lat, {np.mean(final_lons_od) - start_lon:.4f}° Lon")

print("\n[ ANALYSIS ]")
lat_diff = abs(np.mean(final_lats_custom) - np.mean(final_lats_od))
lon_diff = abs(np.mean(final_lons_custom) - np.mean(final_lons_od))
print(f"  Difference in trajectory centroid: {lat_diff:.4f}° Lat, {lon_diff:.4f}° Lon")

