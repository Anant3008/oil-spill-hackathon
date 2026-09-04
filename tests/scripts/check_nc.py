import xarray as xr
ds = xr.open_dataset('output/spatial_integration_test.nc')
status = ds.status.values
lons = ds.lon.values
lats = ds.lat.values
times = ds.time.values

# Check status of particle 0 over time
print("Particle 0 status over 24 hours:")
for i in range(len(times)):
    print(f"Hour {i}: status={status[0, i]}, lon={lons[0, i]:.4f}, lat={lats[0, i]:.4f}")
