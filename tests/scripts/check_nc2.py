import xarray as xr
import numpy as np

ds = xr.open_dataset('output/spatial_integration_test.nc')
status = ds.status.values

# How many particles have status != 0 (active)?
deactivated = np.sum(status[:, -1] != 0)
print(f"Total deactivated particles by end of run: {deactivated} out of {status.shape[0]}")

# Did they hit land or missing_data?
print("Statuses:", np.unique(status[:, -1]))
