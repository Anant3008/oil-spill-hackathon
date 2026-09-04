import xarray as xr
import numpy as np

ds = xr.open_dataset('output/env_forcing.nc')
print("Checking for NaNs in the dataset arrays:")

for var in ['x_wind', 'y_wind', 'x_sea_water_velocity', 'y_sea_water_velocity']:
    has_nan = np.isnan(ds[var].values).any()
    print(f"{var} has NaN values: {has_nan}")
    if has_nan:
        nan_count = np.isnan(ds[var].values).sum()
        total = ds[var].values.size
        print(f"  {nan_count} out of {total} values are NaN!")

