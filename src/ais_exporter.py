import xarray as xr
import numpy as np
import json
import logging
import os

logger = logging.getLogger(__name__)

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between a point and an array of points.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c

def generate_ais_summary(nc_filepath, output_filepath):
    """
    Reads OpenDrift NetCDF output and generates a simple JSON sequence
    of centroids and uncertainty radii for the AIS attribution team.
    """
    logger.info(f"Extracting AIS summary from {nc_filepath}")
    ds = xr.open_dataset(nc_filepath)

    lons = ds.lon.values  # shape: (particles, time)
    lats = ds.lat.values
    times = ds.time.values
    statuses = ds.status.values

    ais_data = []

    for t_idx, t_val in enumerate(times):
        valid_mask = (statuses[:, t_idx] == 0)
        valid_lons = lons[valid_mask, t_idx]
        valid_lats = lats[valid_mask, t_idx]

        if len(valid_lons) == 0:
            continue

        # Calculate centroid (mean position)
        center_lon = float(np.mean(valid_lons))
        center_lat = float(np.mean(valid_lats))

        # Calculate distances of all particles from centroid
        distances_km = haversine_distance(center_lon, center_lat, valid_lons, valid_lats)
        
        # Uncertainty radius could be the maximum distance, or a 95th percentile
        # Using 95th percentile removes extreme outliers that might skew the search area
        uncertainty_radius_km = float(np.percentile(distances_km, 95))
        
        # Ensure minimum uncertainty radius (e.g., 1.0 km) to account for inherent model error
        uncertainty_radius_km = max(1.0, uncertainty_radius_km)

        time_str = str(t_val).split('.')[0] + "Z"

        ais_data.append({
            "time": time_str,
            "predicted_lat": round(center_lat, 6),
            "predicted_lon": round(center_lon, 6),
            "uncertainty_radius_km": round(uncertainty_radius_km, 2)
        })

    with open(output_filepath, 'w') as f:
        json.dump(ais_data, f, indent=2)

    logger.info(f"AIS-ready summary saved to {output_filepath}")
    return output_filepath
