import xarray as xr
import numpy as np
import json
import logging
from scipy.spatial import ConvexHull

logger = logging.getLogger(__name__)

def generate_geojson_polygons(nc_filepath, geojson_filepath):
    """
    Reads OpenDrift NetCDF output and generates a GeoJSON FeatureCollection.
    Each feature is the convex hull polygon representing the spill region at a specific timestep.
    """
    logger.info(f"Extracting spill regions from {nc_filepath}")
    ds = xr.open_dataset(nc_filepath)
    
    lons = ds.lon.values  # shape: (particles, time)
    lats = ds.lat.values
    times = ds.time.values
    statuses = ds.status.values

    features = []

    # Iterate through each timestep
    for t_idx, t_val in enumerate(times):
        # Filter valid particles at this timestep (status == 0 means active)
        valid_mask = (statuses[:, t_idx] == 0)
        
        valid_lons = lons[valid_mask, t_idx]
        valid_lats = lats[valid_mask, t_idx]
        
        # We need at least 3 points to form a polygon
        if len(valid_lons) < 3:
            continue
            
        points = np.column_stack((valid_lons, valid_lats))
        
        try:
            hull = ConvexHull(points)
            # Get the vertices in counter-clockwise order to form the polygon
            hull_points = points[hull.vertices]
            # Close the polygon by appending the first point at the end
            hull_points = np.vstack((hull_points, hull_points[0]))
            
            # Convert to standard GeoJSON coordinates (Lon, Lat)
            coords = [[ [float(pt[0]), float(pt[1])] for pt in hull_points ]]
            
            # Ensure time is represented nicely
            time_str = str(t_val).split('.')[0]  # format: 'YYYY-MM-DDTHH:MM:SS'
            
            feature = {
                "type": "Feature",
                "properties": {
                    "time": time_str,
                    "timestep": t_idx,
                    "particle_count": len(valid_lons)
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": coords
                }
            }
            features.append(feature)
        except Exception as e:
            logger.warning(f"Could not create hull for timestep {t_idx}: {e}")

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(geojson_filepath, 'w') as f:
        json.dump(geojson, f, indent=2)
        
    logger.info(f"GeoJSON successfully saved to {geojson_filepath}")
    return geojson_filepath
