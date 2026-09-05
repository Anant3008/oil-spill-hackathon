import logging
from datetime import datetime, timedelta, timezone
import pandas as pd
import os

from environment import EnvironmentManager
from drift_model import SpillModel
from exporter import generate_geojson_polygons

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_forward_prediction(spill_lat, spill_lon, detection_time_str, duration_hours=24, radius_m=1000, oil_type='GENERIC HEAVY CRUDE'):
    """
    Executes Part 3: Forward Prediction
    Input: location, detection time, dynamic spill properties
    Output: predicted spill region over time (GeoJSON polygons)
    """
    logger.info("="*60)
    logger.info(f"FORWARD PREDICTION INITIATED")
    logger.info(f"Location: {spill_lat}, {spill_lon}")
    logger.info(f"Detection Time: {detection_time_str}")
    logger.info(f"Spill Radius: {radius_m}m | Oil Type: {oil_type}")
    logger.info("="*60)

    # 1. Parse detection time and format for Open-Meteo
    # Strip timezone info to match the naive UTC timestamps in our NetCDF arrays
    detect_dt = pd.to_datetime(detection_time_str).tz_localize(None)
    start_date_str = detect_dt.strftime('%Y-%m-%d')

    # We fetch a few days ahead to guarantee we cover the full duration + buffer
    end_dt = detect_dt + pd.Timedelta(hours=duration_hours + 48)
    end_date_str = end_dt.strftime('%Y-%m-%d')

    # 2. Fetch Environment Data
    env_manager = EnvironmentManager()

    # Pre-flight check: shift the grid center along the predicted drift path to save API payload
    opt_lat, opt_lon, opt_step = env_manager.calculate_shifted_grid(
        start_lat=spill_lat,
        start_lon=spill_lon,
        start_date=start_date_str,
        end_date=end_date_str,
        duration_hours=duration_hours,
        is_backward=False
    )

    # Request a dynamically shifted grid at the optimal resolution
    reader_env, _ = env_manager.add_openmeteo_grid(
        center_lat=opt_lat,
        center_lon=opt_lon,
        step_deg=opt_step
    )

    # 3. Run OpenOil Forward Simulation
    model = SpillModel(
        start_lat=spill_lat,
        start_lon=spill_lon,
        start_time=detect_dt,
        oil_type=oil_type
    )
    model.add_environment_readers([reader_env])

    nc_output = "output/forward_prediction.nc"

    # Dynamically scale particle count based on the physical size of the spill area
    # Base baseline of 1000, scaling up linearly with the radius to maintain particle density
    optimal_particles = max(1000, int(radius_m * 1.5))

    model.run_simulation(
        duration_hours=duration_hours,
        num_particles=optimal_particles,
        radius_m=radius_m,
        outfile=nc_output
    )

    # 4. Export Predicted Regions Over Time
    geojson_output = "output/predicted_regions.geojson"
    generate_geojson_polygons(nc_output, geojson_output)

    logger.info("="*60)
    logger.info("FORWARD PREDICTION COMPLETE")
    logger.info(f"Output available at: {geojson_output}")
    logger.info("="*60)

    return geojson_output

if __name__ == "__main__":
    # Example scenario: Ship reports oil spill today
    # Using 12:00 PM UTC yesterday to ensure API data is fully available
    # Using replace(tzinfo=None) to ensure the initial dummy string mimics a naive or aware str
    detect_time = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)

    run_forward_prediction(
        spill_lat=18.5,
        spill_lon=71.5,
        detection_time_str=detect_time.isoformat() + "Z", # Simulating an aware ISO string from a frontend API
        duration_hours=24,
        radius_m=1200,               # Dynamic parameter: user/satellite estimated radius
        oil_type='GENERIC HEAVY CRUDE' # Dynamic parameter: user selected oil type
    )