import logging
from datetime import datetime, timedelta, timezone
import pandas as pd
import os

from environment import EnvironmentManager
from drift_model import SpillModel
from exporter import generate_geojson_polygons
from ais_exporter import generate_ais_summary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_backward_hindcasting(observed_lat, observed_lon, observation_time_str, backward_duration_hours=24):
    """
    Executes Part 4: Backward Hindcasting
    Input: observed spill location, detection time
    Output: predicted origin probability region over time (GeoJSON polygons)
    """
    logger.info("="*60)
    logger.info(f"BACKWARD HINDCASTING INITIATED")
    logger.info(f"Observed Location: {observed_lat}, {observed_lon}")
    logger.info(f"Observation Time: {observation_time_str}")
    logger.info("="*60)

    # 1. Parse observation time
    obs_dt = pd.to_datetime(observation_time_str).tz_localize(None)
    end_date_str = obs_dt.strftime('%Y-%m-%d')

    # We fetch a few days BEFORE the observation to cover the backward drift
    start_dt = obs_dt - pd.Timedelta(hours=backward_duration_hours + 48)
    start_date_str = start_dt.strftime('%Y-%m-%d')

    # 2. Fetch Environment Data
    env_manager = EnvironmentManager()

    # Pre-flight check: shift the grid center backwards along the expected origin path
    opt_lat, opt_lon, opt_step = env_manager.calculate_shifted_grid(
        start_lat=observed_lat,
        start_lon=observed_lon,
        start_date=start_date_str,
        end_date=end_date_str,
        duration_hours=backward_duration_hours,
        is_backward=True
    )

    # Request a dynamically shifted grid at the optimal resolution
    reader_env, _ = env_manager.add_openmeteo_grid(
        center_lat=opt_lat,
        center_lon=opt_lon,
        step_deg=opt_step
    )

    # 3. Run OpenOil BACKWARD Simulation
    model = SpillModel(
        start_lat=observed_lat,
        start_lon=observed_lon,
        start_time=obs_dt,
        oil_type='GENERIC HEAVY CRUDE'
    )
    model.add_environment_readers([reader_env])

    nc_output = "output/backward_hindcast.nc"

    model.run_simulation(
        duration_hours=backward_duration_hours,
        time_step_hours=-1, # <--- NEGATIVE TIMESTEP FOR HINDCASTING
        num_particles=1000,
        radius_m=2500, # Initial larger radius representing the uncertainty of the observed slick
        outfile=nc_output
    )

    # 4. Export Probability Regions Over Time
    geojson_output = "output/origin_probability_regions.geojson"
    generate_geojson_polygons(nc_output, geojson_output, is_backward=True)

    # 5. Export AIS-ready summary
    ais_output = "output/ais_vessel_search_areas.json"
    generate_ais_summary(nc_output, ais_output)

    logger.info("="*60)
    logger.info("HINDCASTING COMPLETE")
    logger.info(f"Visual polygon output available at: {geojson_output}")
    logger.info(f"AIS Vessel attribution input available at: {ais_output}")
    logger.info("The final polygon in the GeoJSON represents the highest-probability region of origin.")
    logger.info("="*60)

    return geojson_output, ais_output

if __name__ == "__main__":
    # Example scenario: Ship reports oil spill today
    obs_time = (datetime.now(timezone.utc).replace(tzinfo=None)).replace(hour=12, minute=0, second=0, microsecond=0)

    run_backward_hindcasting(
        observed_lat=18.5,
        observed_lon=71.5,
        observation_time_str=obs_time.isoformat() + "Z",
        backward_duration_hours=24
    )