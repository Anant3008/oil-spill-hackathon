import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import logging
import json
import pandas as pd
from datetime import datetime, timedelta, timezone

from src.forward_predictor import run_forward_prediction
from src.hindcaster import run_backward_hindcasting
from src.ais_exporter import haversine_distance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 10 Diverse Locations around the Indian Coastline
TEST_LOCATIONS = [
    {"name": "Mumbai Coast", "lat": 18.92, "lon": 72.82},
    {"name": "Chennai Port", "lat": 13.08, "lon": 80.30},
    {"name": "Kochi", "lat": 9.93, "lon": 76.25},
    {"name": "Visakhapatnam", "lat": 17.68, "lon": 83.21},
    {"name": "Paradip", "lat": 20.26, "lon": 86.67},
    {"name": "Gulf of Kutch", "lat": 22.50, "lon": 69.50},
    {"name": "Port Blair (Andaman)", "lat": 11.62, "lon": 92.73},
    {"name": "Mangalore", "lat": 12.87, "lon": 74.83},
    {"name": "Haldia (Hooghly River Mouth)", "lat": 21.60, "lon": 88.00},
    {"name": "Tuticorin", "lat": 8.76, "lon": 78.13}
]

def run_roundtrip_validation():
    duration_hours = 24
    # Use a fixed date in the recent past to ensure Open-Meteo has data
    base_time = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=2)).replace(hour=12, minute=0, second=0, microsecond=0)
    start_time_str = base_time.isoformat() + "Z"

    results = []

    for loc in TEST_LOCATIONS:
        logger.info(f"\n{'='*50}\nTESTING: {loc['name']} ({loc['lat']}, {loc['lon']})\n{'='*50}")
        try:
            # 1. Forward Prediction
            # Simulate a spill originating here
            logger.info("-> Running Forward Prediction...")
            run_forward_prediction(loc['lat'], loc['lon'], start_time_str, duration_hours)
            
            # Read the forward prediction AIS output to get the final location
            # (We need to modify forward_predictor to also output AIS JSON, or just use the generated GeoJSON,
            # wait, let's just generate the AIS summary for the forward run manually to find the centroid)
            from src.ais_exporter import generate_ais_summary
            forward_ais = "output/forward_prediction_ais.json"
            generate_ais_summary("output/forward_prediction.nc", forward_ais)
            
            with open(forward_ais, 'r') as f:
                forward_data = json.load(f)
                
            # Final point of the forward run is our "observed spill" 24 hours later
            final_forward = forward_data[-1]
            observed_lat = final_forward['predicted_lat']
            observed_lon = final_forward['predicted_lon']
            observation_time = final_forward['time']
            
            # 2. Backward Hindcasting
            # We found the spill here. Where did it come from?
            logger.info("-> Running Backward Hindcasting...")
            run_backward_hindcasting(observed_lat, observed_lon, observation_time, backward_duration_hours=duration_hours)
            
            # Read the hindcast AIS output
            backward_ais = "output/ais_vessel_search_areas.json"
            with open(backward_ais, 'r') as f:
                backward_data = json.load(f)
                
            # The final backward point (index -1) represents the predicted origin at T=0
            predicted_origin = backward_data[-1]
            pred_lat = predicted_origin['predicted_lat']
            pred_lon = predicted_origin['predicted_lon']
            uncertainty_radius = predicted_origin['uncertainty_radius_km']
            
            # 3. Validation Metric
            # Calculate distance between TRUE origin and PREDICTED origin
            error_distance_km = haversine_distance(loc['lon'], loc['lat'], pred_lon, pred_lat)
            
            # Success criterion: Is the TRUE origin within the PREDICTED uncertainty radius?
            is_captured = error_distance_km <= uncertainty_radius
            
            results.append({
                "Location": loc['name'],
                "Error Distance (km)": round(float(error_distance_km), 2),
                "Uncertainty Radius (km)": round(float(uncertainty_radius), 2),
                "Captured in Search Area": is_captured
            })
            
            logger.info(f"Result for {loc['name']}: Error = {error_distance_km:.2f}km | Captured = {is_captured}")

        except Exception as e:
            logger.error(f"Failed testing {loc['name']}: {e}")
            results.append({
                "Location": loc['name'],
                "Error Distance (km)": "ERROR",
                "Uncertainty Radius (km)": "ERROR",
                "Captured in Search Area": "ERROR"
            })

        # Sleep to avoid hitting API rate limits on Open-Meteo
        logger.info("Sleeping for 30 seconds to respect API rate limits...")
        time.sleep(30)

    # Print Summary Report
    df = pd.DataFrame(results)
    print("\n\n" + "="*60)
    print(" ROUND-TRIP VALIDATION REPORT ".center(60, "="))
    print("="*60)
    print(df.to_markdown(index=False))
    print("="*60)

if __name__ == "__main__":
    run_roundtrip_validation()
