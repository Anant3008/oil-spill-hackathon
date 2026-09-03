import numpy as np
import logging
from data_fetcher import get_environmental_data
from vector_math import speed_dir_to_uv
from physics import calculate_oil_velocity

logger = logging.getLogger(__name__)

# Mean Earth Radius in meters
EARTH_RADIUS_M = 6371000.0

def move_particle(lat, lon, u_oil_ms, v_oil_ms, dt_seconds=3600):
    """
    Moves a particle from (lat, lon) using the given oil velocity over a time step.
    Accounts for spherical earth (longitude distance changing with latitude).
    
    Args:
        lat (float): Current latitude in degrees.
        lon (float): Current longitude in degrees.
        u_oil_ms (float): East-West oil velocity in m/s.
        v_oil_ms (float): North-South oil velocity in m/s.
        dt_seconds (float): Time step in seconds. Default is 1 hour (3600s).
        
    Returns:
        tuple: (new_lat, new_lon)
    """
    # 1. Calculate distance traveled in meters
    dx_m = u_oil_ms * dt_seconds
    dy_m = v_oil_ms * dt_seconds
    
    # 2. Convert meters to latitude/longitude degrees
    # 1 degree of latitude is constant on a sphere
    dlat = (dy_m / EARTH_RADIUS_M) * (180.0 / np.pi)
    
    # 1 degree of longitude scales by the cosine of the latitude
    lat_rad = np.radians(lat)
    dlon = (dx_m / (EARTH_RADIUS_M * np.cos(lat_rad))) * (180.0 / np.pi)
    
    new_lat = lat + dlat
    new_lon = lon + dlon
    
    return new_lat, new_lon

def run_one_particle_simulation(start_lat, start_lon):
    """
    Fetches real API data and moves one particle for 1 hour.
    """
    # Disable excessive logging from data_fetcher for clean output
    logging.getLogger('src.data_fetcher').setLevel(logging.WARNING)
    
    # 1. Fetch data
    df = get_environmental_data(start_lat, start_lon)
    first_hour = df.iloc[0]
    
    # Open-Meteo returns speeds in km/h. Convert to m/s for physics (distance = v * t).
    wind_speed_ms = first_hour["wind_speed_10m"] / 3.6
    wind_dir = first_hour["wind_direction_10m"]
    
    current_speed_ms = first_hour["ocean_current_velocity"] / 3.6
    current_dir = first_hour["ocean_current_direction"]
    
    # 2. Convert speed/direction to U/V vectors
    u_wind, v_wind = speed_dir_to_uv(wind_speed_ms, wind_dir, is_wind=True)
    u_current, v_current = speed_dir_to_uv(current_speed_ms, current_dir, is_wind=False)
    
    # 3. Calculate final oil drift velocity
    u_oil, v_oil = calculate_oil_velocity(u_wind, v_wind, u_current, v_current, windage=0.035)
    
    # 4. Move the particle
    new_lat, new_lon = move_particle(start_lat, start_lon, u_oil, v_oil, dt_seconds=3600)
    
    # Print the results as requested
    print("\n" + "="*50)
    print("=== Phase 4: Single Particle Movement (1 Hour) ===")
    print("="*50)
    print(f"Start Position: Lat {start_lat:.6f}, Lon {start_lon:.6f}")
    print(f"Environmental Data used (Time: {first_hour['time']}):")
    print(f"  Wind: {wind_speed_ms:.2f} m/s, {wind_dir}°")
    print(f"  Curr: {current_speed_ms:.2f} m/s, {current_dir}°")
    print(f"Calculated Oil Velocity (m/s): U = {u_oil:.3f}, V = {v_oil:.3f}")
    print("-" * 50)
    print(f"End Position:   Lat {new_lat:.6f}, Lon {new_lon:.6f}")
    print("="*50 + "\n")
    
    return new_lat, new_lon

if __name__ == "__main__":
    # Test for problem statement requirement
    run_one_particle_simulation(19.07, 72.88)
