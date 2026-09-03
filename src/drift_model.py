import numpy as np
import logging
from data_fetcher import get_environmental_data
from vector_math import speed_dir_to_uv
from physics import calculate_oil_velocity

logger = logging.getLogger(__name__)

# Mean Earth Radius in meters
EARTH_RADIUS_M = 6371000.0

def move_particles(lats, lons, u_oil_ms, v_oil_ms, dt_seconds=3600):
    """
    Moves an array of particles using the given oil velocity over a time step.
    Vectorized with NumPy for maximum efficiency.
    
    Args:
        lats (np.ndarray): Array of current latitudes in degrees.
        lons (np.ndarray): Array of current longitudes in degrees.
        u_oil_ms (float): East-West oil velocity in m/s.
        v_oil_ms (float): North-South oil velocity in m/s.
        dt_seconds (float): Time step in seconds. Default is 1 hour (3600s).
        
    Returns:
        tuple: (new_lats, new_lons)
    """
    dx_m = u_oil_ms * dt_seconds
    dy_m = v_oil_ms * dt_seconds
    
    # 1 degree of latitude is constant
    dlat = (dy_m / EARTH_RADIUS_M) * (180.0 / np.pi)
    
    # 1 degree of longitude scales by the cosine of the latitude
    lat_rad = np.radians(lats)
    dlon = (dx_m / (EARTH_RADIUS_M * np.cos(lat_rad))) * (180.0 / np.pi)
    
    new_lats = lats + dlat
    new_lons = lons + dlon
    
    return new_lats, new_lons

class ParticleCloud:
    def __init__(self, start_lat, start_lon, num_particles=1000, spread_deg=0.001):
        """
        Initializes a cloud of particles with a random Gaussian spatial spread.
        """
        self.num_particles = num_particles
        self.lats = np.random.normal(start_lat, spread_deg, num_particles)
        self.lons = np.random.normal(start_lon, spread_deg, num_particles)
        
        # Store history of positions: lists of arrays [time_step][particle_array]
        self.history_lats = [self.lats.copy()]
        self.history_lons = [self.lons.copy()]
        
    def step(self, u_oil_ms, v_oil_ms, dt_seconds=3600):
        self.lats, self.lons = move_particles(self.lats, self.lons, u_oil_ms, v_oil_ms, dt_seconds)
        
        self.history_lats.append(self.lats.copy())
        self.history_lons.append(self.lons.copy())

def run_cloud_simulation(start_lat, start_lon, num_particles=1000, hours=6):
    """
    Simulates a particle cloud over multiple hours using live environmental data.
    """
    logging.getLogger('src.data_fetcher').setLevel(logging.WARNING)
    
    df = get_environmental_data(start_lat, start_lon)
    
    print("\n" + "="*65)
    print(f"=== Phase 5: Particle Cloud Simulation ({num_particles} pts, {hours}h) ===")
    print("="*65)
    
    cloud = ParticleCloud(start_lat, start_lon, num_particles=num_particles)
    print(f"Initialized {num_particles} particles at ({start_lat}, {start_lon}) with Gaussian spread.")
    
    for hour in range(hours):
        row = df.iloc[hour]
        
        wind_speed_ms = row["wind_speed_10m"] / 3.6
        wind_dir = row["wind_direction_10m"]
        current_speed_ms = row["ocean_current_velocity"] / 3.6
        current_dir = row["ocean_current_direction"]
        
        u_wind, v_wind = speed_dir_to_uv(wind_speed_ms, wind_dir, is_wind=True)
        u_current, v_current = speed_dir_to_uv(current_speed_ms, current_dir, is_wind=False)
        u_oil, v_oil = calculate_oil_velocity(u_wind, v_wind, u_current, v_current, windage=0.035)
        
        cloud.step(u_oil, v_oil, dt_seconds=3600)
        
        # Print centroid for logging
        centroid_lat = np.mean(cloud.lats)
        centroid_lon = np.mean(cloud.lons)
        print(f"Hour {hour+1:02d} | W: {wind_speed_ms:4.1f}m/s {wind_dir:3.0f}° | "
              f"C: {current_speed_ms:4.2f}m/s {current_dir:3.0f}° -> Centroid (Lat {centroid_lat:.5f}, Lon {centroid_lon:.5f})")

    print("="*65 + "\n")
    return cloud

if __name__ == "__main__":
    # Test 1000 particles over 6 hours as requested in Phase 5
    run_cloud_simulation(19.07, 72.88, num_particles=1000, hours=6)
