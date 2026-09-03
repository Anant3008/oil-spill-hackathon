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
        u_oil_ms (float or np.ndarray): East-West oil velocity in m/s.
        v_oil_ms (float or np.ndarray): North-South oil velocity in m/s.
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
    def __init__(self, start_lat, start_lon, num_particles=1000, spread_deg=0.001, diffusion_k=10.0):
        """
        Initializes a cloud of particles with a random Gaussian spatial spread.
        
        Args:
            start_lat (float): Initial latitude.
            start_lon (float): Initial longitude.
            num_particles (int): Number of virtual particles.
            spread_deg (float): Initial standard deviation of particle positions in degrees.
            diffusion_k (float): Diffusion coefficient in m^2/s. Realistic ocean values are 1-100.
        """
        self.num_particles = num_particles
        self.diffusion_k = diffusion_k
        self.lats = np.random.normal(start_lat, spread_deg, num_particles)
        self.lons = np.random.normal(start_lon, spread_deg, num_particles)
        
        # Store history of positions: lists of arrays [time_step][particle_array]
        self.history_lats = [self.lats.copy()]
        self.history_lons = [self.lons.copy()]
        
    def step(self, u_oil_ms, v_oil_ms, dt_seconds=3600):
        """
        Advances the particle cloud by one timestep, applying deterministic drift and random diffusion.
        """
        # Calculate standard deviation of velocity for the random walk diffusion.
        # Variance of position (dx) = 2 * K * dt
        # dx = v * dt, so Variance of v = (2 * K) / dt
        # Therefore, std_dev(v) = sqrt(2 * K / dt)
        std_dev_v = np.sqrt(2 * self.diffusion_k / dt_seconds)
        
        # Generate random velocity perturbations for each particle
        u_diff = np.random.normal(0, std_dev_v, self.num_particles)
        v_diff = np.random.normal(0, std_dev_v, self.num_particles)
        
        # Total velocity = deterministic drift + random diffusion
        total_u = u_oil_ms + u_diff
        total_v = v_oil_ms + v_diff
        
        # Move particles using the combined velocities
        self.lats, self.lons = move_particles(self.lats, self.lons, total_u, total_v, dt_seconds)
        
        self.history_lats.append(self.lats.copy())
        self.history_lons.append(self.lons.copy())

def run_cloud_simulation(start_lat, start_lon, num_particles=1000, hours=6):
    """
    Simulates a particle cloud over multiple hours using live environmental data and diffusion.
    """
    logging.getLogger('src.data_fetcher').setLevel(logging.WARNING)
    
    df = get_environmental_data(start_lat, start_lon)
    
    print("\n" + "="*70)
    print(f"=== Phase 6: Particle Cloud with Diffusion ({num_particles} pts, {hours}h) ===")
    print("="*70)
    
    # K=10 m^2/s is a standard starting point for ocean surface diffusion
    cloud = ParticleCloud(start_lat, start_lon, num_particles=num_particles, diffusion_k=10.0)
    print(f"Initialized {num_particles} particles at ({start_lat}, {start_lon})")
    
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
        
        # Calculate statistics
        centroid_lat = np.mean(cloud.lats)
        centroid_lon = np.mean(cloud.lons)
        
        # Measure spreading (approximate distance in degrees)
        lat_spread = np.std(cloud.lats)
        lon_spread = np.std(cloud.lons)
        
        print(f"Hour {hour+1:02d} | Drift: U {u_oil:5.2f} V {v_oil:5.2f} | "
              f"Centroid ({centroid_lat:.5f}, {centroid_lon:.5f}) | "
              f"Spread (std): {lat_spread:.5f}°")

    print("="*70 + "\n")
    return cloud

if __name__ == "__main__":
    # Test 1000 particles over 6 hours
    run_cloud_simulation(19.07, 72.88, num_particles=1000, hours=6)
