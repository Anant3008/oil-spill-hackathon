import os
import json
import numpy as np
import logging
from data_fetcher import get_environmental_data
from vector_math import speed_dir_to_uv
from physics import calculate_oil_velocity

logger = logging.getLogger(__name__)

EARTH_RADIUS_M = 6371000.0

def move_particles(lats, lons, u_oil_ms, v_oil_ms, dt_seconds=3600):
    dx_m = u_oil_ms * dt_seconds
    dy_m = v_oil_ms * dt_seconds
    
    dlat = (dy_m / EARTH_RADIUS_M) * (180.0 / np.pi)
    lat_rad = np.radians(lats)
    dlon = (dx_m / (EARTH_RADIUS_M * np.cos(lat_rad))) * (180.0 / np.pi)
    
    return lats + dlat, lons + dlon

class ParticleCloud:
    def __init__(self, start_lat, start_lon, num_particles=1000, spread_deg=0.001, diffusion_k=0.0):
        self.num_particles = num_particles
        self.diffusion_k = diffusion_k
        self.lats = np.random.normal(start_lat, spread_deg, num_particles)
        self.lons = np.random.normal(start_lon, spread_deg, num_particles)
        
        self.history_lats = [self.lats.copy()]
        self.history_lons = [self.lons.copy()]
        
    def step(self, u_oil_ms, v_oil_ms, dt_seconds=3600):
        if self.diffusion_k > 0.0:
            std_dev_v = np.sqrt(2 * self.diffusion_k / dt_seconds)
            u_diff = np.random.normal(0, std_dev_v, self.num_particles)
            v_diff = np.random.normal(0, std_dev_v, self.num_particles)
            total_u = u_oil_ms + u_diff
            total_v = v_oil_ms + v_diff
        else:
            total_u = u_oil_ms
            total_v = v_oil_ms
            
        self.lats, self.lons = move_particles(self.lats, self.lons, total_u, total_v, dt_seconds)
        
        self.history_lats.append(self.lats.copy())
        self.history_lons.append(self.lons.copy())

def get_env_for_particles(lats, lons, current_hour_data):
    num_particles = len(lats)
    w_spd = np.full(num_particles, current_hour_data["wind_speed_10m"] / 3.6)
    w_dir = np.full(num_particles, current_hour_data["wind_direction_10m"])
    c_spd = np.full(num_particles, current_hour_data["ocean_current_velocity"] / 3.6)
    c_dir = np.full(num_particles, current_hour_data["ocean_current_direction"])
    return w_spd, w_dir, c_spd, c_dir

def run_cloud_simulation(start_lat, start_lon, num_particles=1000, hours=24, diffusion_k=0.0):
    logging.getLogger('src.data_fetcher').setLevel(logging.WARNING)
    
    df = get_environmental_data(start_lat, start_lon)
    
    print("\n" + "="*70)
    print(f"=== Phase 7: 24-Hour Simulation ({num_particles} pts, D={diffusion_k}) ===")
    print("="*70)
    
    cloud = ParticleCloud(start_lat, start_lon, num_particles=num_particles, diffusion_k=diffusion_k)
    
    checkpoints = {}
    checkpoint_hours = {0, 6, 12, 18, 24}
    
    def get_summary_str(lats, lons):
        return (f"Centroid: ({np.mean(lats):.4f}, {np.mean(lons):.4f}), "
                f"Spread: {np.std(lats):.5f}°")
    
    t0_time = str(df.iloc[0]["time"])
    checkpoints["T+0"] = {
        "timestamp": t0_time,
        "lats": cloud.lats.tolist(),
        "lons": cloud.lons.tolist()
    }
    print(f"Checkpoint saved: T+0  (Time: {t0_time}) | {get_summary_str(cloud.lats, cloud.lons)}")
    
    for hour in range(hours):
        row = df.iloc[hour]
        
        w_spd, w_dir, c_spd, c_dir = get_env_for_particles(cloud.lats, cloud.lons, row)
        
        u_wind, v_wind = speed_dir_to_uv(w_spd, w_dir, is_wind=True)
        u_current, v_current = speed_dir_to_uv(c_spd, c_dir, is_wind=False)
        u_oil, v_oil = calculate_oil_velocity(u_wind, v_wind, u_current, v_current, windage=0.035)
        
        cloud.step(u_oil, v_oil, dt_seconds=3600)
        
        current_t = hour + 1
        if current_t in checkpoint_hours:
            label = f"T+{current_t}"
            timestamp_str = str(df.iloc[current_t]["time"]) if current_t < len(df) else "End of Simulation"
            
            checkpoints[label] = {
                "timestamp": timestamp_str,
                "lats": cloud.lats.tolist(),
                "lons": cloud.lons.tolist()
            }
            
            print(f"Checkpoint saved: {label:<4} (Time: {timestamp_str}) | {get_summary_str(cloud.lats, cloud.lons)}")

    os.makedirs("output", exist_ok=True)
    output_path = "output/trajectory_checkpoints.json"
    with open(output_path, "w") as f:
        json.dump(checkpoints, f)
        
    print(f"\nSimulation complete. Full particle datasets saved to {output_path}")
    print("="*70 + "\n")
    return cloud

if __name__ == "__main__":
    start_lat, start_lon = 19.07, 72.88
    # Test 1000 particles over 24 hours with zero diffusion
    cloud = run_cloud_simulation(start_lat, start_lon, num_particles=1000, hours=24, diffusion_k=0.0)
    
    # Phase 8: Generate Animation
    from visualizer import create_animation
    create_animation(cloud, start_lat, start_lon)
