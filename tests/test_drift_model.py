import numpy as np
from src.drift_model import move_particles, ParticleCloud, EARTH_RADIUS_M

def test_move_particles_array_north():
    # Test vectorized movement of 3 particles north simultaneously
    lats = np.array([0.0, 10.0, 20.0])
    lons = np.array([0.0, 0.0, 0.0])
    
    dy_m = (np.pi / 180.0) * EARTH_RADIUS_M
    dt_seconds = 3600.0
    v_oil_ms = dy_m / dt_seconds # Speed needed to travel exactly 1 degree North in 1 hour
    
    new_lats, new_lons = move_particles(lats, lons, 0.0, v_oil_ms, dt_seconds)
    
    # All latitudes should increase exactly by 1.0
    np.testing.assert_almost_equal(new_lats, np.array([1.0, 11.0, 21.0]))
    np.testing.assert_almost_equal(new_lons, np.array([0.0, 0.0, 0.0]))

def test_particle_cloud_initialization():
    cloud = ParticleCloud(start_lat=10.0, start_lon=20.0, num_particles=100)
    assert len(cloud.lats) == 100
    assert len(cloud.lons) == 100
    
    # Centroid should be very close to origin (due to normal dist)
    assert abs(np.mean(cloud.lats) - 10.0) < 0.1
    assert abs(np.mean(cloud.lons) - 20.0) < 0.1
    
    # History should contain exactly the initial state
    assert len(cloud.history_lats) == 1
    
def test_particle_cloud_step():
    cloud = ParticleCloud(start_lat=0.0, start_lon=0.0, num_particles=50)
    
    # Move north for 1 step
    dy_m = (np.pi / 180.0) * EARTH_RADIUS_M
    v_oil_ms = dy_m / 3600.0
    cloud.step(0.0, v_oil_ms, dt_seconds=3600)
    
    assert len(cloud.history_lats) == 2
    # The new centroid latitude should have increased by ~1.0
    assert abs(np.mean(cloud.history_lats[-1]) - np.mean(cloud.history_lats[0]) - 1.0) < 0.001
