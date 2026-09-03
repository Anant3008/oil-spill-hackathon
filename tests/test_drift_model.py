import numpy as np
from src.drift_model import move_particle, EARTH_RADIUS_M

def test_move_particle_north():
    # If a particle moves North at enough speed to cover exactly 1 degree of latitude
    # 1 degree of latitude in meters is exactly (pi / 180) * EARTH_RADIUS_M
    dy_m = (np.pi / 180.0) * EARTH_RADIUS_M
    dt_seconds = 3600.0
    v_oil_ms = dy_m / dt_seconds # Speed needed to travel exactly 1 degree North in 1 hour
    
    new_lat, new_lon = move_particle(0.0, 0.0, 0.0, v_oil_ms, dt_seconds)
    
    # Should be exactly at Lat 1.0, Lon 0.0
    assert round(new_lat, 5) == 1.00000
    assert round(new_lon, 5) == 0.00000

def test_move_particle_east_equator():
    # Similar test for East at the equator where cos(lat) == 1
    dx_m = (np.pi / 180.0) * EARTH_RADIUS_M
    dt_seconds = 3600.0
    u_oil_ms = dx_m / dt_seconds
    
    new_lat, new_lon = move_particle(0.0, 0.0, u_oil_ms, 0.0, dt_seconds)
    
    assert round(new_lat, 5) == 0.00000
    assert round(new_lon, 5) == 1.00000

def test_move_particle_east_high_latitude():
    # At latitude 60, cos(60 deg) is 0.5. 
    # Covering the same dx_m should result in TWICE the degree change in longitude.
    dx_m = (np.pi / 180.0) * EARTH_RADIUS_M
    dt_seconds = 3600.0
    u_oil_ms = dx_m / dt_seconds
    
    new_lat, new_lon = move_particle(60.0, 0.0, u_oil_ms, 0.0, dt_seconds)
    
    assert round(new_lat, 5) == 60.00000
    assert round(new_lon, 5) == 2.00000
