import numpy as np

def speed_dir_to_uv(speed, direction_deg, is_wind=False):
    """
    Converts speed and direction (degrees clockwise from North) into U (East) and V (North) vectors.
    
    Args:
        speed (float or np.ndarray): Magnitude of the flow.
        direction_deg (float or np.ndarray): Direction in degrees (0=North, 90=East, 180=South, 270=West).
        is_wind (bool): If True, treats direction as meteorological (where it comes FROM).
                        If False, treats direction as oceanographic (where it goes TOWARDS).
                        
    Returns:
        tuple: (u, v) where u is East-West velocity (+ is East), v is North-South velocity (+ is North).
    """
    # Meteorological wind is where it comes FROM. We add 180 to get where it goes TOWARDS.
    if is_wind:
        direction_deg = direction_deg + 180.0
        
    # Standard math angle is counter-clockwise from East (x-axis).
    # Geographic angle is clockwise from North (y-axis).
    # Conversion: math_angle = 90 - geographic_angle
    math_angle_rad = np.radians(90.0 - direction_deg)
    
    u = speed * np.cos(math_angle_rad)
    v = speed * np.sin(math_angle_rad)
    
    # Clean up extremely small floating point artifacts (like 6.12e-17 -> 0.0)
    u = np.round(u, 10)
    v = np.round(v, 10)
    
    return u, v

if __name__ == "__main__":
    test_directions = [0, 90, 180, 270]
    speed = 1.0 # arbitrary speed for testing
    
    print("=== Ocean Current Convention (Where it goes TOWARDS) ===")
    for d in test_directions:
        u, v = speed_dir_to_uv(speed, d, is_wind=False)
        print(f"Dir: {d:3d}° -> U (East): {u:4.1f}, V (North): {v:4.1f}")
        
    print("\n=== Meteorological Wind Convention (Where it comes FROM) ===")
    for d in test_directions:
        u, v = speed_dir_to_uv(speed, d, is_wind=True)
        print(f"Dir: {d:3d}° -> U (East): {u:4.1f}, V (North): {v:4.1f}")
