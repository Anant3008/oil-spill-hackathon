import numpy as np

def calculate_oil_velocity(u_wind, v_wind, u_current, v_current, windage=0.035):
    """
    Calculates the final oil velocity vector combining windage and ocean currents.
    
    Args:
        u_wind (float or np.ndarray): East-West wind velocity.
        v_wind (float or np.ndarray): North-South wind velocity.
        u_current (float or np.ndarray): East-West ocean current velocity.
        v_current (float or np.ndarray): North-South ocean current velocity.
        windage (float): Configurable windage factor. Default 0.035 (3.5%).
        
    Returns:
        tuple: (u_oil, v_oil) representing the resulting oil velocity.
    """
    u_oil = (u_wind * windage) + u_current
    v_oil = (v_wind * windage) + v_current
    
    return u_oil, v_oil

if __name__ == "__main__":
    # Test with sample values
    u_w, v_w = 10.0, -5.0   # Wind blowing 10 m/s East, 5 m/s South
    u_c, v_c = 0.5, 0.2     # Current flowing 0.5 m/s East, 0.2 m/s North
    
    u_oil, v_oil = calculate_oil_velocity(u_w, v_w, u_c, v_c)
    
    print("=== Oil Velocity Calculation ===")
    print(f"Wind:    U={u_w:5.1f}, V={v_w:5.1f}")
    print(f"Current: U={u_c:5.1f}, V={v_c:5.1f}")
    print(f"Result:  U={u_oil:5.3f}, V={v_oil:5.3f} (using 3.5% windage)")
