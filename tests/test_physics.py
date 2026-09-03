from physics import calculate_oil_velocity

def test_calculate_oil_velocity():
    # Test case 1: Standard values
    u_w, v_w = 10.0, 10.0
    u_c, v_c = 1.0, 1.0
    windage = 0.035
    
    u_oil, v_oil = calculate_oil_velocity(u_w, v_w, u_c, v_c, windage)
    
    # Expected: (10 * 0.035) + 1.0 = 0.35 + 1.0 = 1.35
    assert round(u_oil, 3) == 1.350
    assert round(v_oil, 3) == 1.350

def test_no_wind():
    # Test case 2: No wind, drift should exactly match ocean current
    u_oil, v_oil = calculate_oil_velocity(0.0, 0.0, 1.5, -0.5)
    assert u_oil == 1.5
    assert v_oil == -0.5

def test_no_current():
    # Test case 3: No current, drift should be pure windage
    u_oil, v_oil = calculate_oil_velocity(100.0, -100.0, 0.0, 0.0, windage=0.03)
    assert round(u_oil, 2) == 3.00
    assert round(v_oil, 2) == -3.00
