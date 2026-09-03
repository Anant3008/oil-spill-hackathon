from vector_math import speed_dir_to_uv

def test_ocean_current_vectors():
    # Ocean current heading North (0 deg) -> Positive V, Zero U
    u, v = speed_dir_to_uv(1.0, 0, is_wind=False)
    assert u == 0.0 and v == 1.0

    # Ocean current heading East (90 deg) -> Positive U, Zero V
    u, v = speed_dir_to_uv(1.0, 90, is_wind=False)
    assert u == 1.0 and v == 0.0

    # Ocean current heading South (180 deg) -> Negative V, Zero U
    u, v = speed_dir_to_uv(1.0, 180, is_wind=False)
    assert u == 0.0 and v == -1.0

    # Ocean current heading West (270 deg) -> Negative U, Zero V
    u, v = speed_dir_to_uv(1.0, 270, is_wind=False)
    assert u == -1.0 and v == 0.0

def test_wind_vectors():
    # Wind from North (0 deg) -> Blows towards South -> Negative V, Zero U
    u, v = speed_dir_to_uv(1.0, 0, is_wind=True)
    assert u == 0.0 and v == -1.0

    # Wind from East (90 deg) -> Blows towards West -> Negative U, Zero V
    u, v = speed_dir_to_uv(1.0, 90, is_wind=True)
    assert u == -1.0 and v == 0.0

    # Wind from South (180 deg) -> Blows towards North -> Positive V, Zero U
    u, v = speed_dir_to_uv(1.0, 180, is_wind=True)
    assert u == 0.0 and v == 1.0

    # Wind from West (270 deg) -> Blows towards East -> Positive U, Zero V
    u, v = speed_dir_to_uv(1.0, 270, is_wind=True)
    assert u == 1.0 and v == 0.0
