# 🌊 Advanced Oil Spill Trajectory & Hindcasting Model

This repository contains the advanced physics-based oil spill drift and trajectory model developed for **SIH Problem Statement 26143**. 

Built upon the robust [OpenDrift / OpenOil](https://github.com/OpenDrift/opendrift) framework, this system provides highly localized, API-optimized marine simulations capable of predicting where an oil spill will go, and critically, **where it came from**.

---

## ✨ Key Features

* 🔮 **Forward Forecasting:** Predicts the future trajectory and spreading footprint of a newly detected oil spill.
* ⏪ **Backward Hindcasting:** Tracks an observed slick backward in time to generate a "Probability of Origin" zone, explicitly avoiding inaccurate single-point origin claims.
* 🧠 **Adaptive Spatial Coverage:** Uses a lightweight "pre-flight" mathematical check (combining ocean current vectors with 3% wind drift) to dynamically shift bounding boxes down the drift path. This prevents the spill from hitting "invisible walls" while keeping the API payload extremely small.
* 📡 **AIS-Ready Vessel Attribution:** Condenses complex particle clouds into simple, actionable targets (Centroid Lat/Lon + Uncertainty Radius in km) ready for immediate cross-referencing with AIS transponder databases.
* 🗺️ **GeoJSON Integration:** Automatically wraps particle clouds into time-sequenced Convex Hull polygons for seamless frontend mapping integration.

---

## 🏗️ System Architecture & Workflow

The system is highly modular, separating the environmental data fetching from the actual physics simulation.

1. **Input Parameters:** The system accepts the spill coordinates, detection time, oil type (e.g., Generic Heavy Crude), and an initial uncertainty radius.
2. **Environmental Pre-Flight (`EnvironmentManager`):** Queries the Open-Meteo API for a single point to establish the general drift vector for the region.
3. **Optimized Grid Fetch:** Calculates a shifted bounding box and dynamically scales the resolution (capped at 0.08° / ~8km) to download a highly localized NetCDF array containing:
   * Ocean Current Velocity & Direction
   * Wind Speed & Direction (10m)
   * Wave Height
4. **Physics Engine (`SpillModel`):** Seeds optimal particle counts (auto-scaling based on the spill radius) and runs the `OpenOil` 2D surface model (forward or backward in time).
5. **Exporter Pipeline:**
   * `exporter.py`: Drops inactive particles and computes the `scipy.spatial.ConvexHull` polygon for each hour, exporting to `predicted_regions.geojson`.
   * `ais_exporter.py`: Calculates the spatial centroid and the 95th-percentile Haversine distance to generate `ais_vessel_search_areas.json`.

---

## 🚀 Setup & Installation

Because OpenDrift relies on complex C-libraries for spatial operations, using `conda` or `mamba` is highly recommended.

### 1. Create the Environment
```bash
# It is recommended to use mamba for faster dependency resolution
mamba create -n opendrift python=3.12
mamba activate opendrift
```

### 2. Install Dependencies
```bash
# Install core dependencies (OpenDrift, xarray, pandas, scipy, requests)
pip install -r requirements.txt

# Install tabulate for the validation script report
pip install tabulate
```

---

## 💻 Usage

Make sure you export the project root to your Python path before running the scripts:
```bash
export PYTHONPATH=$(pwd)
```

### 1. Run Forward Prediction
Simulates a spill moving forward in time.
```bash
python src/forward_predictor.py
```
**Outputs:** 
* `output/forward_prediction.nc` (Raw OpenDrift Data)
* `output/predicted_regions.geojson` (Frontend polygons)

### 2. Run Backward Hindcasting
Simulates an observed spill moving backward in time to find its source.
```bash
python src/hindcaster.py
```
**Outputs:** 
* `output/backward_hindcast.nc`
* `output/origin_probability_regions.geojson`
* `output/ais_vessel_search_areas.json` (AIS Vessel target list)

### 3. Visualizer
A lightweight native Matplotlib script to animate the generated GeoJSON files without needing a frontend map.
```bash
# To test both Forward and Backward visualizers sequentially:
python src/visualize_geojson.py
```

### 4. Automated 10-Location Validation
Runs a comprehensive "round-trip" test across 10 diverse Indian coastal locations (Mumbai, Chennai, Kochi, etc.). It simulates a forward spill for 24 hours, then uses the final location to hindcast backward, validating if the true origin was captured within the resulting AIS uncertainty radius.
```bash
python tests/scripts/validate_locations.py
```
*(Note: This script includes 30-second sleep intervals to respect Open-Meteo's free API rate limits).*

---

## 🔮 Future Enhancements (Production Readiness)
* **API Wrapper:** Exposing the `run_forward_prediction` and `run_backward_hindcasting` functions via FastAPI/Flask.
* **INCOIS Integration:** Swapping the Open-Meteo API wrapper for direct local ingestion of daily INCOIS/Copernicus NetCDF files for zero-latency, enterprise-grade data.
* **3D Weathering:** Enabling full 3D vertical mixing and chemical weathering properties for specific oil profiles.