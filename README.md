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

## 🌐 FastAPI Server (`src/server.py`)

Exposes the two engines as HTTP endpoints for the React dashboard.

| Endpoint | Body fields | Returns |
|---|---|---|
| `GET /health` | — | `{ status, opendrift_available, note }` |
| `POST /forecast` | `spill_lat`, `spill_lon`, `detection_time` (ISO, optional → now), `duration_hours`, `radius_m`, `oil_type` | `{ success, mode, generated_at, geojson, ais_areas: null }` |
| `POST /hindcast` | `observed_lat`, `observed_lon`, `observation_time` (ISO, optional → now), `backward_duration_hours`, `radius_m`, `oil_type` | `{ success, mode, generated_at, geojson, ais_areas: [...] }` |

`geojson` is a `FeatureCollection` of per-timestep spill/origin polygons (fed straight into MapLibre/Leaflet). `ais_areas` is the `[{ time, predicted_lat, predicted_lon, uncertainty_radius_km }]` search-target list.

### Run it

```bash
# In the opendrift conda env (has OpenDrift + xarray + scipy):
mamba activate opendrift
pip install -r requirements.txt          # adds fastapi + uvicorn
uvicorn src.server:app --reload --port 8000
# Interactive docs:  http://localhost:8000/docs
```

```bash
# Quick check
curl -s localhost:8000/health
curl -s localhost:8000/forecast \
  -H 'Content-Type: application/json' \
  -d '{"spill_lat": 18.5, "spill_lon": 71.5, "detection_time": "2026-09-04T12:00:00Z", "duration_hours": 24, "radius_m": 1200}' \
  -o forecast.json
```

**Behaviour notes**
* The server boots with only `fastapi`/`uvicorn` installed. Model modules load **lazily per request**; if OpenDrift is missing the model endpoints return a clean `503` (see `GET /health` → `opendrift_available`).
* Runs are synchronous: a request blocks until the Open-Meteo fetch + OpenOil simulation finish (typically 1–5+ min). Output files are written under `output/` relative to the launch directory.
* For hindcasts, pass an `observation_time` far enough in the past that Open-Meteo has marine/current data for the window.

---

## 🔮 Future Enhancements (Production Readiness)
* ~~**API Wrapper:**~~ ✅ Done — see `src/server.py` above.
* **INCOIS Integration:** Swapping the Open-Meteo API wrapper for direct local ingestion of daily INCOIS/Copernicus NetCDF files for zero-latency, enterprise-grade data.
* **3D Weathering:** Enabling full 3D vertical mixing and chemical weathering properties for specific oil profiles.