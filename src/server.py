"""FastAPI wrapper for the oil-spill trajectory model (SIH 26143).

Exposes the forward-forecast and backward-hindcast engines as HTTP endpoints.
The OpenDrift / xarray heavy-lifting is imported lazily inside each request so
the server can boot and report a clean 503 even where the model environment
(conda "opendrift" env) is not installed.

Run from the repo root:
    uvicorn src.server:app --reload --port 8000

Endpoints:
    GET  /health     -> dependency availability (no model import)
    POST /forecast   -> run_forward_prediction
    POST /hindcast   -> run_backward_hindcasting
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# The model scripts use top-level imports (`from environment import ...`), so
# make this file's directory importable no matter where uvicorn is launched.
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

logger = logging.getLogger("sih.api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="SIH 26143 — Oil Spill Trajectory & Hindcasting API",
    version="1.0.0",
    description=(
        "Exposes the OpenDrift/OpenOil trajectory engine. POST /forecast predicts "
        "where a detected spill drifts; POST /hindcast back-tracks an observed slick "
        "to a probability region of origin plus AIS search areas."
    ),
)

# Dev-only broad CORS so the Vite dashboard can call this from another origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _now_iso() -> str:
    """Current UTC time as a naive ISO string (the model expects naive datetimes)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _model_available() -> bool:
    return importlib.util.find_spec("opendrift") is not None


def _load_model_module(name: str):
    """Import a model module, translating a missing model env into a clean 503."""
    if not _model_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "OpenDrift is not installed. Create the model environment first: "
                "`mamba create -n opendrift python=3.12 && mamba activate opendrift "
                "&& pip install -r requirements.txt` — then start this server inside it."
            ),
        )
    try:
        return importlib.import_module(name)
    except ImportError as exc:  # some other model dependency missing
        logger.exception("Model module %s failed to import", name)
        raise HTTPException(
            status_code=503,
            detail=f"Model dependencies missing for {name}: {exc}",
        ) from exc


def _read_json(rel_path: str) -> dict | list:
    try:
        with open(rel_path, "r") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.exception("Failed to read model output %s", rel_path)
        raise HTTPException(
            status_code=502,
            detail=f"Model ran but its output ({rel_path}) could not be read: {exc}",
        ) from exc


class _BaseRequest(BaseModel):
    oil_type: str = Field(default="GENERIC HEAVY CRUDE")


class ForecastRequest(_BaseRequest):
    spill_lat: float = Field(..., ge=-90, le=90)
    spill_lon: float = Field(..., ge=-180, le=180)
    detection_time: str | None = Field(
        default=None,
        description="ISO-8601 detection time. Defaults to now (UTC).",
    )
    duration_hours: int = Field(default=24, ge=1, le=168)
    radius_m: float = Field(default=1000, gt=0, le=100_000)


class HindcastRequest(_BaseRequest):
    observed_lat: float = Field(..., ge=-90, le=90)
    observed_lon: float = Field(..., ge=-180, le=180)
    observation_time: str | None = Field(
        default=None,
        description="ISO-8601 observation time. Defaults to now (UTC).",
    )
    backward_duration_hours: int = Field(default=24, ge=1, le=168)
    radius_m: float = Field(default=2500, gt=0, le=100_000)


def _meta(mode: str) -> dict:
    return {"mode": mode, "generated_at": _now_iso()}


@app.get("/", include_in_schema=False)
def root():
    return {"service": "sih-trajectory-api", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    """Liveness + model-dependency probe. Never imports the model."""
    ready = _model_available()
    return {
        "status": "ok",
        "service": "sih-trajectory-api",
        "opendrift_available": ready,
        "note": "Model ready." if ready
        else "Install the opendrift conda env before running forecast/hindcast.",
    }


@app.post("/forecast")
def forecast(req: ForecastRequest):
    payload = {
        "spill_lat": req.spill_lat,
        "spill_lon": req.spill_lon,
        "detection_time_str": req.detection_time or _now_iso(),
        "duration_hours": req.duration_hours,
        "radius_m": req.radius_m,
        "oil_type": req.oil_type,
    }
    mod = _load_model_module("forward_predictor")
    logger.info("[forecast] running for %s, %s", req.spill_lat, req.spill_lon)
    try:
        geojson_path = mod.run_forward_prediction(**payload)
    except HTTPException:
        raise
    except Exception as exc:  # Open-Meteo network errors, engine failures, etc.
        logger.exception("[forecast] run failed")
        raise HTTPException(
            status_code=502,
            detail=f"run_forward_prediction failed: {type(exc).__name__}: {exc}",
        ) from exc
    return {
        "success": True,
        **_meta("forecast"),
        "geojson": _read_json(geojson_path),
        "ais_areas": None,
    }


@app.post("/hindcast")
def hindcast(req: HindcastRequest):
    payload = {
        "observed_lat": req.observed_lat,
        "observed_lon": req.observed_lon,
        "observation_time_str": req.observation_time or _now_iso(),
        "backward_duration_hours": req.backward_duration_hours,
        "radius_m": req.radius_m,
        "oil_type": req.oil_type,
    }
    mod = _load_model_module("hindcaster")
    logger.info("[hindcast] running for %s, %s", req.observed_lat, req.observed_lon)
    try:
        geojson_path, ais_path = mod.run_backward_hindcasting(**payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[hindcast] run failed")
        raise HTTPException(
            status_code=502,
            detail=f"run_backward_hindcasting failed: {type(exc).__name__}: {exc}",
        ) from exc
    return {
        "success": True,
        **_meta("hindcast"),
        "geojson": _read_json(geojson_path),
        "ais_areas": _read_json(ais_path),
    }
