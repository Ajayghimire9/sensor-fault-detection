"""FastAPI inference service for SensorWatch."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.getenv("SENSORWATCH_MODEL_PATH", "artifacts/model.joblib"))
app = FastAPI(title="SensorWatch Predictive Maintenance API", version="2.0.0")
_bundle: dict[str, Any] | None = None

class SensorReading(BaseModel):
    readings: dict[str, float | None] = Field(min_length=1)

@app.on_event("startup")
def load_model() -> None:
    global _bundle
    if MODEL_PATH.exists():
        _bundle = joblib.load(MODEL_PATH)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/ready")
def ready() -> dict[str, Any]:
    return {"ready": _bundle is not None, "model_path": str(MODEL_PATH)}

@app.post("/v1/predict")
def predict(payload: SensorReading) -> dict[str, Any]:
    if _bundle is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded")
    expected = _bundle["features"]
    missing = sorted(set(expected) - set(payload.readings))
    unknown = sorted(set(payload.readings) - set(expected))
    if missing or unknown:
        raise HTTPException(status_code=422, detail={"missing": missing, "unknown": unknown})
    row = pd.DataFrame([{f: payload.readings[f] for f in expected}])
    probability = float(_bundle["model"].predict_proba(row)[:, 1][0])
    threshold = float(_bundle["threshold"])
    return {"fault_probability": probability, "fault": probability >= threshold, "threshold": threshold}
