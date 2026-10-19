"""FastAPI inference endpoint with structured request logging."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from .schema import FEATURES

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("heart_api")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/heart_disease_model.joblib"))
app = FastAPI(title="Heart Disease Risk Classifier", version="1.0.0", description="Educational prediction service; not a medical device.")
model = None
request_count = 0


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


@app.on_event("startup")
def load_model() -> None:
    global model
    if MODEL_PATH.is_file():
        model = joblib.load(MODEL_PATH)
        logger.info("model_loaded path=%s", MODEL_PATH)
    else:
        logger.error("model_missing path=%s", MODEL_PATH)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    global request_count
    started = time.perf_counter()
    response = await call_next(request)
    request_count += 1
    logger.info("request method=%s path=%s status=%s duration_ms=%.2f", request.method, request.url.path, response.status_code, (time.perf_counter() - started) * 1000)
    return response


@app.get("/health")
def health():
    return {"status": "ok" if model is not None else "model_unavailable", "model_loaded": model is not None}


@app.get("/metrics")
def metrics():
    from fastapi.responses import PlainTextResponse
    body = (
        "# HELP api_requests_total Total HTTP requests handled by this process.\n"
        "# TYPE api_requests_total counter\n"
        f"api_requests_total {request_count}\n"
        "# HELP model_loaded Whether the serialized model is available.\n"
        "# TYPE model_loaded gauge\n"
        f"model_loaded {int(model is not None)}\n"
    )
    return PlainTextResponse(body, media_type="text/plain; version=0.0.4; charset=utf-8")


@app.post("/predict")
def predict(payload: PatientFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    values = payload.model_dump()
    probabilities = model.predict_proba(pd.DataFrame([values], columns=FEATURES))[0]
    predicted = int(probabilities[1] >= 0.5)
    return {"prediction": predicted, "label": "disease_present" if predicted else "disease_absent", "confidence": float(probabilities[predicted]), "positive_class_probability": float(probabilities[1]), "threshold": 0.5}
