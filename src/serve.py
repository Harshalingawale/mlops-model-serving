"""FastAPI model-serving app.

    uvicorn src.serve:app --reload
"""
from __future__ import annotations
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

_model = None
_features: List[str] = []
_metrics: dict = {}


def _load():
    global _model, _features, _metrics
    if (ARTIFACTS / "model.joblib").exists():
        _model = joblib.load(ARTIFACTS / "model.joblib")
        _features = json.loads((ARTIFACTS / "features.json").read_text())
        _metrics = json.loads((ARTIFACTS / "metrics.json").read_text())


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    yield


app = FastAPI(title="MLOps Model Serving", version="1.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    features: List[float] = Field(..., description="Feature vector in training order")


class PredictResponse(BaseModel):
    prediction: int
    probability: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/metrics")
def metrics():
    return _metrics or {"detail": "train the model first: python -m src.train"}


@app.get("/features")
def features():
    return {"n_features": len(_features), "features": _features}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded. Run: python -m src.train")
    if len(req.features) != len(_features):
        raise HTTPException(422, f"Expected {len(_features)} features, got {len(req.features)}")
    X = pd.DataFrame([req.features], columns=_features)
    proba = float(_model.predict_proba(X)[0][1])
    return PredictResponse(prediction=int(proba >= 0.5), probability=round(proba, 4))
