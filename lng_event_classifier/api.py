"""FastAPI interface for the synthetic LNG event classification demo.

Train locally before serving. Loading an untrusted joblib model is unsafe.
"""
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .labels import LABELS


MODEL_NOTE = "Uncalibrated model score; not a verification or evidence confidence rating."


class PredictionRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=2000,
        description="A short LNG shipping-event description (maximum 2,000 characters).",
    )

    @field_validator("text")
    @classmethod
    def reject_whitespace_only(cls, text: str) -> str:
        if not text.strip():
            raise ValueError("Event text must not be blank.")
        return text.strip()


class PredictionResponse(BaseModel):
    label: str = Field(description="Predicted event category from the ten-label demo taxonomy.")
    model_probability: float = Field(ge=0, le=1, description="Uncalibrated class probability.")
    note: str


class HealthResponse(BaseModel):
    status: str
    model_ready: bool


def create_app(model_path: str | Path = "artifacts/event_classifier.joblib") -> FastAPI:
    """Create an app for a trusted, locally trained model file."""
    path = Path(model_path)
    api = FastAPI(
        title="LNG Event Classifier API",
        description=(
            "Synthetic-data NLP demonstration. Predictions are not evidence that "
            "a shipping or sanctions event occurred."
        ),
        version="0.2.0",
    )

    @lru_cache(maxsize=1)
    def load_model() -> Any:
        if not path.is_file():
            raise HTTPException(
                status_code=503,
                detail="Model is unavailable. First run: python -m lng_event_classifier.train",
            )
        # Only load your own locally trained model; joblib deserialization is unsafe
        # for untrusted files. The model is cached to avoid repeated disk reads.
        return joblib.load(path)

    @api.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        ready = path.is_file()
        return HealthResponse(status="ready" if ready else "model_missing", model_ready=ready)

    @api.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        model = load_model()
        predicted = str(model.predict([request.text])[0])
        if predicted not in LABELS:
            raise HTTPException(status_code=500, detail="Model returned an unknown label.")
        probability = round(float(model.predict_proba([request.text]).max()), 4)
        return PredictionResponse(
            label=predicted,
            model_probability=probability,
            note=MODEL_NOTE,
        )

    return api


app = create_app()
