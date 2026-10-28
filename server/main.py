"""FastAPI inference microservice: MLP regressor + MLP classifier behind POST /predict."""
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

MODELS_DIR = Path(__file__).parent / "models"
CLASS_LABELS = {0: "Under-Stimulated", 1: "Optimal Hypertrophy", 2: "Excessive Fatigue / Junk Volume"}


# Bounds follow the training domain (OpenPowerlifting filter + simulator ranges) so the MLP is
# never asked to extrapolate far outside the data it was fit on.
MAX_RELATIVE_VOLUME = 50.0  # session volume / 1RM; training max is ~46


class WorkoutPayload(BaseModel):
    bodyweight_kg: float = Field(ge=40, le=200)
    age: float = Field(ge=16, le=80)
    sex_binary: Literal[0, 1] = Field(description="1 = male, 0 = female")
    bench_1rm_baseline: float = Field(ge=20, le=330)
    session_volume_kg: float = Field(gt=0, le=10000)
    mean_rpe: float = Field(ge=6.0, le=10.0)
    fatigue_drop_off: float = Field(ge=0, le=100, description="% rep drop, set 1 to final set")

    @model_validator(mode="after")
    def volume_within_training_domain(self) -> "WorkoutPayload":
        if self.session_volume_kg / self.bench_1rm_baseline > MAX_RELATIVE_VOLUME:
            raise ValueError(
                f"session_volume_kg must be at most {MAX_RELATIVE_VOLUME:g} x bench_1rm_baseline"
            )
        return self


class PredictResponse(BaseModel):
    prescribed_load_kg: float
    volume_class_id: int
    volume_class_label: str
    inference_time_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    app.state.regressor = joblib.load(MODELS_DIR / "mlp_regressor.pkl")
    app.state.classifier = joblib.load(MODELS_DIR / "mlp_classifier.pkl")
    yield


app = FastAPI(title="Inference Microservice", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",  # any local dev port
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: WorkoutPayload, request: Request) -> PredictResponse:
    start = time.perf_counter()
    scaler = request.app.state.scaler
    # Column order comes from the fitted scaler, guaranteeing it matches training.
    row = pd.DataFrame([payload.model_dump()])[scaler.feature_names_in_]
    x = scaler.transform(row)
    load = float(request.app.state.regressor.predict(x)[0])
    class_id = int(request.app.state.classifier.predict(x)[0])
    return PredictResponse(
        prescribed_load_kg=round(load, 2),
        volume_class_id=class_id,
        volume_class_label=CLASS_LABELS[class_id],
        inference_time_ms=round((time.perf_counter() - start) * 1000, 3),
    )
