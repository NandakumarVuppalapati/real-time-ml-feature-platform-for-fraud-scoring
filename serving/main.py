"""
FastAPI fraud-scoring service.

POST /score  — the sub-second path: fetch this card's latest features from
               Feast's online store (Redis), run them through the current
               @champion model from the MLflow registry, return a score.
GET  /health — liveness/readiness, reports whether a model is loaded.
POST /admin/reload — hot-swap in a newly trained + re-aliased model without
               restarting the process.

Run locally:
    uvicorn serving.main:app --reload --port 8000
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from serving import model_loader
from serving.schemas import HealthResponse, ScoreRequest, ScoreResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("serving.main")

FRAUD_THRESHOLD = 0.5


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_loader.startup()
    yield


app = FastAPI(
    title="Real-Time Fraud Scoring API",
    description=(
        "Serves fraud predictions using features materialized in real time "
        "by the Kafka -> Flink -> Feast pipeline, and a model trained on "
        "point-in-time-correct historical features from the same feature "
        "definitions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if model_loader.is_ready() else "degraded",
        model_loaded=model_loader.is_ready(),
        model_version=model_loader.model_version(),
        feast_online_store="connected",
    )


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    if not model_loader.is_ready():
        raise HTTPException(
            status_code=503,
            detail="No model loaded yet — run training/train.py, then POST /admin/reload.",
        )

    start = time.perf_counter()
    features = model_loader.get_online_features(request.card_id)
    fraud_score = model_loader.predict(features)
    latency_ms = (time.perf_counter() - start) * 1000

    return ScoreResponse(
        card_id=request.card_id,
        fraud_score=round(fraud_score, 6),
        is_fraud=fraud_score >= FRAUD_THRESHOLD,
        threshold=FRAUD_THRESHOLD,
        model_name=model_loader.MODEL_NAME,
        model_version=model_loader.model_version() or "unknown",
        features_used=features,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/admin/reload")
def reload_model() -> dict:
    try:
        version = model_loader.reload_model()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"reloaded": True, "model_version": version}
