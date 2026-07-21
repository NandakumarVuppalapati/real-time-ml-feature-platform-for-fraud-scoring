from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    card_id: str = Field(..., examples=["card_0001"])
    amount: float = Field(..., gt=0, examples=[249.99])
    merchant_category: str = Field(default="unknown", examples=["mcc_017"])


class ScoreResponse(BaseModel):
    card_id: str
    fraud_score: float
    is_fraud: bool
    threshold: float
    model_name: str
    model_version: str
    features_used: dict
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    feast_online_store: str
