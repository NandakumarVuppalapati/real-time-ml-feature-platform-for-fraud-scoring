"""
API-layer tests with model_loader mocked out — these validate request/
response contracts and error handling, not the ML pipeline itself (that's
tests/test_feature_repo.py and a manual `make train` + curl for the model).
Keeping these independent of a real trained model / Feast / Redis means
they run in CI with no external services at all.
"""
from fastapi.testclient import TestClient

from serving import model_loader
from serving.main import app


def test_health_when_model_not_loaded(monkeypatch):
    monkeypatch.setattr(model_loader, "is_ready", lambda: False)
    monkeypatch.setattr(model_loader, "model_version", lambda: None)
    monkeypatch.setattr(model_loader, "startup", lambda: None)

    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded"
    assert r.json()["model_loaded"] is False


def test_score_returns_503_without_a_model(monkeypatch):
    monkeypatch.setattr(model_loader, "is_ready", lambda: False)
    monkeypatch.setattr(model_loader, "startup", lambda: None)

    with TestClient(app) as client:
        r = client.post("/score", json={"card_id": "card_0001", "amount": 10.0})
    assert r.status_code == 503


def test_score_happy_path(monkeypatch):
    monkeypatch.setattr(model_loader, "startup", lambda: None)
    monkeypatch.setattr(model_loader, "is_ready", lambda: True)
    monkeypatch.setattr(model_loader, "model_version", lambda: "7")
    monkeypatch.setattr(
        model_loader,
        "get_online_features",
        lambda card_id: {
            "txn_count_1m": 1, "txn_count_5m": 3, "txn_count_1h": 9,
            "txn_amount_sum_5m": 150.0, "txn_amount_avg_1h": 60.0,
            "time_since_last_txn_sec": 20.0, "distinct_merchant_count_1h": 4,
        },
    )
    monkeypatch.setattr(model_loader, "predict", lambda features: 0.87)

    with TestClient(app) as client:
        r = client.post(
            "/score", json={"card_id": "card_0001", "amount": 249.99, "merchant_category": "mcc_012"}
        )

    assert r.status_code == 200
    body = r.json()
    assert body["card_id"] == "card_0001"
    assert body["fraud_score"] == 0.87
    assert body["is_fraud"] is True
    assert body["model_version"] == "7"
    assert "txn_count_5m" in body["features_used"]


def test_score_rejects_non_positive_amount():
    with TestClient(app) as client:
        r = client.post("/score", json={"card_id": "card_0001", "amount": -5.0})
    assert r.status_code == 422
