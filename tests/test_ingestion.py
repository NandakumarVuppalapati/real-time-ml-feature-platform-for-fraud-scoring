"""Pure-function tests for the producer's enrichment/serialization logic —
no Kafka broker needed."""
import pandas as pd

from ingestion.producer import enrich, to_event


def test_enrich_preserves_existing_card_id():
    df = pd.DataFrame(
        {"Time": [0, 1], "V1": [0.1, 0.2], "Amount": [10.0, 20.0], "Class": [0, 0],
         "card_id": ["card_A", "card_B"], "merchant_category": ["mcc_1", "mcc_2"]}
    )
    out = enrich(df)
    assert list(out["card_id"]) == ["card_A", "card_B"]


def test_enrich_synthesizes_when_missing():
    df = pd.DataFrame(
        {"Time": [0, 1], "V1": [0.1, 0.2], "V2": [0.1, 0.2], "V3": [0.1, 0.2],
         "V4": [0.1, 0.2], "V5": [0.1, 0.2], "Amount": [10.0, 20.0], "Class": [0, 0]}
    )
    out = enrich(df)
    assert "card_id" in out.columns
    assert "merchant_category" in out.columns
    assert out["card_id"].str.startswith("card_").all()


def test_enrich_is_deterministic():
    df = pd.DataFrame(
        {"Time": [5.0], "V1": [1.23456], "V2": [0.0], "V3": [0.0], "V4": [0.0], "V5": [0.0],
         "Amount": [50.0], "Class": [0]}
    )
    out1 = enrich(df.copy())
    out2 = enrich(df.copy())
    assert out1["card_id"].iloc[0] == out2["card_id"].iloc[0]


def test_to_event_shape():
    row = pd.Series(
        {"Time": 3600.0, "V1": 0.5, "Amount": 99.99, "card_id": "card_0001",
         "merchant_category": "mcc_007", "Class": 1}
    )
    event = to_event(row)
    assert event["card_id"] == "card_0001"
    assert event["amount"] == 99.99
    assert event["label"] == 1
    assert "event_time" in event
    assert event["V1"] == 0.5
