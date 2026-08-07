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


def test_enrich_maps_real_dataset_columns():
    """kartik2112/fraud-detection schema — real fields, nothing synthesized."""
    df = pd.DataFrame(
        {
            "trans_date_trans_time": ["2019-01-01 00:00:18", "2019-01-01 00:05:00"],
            "cc_num": [4111111111111111, 4111111111111111],
            "merchant": ["fraud_Kirlin and Sons", "fraud_Sporer-Keebler"],
            "category": ["grocery_pos", "shopping_net"],
            "amt": [4.97, 220.11],
            "is_fraud": [0, 1],
        }
    )
    out = enrich(df)
    assert list(out["card_id"]) == ["card_4111111111111111", "card_4111111111111111"]
    assert list(out["merchant_category"]) == ["grocery_pos", "shopping_net"]
    assert list(out["amount"]) == [4.97, 220.11]
    assert list(out["label"]) == [0, 1]
    assert out["event_time"].dt.tz is not None


def test_to_event_shape():
    row = pd.Series(
        {
            "card_id": "card_0001",
            "merchant_category": "mcc_007",
            "amount": 99.99,
            "event_time": pd.Timestamp("2013-09-01T01:00:00", tz="UTC"),
            "label": 1,
            "V1": 0.5,
        }
    )
    event = to_event(row)
    assert event["card_id"] == "card_0001"
    assert event["amount"] == 99.99
    assert event["label"] == 1
    assert event["event_time"] == "2013-09-01T01:00:00Z"
    assert event["V1"] == 0.5
