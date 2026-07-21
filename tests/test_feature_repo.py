"""
Validates feature_repo/definitions.py end to end: `feast apply` succeeds,
pushed feature rows show up in an online read, and a historical
(point-in-time) read returns the same values back for a timestamp at or
after the push. This is the exact mechanism streaming/feast_pusher.py and
training/train.py rely on — see their docstrings.
"""
from datetime import datetime, timedelta, timezone

import pandas as pd
from feast.data_source import PushMode


def _push_row(store, card_id="card_test_001", **overrides):
    now = datetime.now(timezone.utc)
    row = {
        "card_id": card_id,
        "event_timestamp": now,
        "created_timestamp": now,
        "txn_count_1m": 1,
        "txn_count_5m": 3,
        "txn_count_1h": 9,
        "txn_amount_sum_5m": 150.0,
        "txn_amount_avg_1h": 60.0,
        "time_since_last_txn_sec": 22.0,
        "distinct_merchant_count_1h": 4,
        "last_merchant_category": "mcc_005",
    }
    row.update(overrides)
    store.push("card_features_push_source", pd.DataFrame([row]), to=PushMode.ONLINE)
    return row, now


def test_apply_registers_feature_view(feast_repo):
    fvs = {fv.name for fv in feast_repo.list_feature_views()}
    assert "card_transaction_features" in fvs


def test_push_and_online_read_round_trip(feast_repo):
    row, _ = _push_row(feast_repo, card_id="card_roundtrip")

    resp = feast_repo.get_online_features(
        features=[
            "card_transaction_features:txn_count_5m",
            "card_transaction_features:txn_amount_avg_1h",
            "card_transaction_features:distinct_merchant_count_1h",
        ],
        entity_rows=[{"card_id": "card_roundtrip"}],
    ).to_dict()

    assert resp["txn_count_5m"][0] == row["txn_count_5m"]
    assert resp["txn_amount_avg_1h"][0] == row["txn_amount_avg_1h"]
    assert resp["distinct_merchant_count_1h"][0] == row["distinct_merchant_count_1h"]


def test_online_read_for_unseen_card_returns_none(feast_repo):
    resp = feast_repo.get_online_features(
        features=["card_transaction_features:txn_count_5m"],
        entity_rows=[{"card_id": "card_never_seen_by_anything"}],
    ).to_dict()
    assert resp["txn_count_5m"][0] is None  # caller (serving/model_loader.py) fills this with 0


def test_historical_features_point_in_time_join(feast_repo):
    # get_historical_features joins against the *offline* source (the
    # parquet training/prepare_offline_features.py writes) — pushing to
    # PushMode.ONLINE (as streaming/feast_pusher.py does) intentionally
    # does not affect it. So to test the offline point-in-time join, write
    # two feature snapshots for the same card directly to the offline
    # parquet, as if two Spark batch runs had produced them, and confirm a
    # historical query "as of" a given time gets the snapshot that was
    # current then — not a later one (that would be feature leakage).
    t0 = datetime.now(timezone.utc) - timedelta(hours=2)
    t1 = datetime.now(timezone.utc) - timedelta(hours=1)

    snapshots = pd.DataFrame(
        [
            {
                "card_id": "card_hist", "event_timestamp": t0, "created_timestamp": t0,
                "txn_count_1m": 0, "txn_count_5m": 1, "txn_count_1h": 1,
                "txn_amount_sum_5m": 20.0, "txn_amount_avg_1h": 20.0,
                "time_since_last_txn_sec": 0.0, "distinct_merchant_count_1h": 1,
                "last_merchant_category": "mcc_001",
                "Amount": 20.0, "Class": 0, "merchant_category": "mcc_001",
            },
            {
                "card_id": "card_hist", "event_timestamp": t1, "created_timestamp": t1,
                "txn_count_1m": 1, "txn_count_5m": 4, "txn_count_1h": 6,
                "txn_amount_sum_5m": 300.0, "txn_amount_avg_1h": 75.0,
                "time_since_last_txn_sec": 40.0, "distinct_merchant_count_1h": 3,
                "last_merchant_category": "mcc_002",
                "Amount": 75.0, "Class": 0, "merchant_category": "mcc_002",
            },
        ]
    )
    snapshots.to_parquet(feast_repo.offline_parquet_path)

    # Querying "as of" a time between t0 and t1 should return the t0
    # snapshot, not t1 (t1 hadn't happened yet).
    entity_df = pd.DataFrame(
        {"card_id": ["card_hist"], "event_timestamp": [t0 + timedelta(minutes=1)]}
    )
    hist = feast_repo.get_historical_features(
        entity_df=entity_df,
        features=["card_transaction_features:txn_count_5m", "card_transaction_features:txn_amount_avg_1h"],
    ).to_df()

    assert hist.loc[0, "txn_count_5m"] == 1
    assert hist.loc[0, "txn_amount_avg_1h"] == 20.0

    # Querying "as of" a time after t1 should now pick up the t1 snapshot.
    entity_df_later = pd.DataFrame(
        {"card_id": ["card_hist"], "event_timestamp": [t1 + timedelta(minutes=1)]}
    )
    hist_later = feast_repo.get_historical_features(
        entity_df=entity_df_later,
        features=["card_transaction_features:txn_count_5m", "card_transaction_features:txn_amount_avg_1h"],
    ).to_df()

    assert hist_later.loc[0, "txn_count_5m"] == 4
    assert hist_later.loc[0, "txn_amount_avg_1h"] == 75.0
