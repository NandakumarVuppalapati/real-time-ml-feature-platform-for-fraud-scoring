"""
Ensures data/offline/card_features.parquet exists (Feast needs to read its
schema at `feast apply` time, even before any real feature data has been
computed) and then runs `feast apply`.

On a fresh clone, training/prepare_offline_features.py hasn't run yet, so
there's no real offline parquet file. Rather than commit a binary parquet
placeholder to git, generate a minimal schema-correct stub here — it gets
overwritten by the real thing the first time someone runs
training/prepare_offline_features.py.
"""
import os
from datetime import datetime, timezone

import pandas as pd

OFFLINE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "offline", "card_features.parquet")

SCHEMA_COLUMNS = {
    "card_id": "placeholder_card",
    "event_timestamp": datetime.now(timezone.utc),
    "created_timestamp": datetime.now(timezone.utc),
    "txn_count_1m": 0,
    "txn_count_5m": 0,
    "txn_count_1h": 0,
    "txn_amount_sum_5m": 0.0,
    "txn_amount_avg_1h": 0.0,
    "time_since_last_txn_sec": 0.0,
    "distinct_merchant_count_1h": 0,
    "last_merchant_category": "none",
    "Amount": 0.0,
    "Class": 0,
    "merchant_category": "none",
}


def ensure_offline_placeholder() -> None:
    if os.path.exists(OFFLINE_PATH):
        return
    os.makedirs(os.path.dirname(OFFLINE_PATH), exist_ok=True)
    pd.DataFrame([SCHEMA_COLUMNS]).to_parquet(OFFLINE_PATH)
    print(f"No offline features found yet — wrote schema placeholder to {OFFLINE_PATH}")
    print("Run training/prepare_offline_features.py to replace it with real data.")


if __name__ == "__main__":
    ensure_offline_placeholder()
    os.system("feast apply")
