"""
Bridges Flink's output topic (`card_features`, written by
flink_feature_job.py) into Feast's online store.

Why a separate hop instead of Flink writing to Redis directly: Feast owns
the online-store key encoding (how entity keys + feature values are
serialized into Redis), and that encoding is an internal implementation
detail that can change between Feast versions. Going through
`FeatureStore.push()` keeps Flink decoupled from Feast internals — Flink
only needs a Kafka producer, and this small service is the only thing that
needs the Feast Python SDK. It also means a swap of online store backend
(Redis -> DynamoDB, say) is a one-line feature_store.yaml change with zero
changes to the Flink job.

Batches a small window of messages before each push (default 200ms or 50
rows, whichever comes first) rather than pushing row-by-row, since Feast's
online write path is a batch API and per-row round trips would dominate
latency under load.

Usage:
    python streaming/feast_pusher.py
"""
import json
import logging
import os
import time

import pandas as pd
from feast import FeatureStore
from feast.data_source import PushMode
from kafka import KafkaConsumer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("streaming.feast_pusher")

TOPIC = os.environ.get("FEATURE_TOPIC", "card_features")
BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
FEATURE_REPO_PATH = os.environ.get("FEATURE_REPO_PATH", "feature_repo")
BATCH_MAX_ROWS = int(os.environ.get("PUSHER_BATCH_MAX_ROWS", "50"))
BATCH_MAX_WAIT_SEC = float(os.environ.get("PUSHER_BATCH_MAX_WAIT_SEC", "0.2"))

FEATURE_COLUMNS = [
    "txn_count_1m",
    "txn_count_5m",
    "txn_count_1h",
    "txn_amount_sum_5m",
    "txn_amount_avg_1h",
    "time_since_last_txn_sec",
    "distinct_merchant_count_1h",
    "last_merchant_category",
]


def to_row(msg: dict) -> dict:
    row = {"card_id": msg["card_id"]}
    row["event_timestamp"] = pd.to_datetime(msg["event_timestamp"], utc=True)
    row["created_timestamp"] = pd.Timestamp.now(tz="UTC")
    for col in FEATURE_COLUMNS:
        row[col] = msg.get(col)
    # Cold-start rows (a card's first-ever transaction) carry nulls for the
    # lag-based features; default them rather than reject the message.
    row["time_since_last_txn_sec"] = row["time_since_last_txn_sec"] or 0.0
    row["last_merchant_category"] = row["last_merchant_category"] or "unknown"
    for col in ("txn_count_1m", "txn_count_5m", "txn_count_1h", "distinct_merchant_count_1h"):
        row[col] = row[col] if row[col] is not None else 0
    for col in ("txn_amount_sum_5m", "txn_amount_avg_1h"):
        row[col] = row[col] if row[col] is not None else 0.0
    return row


def run() -> None:
    store = FeatureStore(repo_path=FEATURE_REPO_PATH)
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP,
        group_id="feast-pusher",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",
        consumer_timeout_ms=int(BATCH_MAX_WAIT_SEC * 1000),
    )

    logger.info("Consuming '%s' from %s, pushing into Feast repo at %s",
                TOPIC, BOOTSTRAP, FEATURE_REPO_PATH)

    buffer: list[dict] = []
    last_flush = time.monotonic()

    def flush():
        nonlocal buffer, last_flush
        if not buffer:
            last_flush = time.monotonic()
            return
        df = pd.DataFrame(buffer)
        store.push("card_features_push_source", df, to=PushMode.ONLINE)
        logger.info("Pushed %d feature rows to the online store", len(df))
        buffer = []
        last_flush = time.monotonic()

    while True:
        for message in consumer:
            try:
                buffer.append(to_row(message.value))
            except Exception:  # noqa: BLE001
                logger.exception("Dropping malformed message: %s", message.value)
                continue
            if len(buffer) >= BATCH_MAX_ROWS:
                flush()
        if time.monotonic() - last_flush >= BATCH_MAX_WAIT_SEC:
            flush()


if __name__ == "__main__":
    run()
