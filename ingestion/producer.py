"""
Kafka producer: replays the (anonymized, historical) Kaggle transaction
dataset as a live event stream, to stand in for a real card-network feed.

The Kaggle "creditcardfraud" dataset has no card/customer identifier, so if
the input file doesn't already have one (the committed data/sample file
does; the real data/raw/creditcard.csv you download won't), this producer
assigns a synthetic card_id and merchant_category deterministically from
each row's hash — see the module docstring in feature_repo/definitions.py
for why, and why that's a documented simulation choice rather than
something to gloss over.

Playback pacing: the dataset's `Time` column is seconds elapsed between
transactions. We replay respecting those gaps, compressed by --speed (e.g.
--speed 200 replays 48h of data in ~15 minutes), so downstream Flink
windowing sees realistic (if compressed) inter-arrival timing rather than
an unrealistic flat-out burst.

Usage:
    python ingestion/producer.py --speed 300
    python ingestion/producer.py --input data/raw/creditcard.csv --topic transactions.raw
"""
import argparse
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from kafka import KafkaProducer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingestion.producer")

BASE_TIME = datetime(2013, 9, 1, tzinfo=timezone.utc)
N_CARDS = 500
N_MERCHANTS = 80


def _stable_bucket(row_hash: str, n_buckets: int) -> int:
    return int(row_hash, 16) % n_buckets + 1


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Adds card_id / merchant_category if the source file doesn't have them
    (i.e. the raw Kaggle download, as opposed to our pre-enriched sample)."""
    if "card_id" in df.columns and "merchant_category" in df.columns:
        return df

    logger.info("Source has no card_id/merchant_category — synthesizing (see docstring)")
    v_cols = [c for c in df.columns if c.startswith("V")]

    def row_hash(row) -> str:
        payload = f"{row['Time']}:" + ":".join(f"{row[c]:.6f}" for c in v_cols[:5])
        return hashlib.md5(payload.encode()).hexdigest()

    hashes = df.apply(row_hash, axis=1)
    df = df.copy()
    df["card_id"] = hashes.apply(lambda h: f"card_{_stable_bucket(h, N_CARDS):04d}")
    df["merchant_category"] = hashes.apply(lambda h: f"mcc_{_stable_bucket(h[8:], N_MERCHANTS):03d}")
    return df


def to_event(row: pd.Series) -> dict:
    event_time = BASE_TIME + timedelta(seconds=float(row["Time"]))
    return {
        "card_id": row["card_id"],
        "merchant_category": row["merchant_category"],
        "amount": float(row["Amount"]),
        "event_time": event_time.isoformat(),
        # V1-V28 are PCA components of the original (never-disclosed) raw
        # features in the Kaggle dataset; we pass them through so a future
        # model iteration could use them directly instead of only the
        # engineered velocity features.
        **{c: float(row[c]) for c in row.index if c.startswith("V")},
        "label": int(row["Class"]) if "Class" in row else None,
    }


def run(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input)
    df = enrich(df).sort_values("Time").reset_index(drop=True)
    if args.max_rows:
        df = df.head(args.max_rows)

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        linger_ms=5,
    )

    logger.info(
        "Replaying %d transactions to topic '%s' at %sx speed (bootstrap=%s)",
        len(df), args.topic, args.speed, args.bootstrap_servers,
    )

    prev_time = None
    sent = 0
    try:
        for _, row in df.iterrows():
            if prev_time is not None:
                gap = max(float(row["Time"]) - prev_time, 0.0) / args.speed
                if gap > 0:
                    time.sleep(min(gap, args.max_gap_sec))
            prev_time = float(row["Time"])

            event = to_event(row)
            producer.send(args.topic, key=event["card_id"], value=event)
            sent += 1
            if sent % 200 == 0:
                logger.info("Sent %d/%d events", sent, len(df))
        producer.flush()
        logger.info("Done. Sent %d events.", sent)
    finally:
        producer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--topic", default="transactions.raw")
    parser.add_argument(
        "--bootstrap-servers", default="localhost:9092",
        help="Comma-separated Kafka bootstrap servers (docker-compose sets this via env)",
    )
    parser.add_argument("--speed", type=float, default=200.0, help="Playback speed multiplier")
    parser.add_argument("--max-gap-sec", type=float, default=2.0, help="Cap on sleep between events")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    if args.input is None:
        import os
        raw = "data/raw/creditcard.csv"
        args.input = raw if os.path.exists(raw) else "data/sample/sample_creditcard.csv"
        logger.info("--input not given, using %s", args.input)

    run(args)


if __name__ == "__main__":
    main()
