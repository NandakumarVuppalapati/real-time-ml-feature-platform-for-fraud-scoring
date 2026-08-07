"""
Kafka producer: replays a transaction dataset as a live event stream, to
stand in for a real card-network feed.

Column normalization (mapping whichever dataset schema is on disk into a
common card_id/merchant_category/amount/event_time/label shape) lives in
ingestion/enrich.py, shared with training/train.py — see that module's
docstring for why it's split out and what schemas it supports.

Playback pacing: we replay respecting the dataset's real inter-event time
gaps, compressed by --speed (e.g. --speed 300 replays a day of data in a few
minutes) and capped per-event by --max-gap-sec, so downstream Flink windowing
sees realistic (if compressed) inter-arrival timing rather than an
unrealistic flat-out burst.

Usage:
    python ingestion/producer.py --speed 300
    python ingestion/producer.py --input data/raw/fraudTest.csv --max-rows 50000
"""
import argparse
import json
import logging
import os
import time

import pandas as pd
from kafka import KafkaProducer

try:
    from ingestion.enrich import enrich  # repo-root context (tests, local run)
except ImportError:
    from enrich import enrich  # flattened Docker image (see ingestion/Dockerfile)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingestion.producer")


def to_event(row: pd.Series) -> dict:
    out = {
        "card_id": row["card_id"],
        "merchant_category": row["merchant_category"],
        "amount": float(row["amount"]),
        # Flink's JSON format (json.timestamp-format.standard=ISO-8601) only
        # accepts a literal 'Z' suffix for TIMESTAMP_LTZ columns (e.g.
        # "2020-03-27T12:13:14.123Z") -- it does NOT accept a numeric UTC
        # offset. pandas'/Python's isoformat() on a UTC-aware timestamp
        # renders "+00:00", not "Z", so Flink silently fails to parse this
        # field (swallowed by json.ignore-parse-errors=true), leaving
        # event_time NULL and the WATERMARK permanently stalled. event_time
        # is always UTC here, so this replace is safe and exact.
        "event_time": row["event_time"].isoformat().replace("+00:00", "Z"),
        # V1-V28 (only present for the mlg-ulb-derived schemas) are PCA
        # components of that dataset's original raw features; passed through
        # so a model iteration could use them directly. No-op for the real
        # (kartik2112) dataset, which has no V columns.
        **{c: float(row[c]) for c in row.index if c.startswith("V")},
        "label": int(row["label"]) if pd.notna(row.get("label")) else None,
    }
    return out


def run(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input)
    df = enrich(df).sort_values("event_time").reset_index(drop=True)
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
            cur_time = row["event_time"].timestamp()
            if prev_time is not None:
                gap = max(cur_time - prev_time, 0.0) / args.speed
                if gap > 0:
                    time.sleep(min(gap, args.max_gap_sec))
            prev_time = cur_time

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
        "--bootstrap-servers",
        default=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        help="Comma-separated Kafka bootstrap servers (docker-compose sets this via env)",
    )
    parser.add_argument("--speed", type=float, default=200.0, help="Playback speed multiplier")
    parser.add_argument("--max-gap-sec", type=float, default=2.0, help="Cap on sleep between events")
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Cap on rows replayed -- recommended for the full ~556k-row fraudTest.csv "
             "(e.g. --max-rows 50000) so a demo run finishes in a reasonable time.",
    )
    args = parser.parse_args()

    if args.input is None:
        # fraudTest.csv, not fraudTrain.csv: the live/demo stream should be
        # transactions the offline-trained model has never seen -- the same
        # train/serve split a real fraud team would have. fraudTrain.csv is
        # read directly by training/prepare_offline_features.py instead.
        raw = "data/raw/fraudTest.csv"
        args.input = raw if os.path.exists(raw) else "data/sample/sample_creditcard.csv"
        logger.info("--input not given, using %s", args.input)

    run(args)


if __name__ == "__main__":
    main()
