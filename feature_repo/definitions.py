"""
Feast feature definitions for the fraud-scoring platform.

These schemas are the single contract shared by three places that must never
disagree, or you get training/serving skew:

  1. streaming/flink_feature_job.py   — computes these features in real time
     from the Kafka transaction stream and pushes them into Feast (online +
     offline) via streaming/feast_pusher.py.
  2. training/prepare_offline_features.py — recomputes the *same* feature
     definitions in batch over historical data (via Spark), for training.
  3. serving/main.py                  — reads these features back out of the
     Feast online store (Redis) at request time to score a live transaction.

Entity
------
The public Kaggle "creditcardfraud" dataset is PCA-anonymized and has no
card/customer identifier. We synthesize `card_id` (see data/generate_sample.py
and ingestion/producer.py) to have something realistic to key streaming
aggregations on. This is a deliberate, documented simulation choice — call it
out explicitly if you discuss this project.

Features
--------
All features are per-card transaction velocity / amount aggregates over
trailing windows, computed as-of the current event:

  txn_count_1m / txn_count_5m / txn_count_1h  — rolling transaction counts
  txn_amount_sum_5m                            — rolling amount sum
  txn_amount_avg_1h                            — rolling amount average
  time_since_last_txn_sec                      — seconds since this card's
                                                  previous transaction
  distinct_merchant_count_1h                   — merchant diversity, a
                                                  classic card-testing /
                                                  fraud-ring signal
"""
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, PushSource
from feast.types import Float32, Int64, String
from feast.value_type import ValueType

# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------
card = Entity(
    name="card",
    join_keys=["card_id"],
    value_type=ValueType.STRING,
    description="Synthetic card identifier assigned to anonymized Kaggle transactions",
)

# ---------------------------------------------------------------------------
# Batch source: historical features computed offline by Spark
# (training/prepare_offline_features.py), used for point-in-time-correct
# training datasets via Feast's get_historical_features().
# ---------------------------------------------------------------------------
card_features_batch_source = FileSource(
    name="card_features_batch_source",
    # Relative to this feature_repo/ directory (where feature_store.yaml
    # lives) -- the parquet file itself is written by
    # training/prepare_offline_features.py into the shared top-level data/
    # directory so ingestion, training, and the feature repo all read/write
    # one dataset location.
    path="../data/offline/card_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# ---------------------------------------------------------------------------
# Push source: the same feature rows, delivered in real time by
# streaming/feast_pusher.py as Flink computes them. Pushing (rather than
# writing to Redis directly) keeps Feast as the single source of truth for
# both online and offline writes, and keeps Flink decoupled from Feast's
# internal key encoding.
# ---------------------------------------------------------------------------
card_features_push_source = PushSource(
    name="card_features_push_source",
    batch_source=card_features_batch_source,
)

card_transaction_features = FeatureView(
    name="card_transaction_features",
    entities=[card],
    ttl=timedelta(hours=2),
    schema=[
        Field(name="txn_count_1m", dtype=Int64),
        Field(name="txn_count_5m", dtype=Int64),
        Field(name="txn_count_1h", dtype=Int64),
        Field(name="txn_amount_sum_5m", dtype=Float32),
        Field(name="txn_amount_avg_1h", dtype=Float32),
        Field(name="time_since_last_txn_sec", dtype=Float32),
        Field(name="distinct_merchant_count_1h", dtype=Int64),
        Field(name="last_merchant_category", dtype=String),
    ],
    online=True,
    source=card_features_push_source,
    tags={"team": "fraud-platform", "freshness_sla_seconds": "5"},
)
