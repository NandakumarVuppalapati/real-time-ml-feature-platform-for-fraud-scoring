"""
Shared dataset normalization logic — used by both ingestion/producer.py (the
live Kafka replay) and training/train.py (building the point-in-time entity
frame for Feast). Kept in its own module, separate from producer.py, so
importing it doesn't pull in kafka-python: training's Docker image has no
Kafka client installed (it doesn't need one), and a shared function that
transitively required one would force an unnecessary/unused dependency onto
that image just to reuse a few lines of pandas logic.

Handles three possible input schemas, normalized in `enrich()`:

1. kartik2112/fraud-detection (the real dataset — see data/download_dataset.py)
   -- genuine transaction fields (card number, merchant, category, amount,
   real timestamp), nothing to synthesize.
2. data/sample/sample_creditcard.csv -- the small committed sample, matching
   the original mlg-ulb/creditcardfraud schema (Time/V1-V28/Amount/Class)
   plus pre-assigned card_id/merchant_category, so the repo runs end-to-end
   in CI/tests without needing a dataset download at all.
3. The raw mlg-ulb/creditcardfraud schema with no identifier of any kind --
   deterministically synthesizes a card_id/merchant_category from each row's
   hash. Kept for backward compatibility; superseded by (1) for anything
   that needs real scale, since that dataset already has a real identifier.

Whichever schema producer.py and train.py are pointed at, they MUST agree on
how card_id/event_timestamp are derived from it -- that's the whole point of
sharing this module instead of two independently-maintained copies of the
same logic. If they ever disagreed, Feast's point-in-time join in train.py
would quietly line features up against the wrong labels.
"""
import hashlib
from datetime import datetime, timezone

import pandas as pd

BASE_TIME = datetime(2013, 9, 1, tzinfo=timezone.utc)
N_CARDS = 500
N_MERCHANTS = 80

# Row-level sampling percentage used by both training/prepare_offline_features.py
# (Spark, writing the offline parquet) and training/train.py (pandas, building
# the entity_df for Feast's point-in-time join). Both scripts apply
# row_in_sample() below (keyed on each row's real `trans_num`, keeping every
# fraud row unconditionally) so the two independently-run engines select the
# *same* subset of rows without needing to share state -- required so the
# offline parquet and entity_df stay 1:1 aligned, which is what lets every
# sampled entity row find its exact feature snapshot instead of falling back
# to a cold-start (all-null) match.
#
# Empirically tuned down hard (12 -> 3) after verifying live that Feast's
# local/file-based point-in-time join isn't simply "loads everything into
# memory" -- it OOM-killed (exit 137) even with both the entity_df and the
# offline parquet already reduced to a matched ~162k rows on a machine with
# ~7.6GB genuinely available to Docker. That points to real algorithmic
# memory scaling in Feast's join (it doesn't do an efficient grouped
# merge-as-of), not just raw data volume -- so the fix is a much smaller
# working set, not a "slightly smaller" one. A production deployment would
# swap in a warehouse-backed offline store (BigQuery/Snowflake/Redshift via
# Feast's connectors) that pushes the join down into the warehouse instead
# of pulling everything into local memory -- see the Cloud deployment path
# section of the README.
SAMPLE_PCT_DEFAULT = 3


def _stable_bucket(row_hash: str, n_buckets: int) -> int:
    return int(row_hash, 16) % n_buckets + 1


def row_in_sample(trans_num, keep_pct: int = SAMPLE_PCT_DEFAULT) -> bool:
    h = hashlib.md5(str(trans_num).encode()).hexdigest()
    return int(h[:8], 16) % 100 < keep_pct


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes any of the three supported schemas into a common shape
    with real `card_id`, `merchant_category`, `amount`, `label`, and a
    tz-aware UTC `event_time` column to sort and replay/train by."""

    if "cc_num" in df.columns:
        # kartik2112/fraud-detection: real transaction fields already.
        df = df.copy()
        df["card_id"] = "card_" + df["cc_num"].astype(str)
        df["merchant_category"] = df["category"]
        df["amount"] = df["amt"].astype(float)
        # Sparkov's generated timestamps carry no real timezone; treated as
        # UTC by convention, consistent with BASE_TIME below.
        df["event_time"] = pd.to_datetime(df["trans_date_trans_time"]).dt.tz_localize("UTC")
        df["label"] = df["is_fraud"].astype(int) if "is_fraud" in df.columns else None
        return df

    if "card_id" in df.columns and "merchant_category" in df.columns:
        # data/sample/sample_creditcard.csv: pre-enriched, original
        # Time/Amount/Class schema.
        df = df.copy()
        df["amount"] = df["Amount"].astype(float)
        df["label"] = df["Class"].astype(int) if "Class" in df.columns else None
        df["event_time"] = BASE_TIME + pd.to_timedelta(df["Time"], unit="s")
        return df

    # Fallback: raw mlg-ulb/creditcardfraud with no identifier at all --
    # deterministically synthesize one. Documented simulation, not something
    # to gloss over -- see README.
    v_cols = [c for c in df.columns if c.startswith("V")]

    def row_hash(row) -> str:
        payload = f"{row['Time']}:" + ":".join(f"{row[c]:.6f}" for c in v_cols[:5])
        return hashlib.md5(payload.encode()).hexdigest()

    hashes = df.apply(row_hash, axis=1)
    df = df.copy()
    df["card_id"] = hashes.apply(lambda h: f"card_{_stable_bucket(h, N_CARDS):04d}")
    df["merchant_category"] = hashes.apply(lambda h: f"mcc_{_stable_bucket(h[8:], N_MERCHANTS):03d}")
    df["amount"] = df["Amount"].astype(float)
    df["label"] = df["Class"].astype(int) if "Class" in df.columns else None
    df["event_time"] = BASE_TIME + pd.to_timedelta(df["Time"], unit="s")
    return df
