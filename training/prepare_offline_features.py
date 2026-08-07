"""
Batch feature computation — the offline half of the feature platform.

This recomputes, over historical data, the *exact same* feature definitions
that streaming/flink_feature_job.py computes in real time (see the docstring
in feature_repo/definitions.py for why that consistency matters). It's the
free/local stand-in for what "Databricks" does in the resume bullet: managed
Spark for batch feature engineering. Runs identically here via open-source
PySpark, on Databricks Community Edition, or on EMR — only the cluster
config changes, not this code.

Input:  data/raw/fraudTrain.csv (kartik2112/fraud-detection's "historical"
        half — real card/merchant/timestamp fields) or
        data/sample/sample_creditcard.csv as a fallback so the pipeline is
        runnable without a Kaggle account.
Output: data/offline/card_features.parquet — read by Feast's FileSource
        (feature_repo/definitions.py) for point-in-time-correct training
        sets, via `feast.get_historical_features`.

Usage:
    python training/prepare_offline_features.py
    python training/prepare_offline_features.py --input data/raw/fraudTrain.csv
"""
import argparse
import os
from datetime import datetime, timezone

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

try:
    from ingestion.enrich import SAMPLE_PCT_DEFAULT  # repo-root context
except ImportError:
    from enrich import SAMPLE_PCT_DEFAULT  # flattened Docker image

BASE_TIME = datetime(2013, 9, 1, tzinfo=timezone.utc)  # arbitrary anchor, used
# only for the sample/fallback schema, whose "Time" column is seconds
# elapsed since the first transaction rather than a real calendar timestamp.
# The real (kartik2112) dataset carries actual timestamps and needs no anchor
# — see the cc_num branch in compute_features() below.


def build_spark() -> SparkSession:
    # spark.driver.host pinned to loopback so local-mode runs don't depend on
    # the container/host's hostname being DNS-resolvable (bites you in some
    # sandboxed or minimal-network Docker setups).
    return (
        SparkSession.builder.appName("fraud-offline-feature-prep")
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.host", os.environ.get("SPARK_LOCAL_IP", "127.0.0.1"))
        .getOrCreate()
    )


def compute_features(spark: SparkSession, input_path: str):
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    has_trans_num = "trans_num" in df.columns

    if "cc_num" in df.columns:
        # kartik2112/fraud-detection: real transaction fields already —
        # mirrors ingestion/producer.py's enrich() mapping for this schema,
        # since the two must agree on card_id/event_timestamp construction.
        df = (
            df.withColumn("card_id", F.concat(F.lit("card_"), F.col("cc_num").cast("string")))
            .withColumn("merchant_category", F.col("category"))
            .withColumn("amount", F.col("amt").cast("double"))
            .withColumn("event_timestamp", F.to_timestamp("trans_date_trans_time"))
            .withColumn("label", F.col("is_fraud").cast("int"))
        )
    else:
        # data/sample/sample_creditcard.csv (or a raw mlg-ulb file with
        # card_id already assigned) -- original Time/Amount/Class schema.
        df = (
            df.withColumn(
                "event_timestamp",
                (F.lit(BASE_TIME.timestamp()) + F.col("Time")).cast("timestamp"),
            )
            .withColumn("amount", F.col("Amount").cast("double"))
            .withColumn("label", F.col("Class").cast("int"))
        )

    df = df.withColumn("event_ts_long", F.col("event_timestamp").cast("long"))

    order_by_time = F.col("event_ts_long")
    w1m = Window.partitionBy("card_id").orderBy(order_by_time).rangeBetween(-60, 0)
    w5m = Window.partitionBy("card_id").orderBy(order_by_time).rangeBetween(-300, 0)
    w1h = Window.partitionBy("card_id").orderBy(order_by_time).rangeBetween(-3600, 0)
    order_only = Window.partitionBy("card_id").orderBy(order_by_time)

    features = (
        df.withColumn("txn_count_1m", F.count(F.lit(1)).over(w1m))
        .withColumn("txn_count_5m", F.count(F.lit(1)).over(w5m))
        .withColumn("txn_count_1h", F.count(F.lit(1)).over(w1h))
        .withColumn("txn_amount_sum_5m", F.sum("amount").over(w5m))
        .withColumn("txn_amount_avg_1h", F.avg("amount").over(w1h))
        .withColumn(
            "distinct_merchant_count_1h",
            F.size(F.collect_set("merchant_category").over(w1h)),
        )
        .withColumn("prev_event_ts", F.lag("event_ts_long").over(order_only))
        .withColumn(
            "time_since_last_txn_sec",
            F.when(
                F.col("prev_event_ts").isNotNull(),
                F.col("event_ts_long") - F.col("prev_event_ts"),
            ).otherwise(F.lit(None)),
        )
        .withColumn(
            "last_merchant_category", F.lag("merchant_category").over(order_only)
        )
        .withColumn("created_timestamp", F.current_timestamp())
    )

    out_cols = [
        "card_id",
        "event_timestamp",
        "created_timestamp",
        "txn_count_1m",
        "txn_count_5m",
        "txn_count_1h",
        "txn_amount_sum_5m",
        "txn_amount_avg_1h",
        "time_since_last_txn_sec",
        "distinct_merchant_count_1h",
        "last_merchant_category",
        # Kept for convenience when building training entity_df in train.py —
        # not part of the Feast FeatureView schema, so Feast simply ignores
        # them on retrieval.
        "amount",
        "label",
        "merchant_category",
    ]
    if has_trans_num:
        # Needed by sample_features() below to align this output 1:1 with
        # train.py's entity_df sampling; dropped before the final write.
        out_cols.append("trans_num")
    return features.select(out_cols)


def sample_features(features_df, sample_pct: int = SAMPLE_PCT_DEFAULT):
    """Row-level downsample applied *after* window features are computed
    over each card's full history (so the features themselves stay
    correct), keeping every fraud row and a deterministic hash-based sample
    of the rest. See ingestion/enrich.py's SAMPLE_PCT_DEFAULT docstring for
    why this exists: Feast's local point-in-time join loads this entire
    parquet into memory regardless of how small the training entity_df is,
    and OOMs on the full ~1.3M-row dataset even on an idle 8GB machine.
    train.py applies the identical trans_num-hash rule in pandas, so the
    two stay 1:1 aligned without either script needing to know what the
    other selected."""
    if sample_pct <= 0 or "trans_num" not in features_df.columns:
        # 0 (the default): no sampling, write every row -- see this
        # function's callers. Also the fallback for the small sample
        # dataset, which has no trans_num and is already tiny.
        return features_df.drop("trans_num") if "trans_num" in features_df.columns else features_df

    hash_bucket = (
        F.conv(F.substring(F.md5(F.col("trans_num").cast("string")), 1, 8), 16, 10)
        .cast("long") % 100
    )
    return (
        features_df.filter((F.col("label") == 1) | (hash_bucket < sample_pct))
        .drop("trans_num")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="Path to source CSV")
    parser.add_argument(
        "--output", default="data/offline/card_features.parquet", help="Output parquet path"
    )
    parser.add_argument(
        "--sample-pct", type=int, default=0,
        help="Optional row-level sample of the computed features written to "
             "parquet (keeps every fraud row) -- see sample_features() "
             "docstring. Defaults to 0 (write every row): now that train.py "
             "reads this parquet directly instead of through Feast's "
             "get_historical_features(), the OOM this was originally "
             "working around no longer applies, so the full dataset is the "
             "default. Left available as an escape hatch for anyone who "
             "wants to demonstrate Feast's actual point-in-time join on a "
             "smaller slice.",
    )
    args = parser.parse_args()

    input_path = args.input
    if input_path is None:
        # fraudTrain.csv, not fraudTest.csv -- see module docstring.
        raw = "data/raw/fraudTrain.csv"
        sample = "data/sample/sample_creditcard.csv"
        input_path = raw if os.path.exists(raw) else sample
        print(f"--input not given, using {input_path}")

    spark = build_spark()
    try:
        features = compute_features(spark, input_path)
        n_before = features.count()
        features = sample_features(features, args.sample_pct)
        n_after = features.count()
        if n_after != n_before:
            print(f"Sampled {n_before} -> {n_after} feature rows "
                  f"(--sample-pct {args.sample_pct}, all fraud rows kept)")
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        features.coalesce(1).write.mode("overwrite").parquet(args.output)
        print(f"Wrote offline features to {args.output}")
        features.select(
            "card_id", "event_timestamp", "txn_count_5m", "txn_amount_avg_1h", "label"
        ).show(10, truncate=False)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
