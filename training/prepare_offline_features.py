"""
Batch feature computation — the offline half of the feature platform.

This recomputes, over historical data, the *exact same* feature definitions
that streaming/flink_feature_job.py computes in real time (see the docstring
in feature_repo/definitions.py for why that consistency matters). It's the
free/local stand-in for what "Databricks" does in the resume bullet: managed
Spark for batch feature engineering. Runs identically here via open-source
PySpark, on Databricks Community Edition, or on EMR — only the cluster
config changes, not this code.

Input:  data/raw/creditcard.csv (or data/sample/sample_creditcard.csv as a
        fallback so the pipeline is runnable without a Kaggle account)
Output: data/offline/card_features.parquet — read by Feast's FileSource
        (feature_repo/definitions.py) for point-in-time-correct training
        sets, via `feast.get_historical_features`.

Usage:
    python training/prepare_offline_features.py
    python training/prepare_offline_features.py --input data/raw/creditcard.csv
"""
import argparse
import os
from datetime import datetime, timezone

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

BASE_TIME = datetime(2013, 9, 1, tzinfo=timezone.utc)  # arbitrary anchor; the
# Kaggle dataset's "Time" column is seconds elapsed since the first
# transaction in the sample, not a real calendar timestamp.


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

    df = df.withColumn(
        "event_timestamp",
        (F.lit(BASE_TIME.timestamp()) + F.col("Time")).cast("timestamp"),
    ).withColumn("event_ts_long", F.col("Time").cast("long"))

    order_by_time = F.col("event_ts_long")
    w1m = Window.partitionBy("card_id").orderBy(order_by_time).rangeBetween(-60, 0)
    w5m = Window.partitionBy("card_id").orderBy(order_by_time).rangeBetween(-300, 0)
    w1h = Window.partitionBy("card_id").orderBy(order_by_time).rangeBetween(-3600, 0)
    order_only = Window.partitionBy("card_id").orderBy(order_by_time)

    features = (
        df.withColumn("txn_count_1m", F.count(F.lit(1)).over(w1m))
        .withColumn("txn_count_5m", F.count(F.lit(1)).over(w5m))
        .withColumn("txn_count_1h", F.count(F.lit(1)).over(w1h))
        .withColumn("txn_amount_sum_5m", F.sum("Amount").over(w5m))
        .withColumn("txn_amount_avg_1h", F.avg("Amount").over(w1h))
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
        "Amount",
        "Class",
        "merchant_category",
    ]
    return features.select(out_cols)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="Path to source CSV")
    parser.add_argument(
        "--output", default="data/offline/card_features.parquet", help="Output parquet path"
    )
    args = parser.parse_args()

    input_path = args.input
    if input_path is None:
        raw = "data/raw/creditcard_enriched.csv"
        sample = "data/sample/sample_creditcard.csv"
        input_path = raw if os.path.exists(raw) else sample
        print(f"--input not given, using {input_path}")

    spark = build_spark()
    try:
        features = compute_features(spark, input_path)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        features.coalesce(1).write.mode("overwrite").parquet(args.output)
        print(f"Wrote offline features to {args.output}")
        features.select(
            "card_id", "event_timestamp", "txn_count_5m", "txn_amount_avg_1h", "Class"
        ).show(10, truncate=False)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
