"""
Validates the Spark batch feature computation against a tiny synthetic
frame: two transactions on the same card 30 seconds apart should show
txn_count_5m == 2 and time_since_last_txn_sec == 30 on the second row, and
a transaction on a different card shouldn't pollute the first card's
window. This is the offline half of the training/serving parity described
in feature_repo/definitions.py — if this drifts from
streaming/flink_feature_job.py's window logic, that's a training/serving
skew bug.

Spins up a local (single-process) Spark session — no cluster needed — so
this is slower than the other tests (JVM startup) but still self-contained.
"""
import os

import pytest


@pytest.fixture(scope="module")
def spark():
    pyspark = pytest.importorskip("pyspark")
    from training.prepare_offline_features import build_spark

    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    session = build_spark()
    yield session
    session.stop()


def test_rolling_features_on_synthetic_transactions(spark, tmp_path):
    from training.prepare_offline_features import BASE_TIME, compute_features

    csv_path = tmp_path / "mini.csv"
    csv_path.write_text(
        "Time,V1,Amount,Class,card_id,merchant_category\n"
        "0,0.1,10.0,0,card_A,mcc_1\n"
        "30,0.1,20.0,0,card_A,mcc_2\n"
        "10,0.1,999.0,0,card_B,mcc_9\n"
    )

    result = compute_features(spark, str(csv_path)).orderBy("card_id", "event_timestamp").toPandas()

    card_a = result[result.card_id == "card_A"].reset_index(drop=True)
    assert card_a.loc[0, "txn_count_5m"] == 1
    assert card_a.loc[0, "time_since_last_txn_sec"] is None or card_a.loc[0, "time_since_last_txn_sec"] != card_a.loc[0, "time_since_last_txn_sec"]  # NaN
    assert card_a.loc[1, "txn_count_5m"] == 2
    assert card_a.loc[1, "time_since_last_txn_sec"] == 30
    assert card_a.loc[1, "txn_amount_sum_5m"] == 30.0
    assert card_a.loc[1, "last_merchant_category"] == "mcc_1"

    card_b = result[result.card_id == "card_B"].reset_index(drop=True)
    assert card_b.loc[0, "txn_count_5m"] == 1  # unaffected by card_A's transactions
