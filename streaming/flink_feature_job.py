"""
PyFlink Table API / SQL job: the real-time half of the feature platform.

Consumes the raw transaction stream from Kafka (topic `transactions.raw`,
written by ingestion/producer.py) and continuously computes trailing-window
velocity/amount features per card_id, emitting one updated feature row per
incoming transaction to the Kafka topic `card_features`. A small consumer
(streaming/feast_pusher.py) then pushes those rows into Feast's online store
so serving/main.py can read them with single-digit-millisecond latency.

Feature definitions here MUST match feature_repo/definitions.py and
training/prepare_offline_features.py exactly (same names, same window
lengths) — that agreement is what prevents training/serving skew. See the
docstring in feature_repo/definitions.py.

Why SQL OVER windows instead of a DataStream ProcessFunction with manual
state: Flink SQL's RANGE OVER window (PARTITION BY card_id ORDER BY
event_time RANGE BETWEEN INTERVAL '5' MINUTE PRECEDING AND CURRENT ROW) is
exactly "rolling aggregate over the trailing N minutes of this key's
events" — the same semantics we need, expressed declaratively instead of
hand-rolled keyed state + timers. It also makes the watermark/lateness
handling explicit and auditable in one place (the DDL), which matters for
the "feature freshness guarantees" this project is built around.

Run (inside the `flink-taskmanager`/`flink-jobmanager` container, or via
`flink run -py streaming/flink_feature_job.py` against a running cluster —
see docker-compose.yml and the streaming/Dockerfile that installs PyFlink +
the Kafka SQL connector jar):

    flink run -py streaming/flink_feature_job.py
"""
import os

from pyflink.table import EnvironmentSettings, TableEnvironment

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
SOURCE_TOPIC = os.environ.get("SOURCE_TOPIC", "transactions.raw")
SINK_TOPIC = os.environ.get("SINK_TOPIC", "card_features")
# How long a late-arriving event is still accepted into its window before
# the watermark closes it out. This is *the* freshness/completeness
# trade-off: raise it and features are more complete but a few hundred ms
# staler; lower it and features are fresher but a late network retry or
# clock-skewed producer can get silently dropped from its window.
MAX_OUT_OF_ORDERNESS = os.environ.get("MAX_OUT_OF_ORDERNESS", "5")  # seconds


def build_table_env() -> TableEnvironment:
    settings = EnvironmentSettings.in_streaming_mode()
    t_env = TableEnvironment.create(settings)
    t_env.get_config().set("pipeline.name", "fraud-card-feature-job")
    t_env.get_config().set("table.exec.source.idle-timeout", "30 s")
    return t_env


SOURCE_DDL = f"""
CREATE TABLE transactions (
    card_id             STRING,
    merchant_category   STRING,
    amount              DOUBLE,
    event_time          TIMESTAMP(3),
    WATERMARK FOR event_time AS event_time - INTERVAL '{MAX_OUT_OF_ORDERNESS}' SECOND
) WITH (
    'connector' = 'kafka',
    'topic' = '{SOURCE_TOPIC}',
    'properties.bootstrap.servers' = '{KAFKA_BOOTSTRAP}',
    'properties.group.id' = 'flink-card-feature-job',
    'scan.startup.mode' = 'latest-offset',
    'format' = 'json',
    'json.timestamp-format.standard' = 'ISO-8601',
    'json.ignore-parse-errors' = 'true'
)
"""

SINK_DDL = f"""
CREATE TABLE card_features (
    card_id                     STRING,
    event_timestamp             TIMESTAMP(3),
    txn_count_1m                BIGINT,
    txn_count_5m                BIGINT,
    txn_count_1h                BIGINT,
    txn_amount_sum_5m           DOUBLE,
    txn_amount_avg_1h           DOUBLE,
    time_since_last_txn_sec     DOUBLE,
    distinct_merchant_count_1h  BIGINT,
    last_merchant_category      STRING,
    PRIMARY KEY (card_id) NOT ENFORCED
) WITH (
    'connector' = 'kafka',
    'topic' = '{SINK_TOPIC}',
    'properties.bootstrap.servers' = '{KAFKA_BOOTSTRAP}',
    'format' = 'json'
)
"""

# NOTE: COUNT(DISTINCT ...) and LAG-style "previous value" aren't directly
# expressible in a single OVER clause the way COUNT/SUM/AVG are, so
# distinct-merchant-count and time-since-last-txn are computed via a second
# pass over an unbounded-preceding ROWS window using LISTAGG-then-count and
# LAG, which Flink SQL supports on OVER windows.
FEATURE_QUERY = """
INSERT INTO card_features
SELECT
    card_id,
    event_time AS event_timestamp,
    COUNT(*) OVER w1m AS txn_count_1m,
    COUNT(*) OVER w5m AS txn_count_5m,
    COUNT(*) OVER w1h AS txn_count_1h,
    SUM(amount) OVER w5m AS txn_amount_sum_5m,
    AVG(amount) OVER w1h AS txn_amount_avg_1h,
    CAST(TIMESTAMPDIFF(SECOND, LAG(event_time) OVER w_order, event_time) AS DOUBLE)
        AS time_since_last_txn_sec,
    COUNT(DISTINCT merchant_category) OVER w1h AS distinct_merchant_count_1h,
    LAG(merchant_category) OVER w_order AS last_merchant_category
FROM transactions
WINDOW
    w1m AS (
        PARTITION BY card_id ORDER BY event_time
        RANGE BETWEEN INTERVAL '1' MINUTE PRECEDING AND CURRENT ROW
    ),
    w5m AS (
        PARTITION BY card_id ORDER BY event_time
        RANGE BETWEEN INTERVAL '5' MINUTE PRECEDING AND CURRENT ROW
    ),
    w1h AS (
        PARTITION BY card_id ORDER BY event_time
        RANGE BETWEEN INTERVAL '1' HOUR PRECEDING AND CURRENT ROW
    ),
    w_order AS (
        PARTITION BY card_id ORDER BY event_time
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )
"""


def main() -> None:
    t_env = build_table_env()
    t_env.execute_sql(SOURCE_DDL)
    t_env.execute_sql(SINK_DDL)
    result = t_env.execute_sql(FEATURE_QUERY)
    result.wait()  # blocks — this is a long-running streaming job


if __name__ == "__main__":
    main()
