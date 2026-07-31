"""
PyFlink DataStream job: the real-time half of the feature platform.

Consumes the raw transaction stream from Kafka (topic `transactions.raw`,
written by ingestion/producer.py) and computes trailing-window velocity/amount
features per card_id, emitting one updated feature row per incoming
transaction to the Kafka topic `card_features`. A small consumer
(streaming/feast_pusher.py) then pushes those rows into Feast's online store
so serving/main.py can read them with single-digit-millisecond latency.

Feature definitions here MUST match feature_repo/definitions.py and
training/prepare_offline_features.py exactly (same names, same window
lengths) — that agreement is what prevents training/serving skew. See the
docstring in feature_repo/definitions.py.

--- Why DataStream + KeyedProcessFunction instead of SQL OVER windows/joins ---

The original implementation used four SQL `OVER` windows (1m/5m/1h counts +
sums + a LAG-based "time since last txn") joined back together on
(card_id, event_time), because Flink's streaming OVER-aggregate operator
only supports one window bound per query. That design hit a real, structural
bug: the LAG view has no RANGE bound, so it processes and emits almost 1:1
with its input, while the RANGE-bounded 1m/5m/1h views buffer much more
internally and fall far behind in wall-clock time — even though every view
is derived from the exact same source rows. By the time a windowed view's
row reached the final join, the LAG branch's matching row (which arrived
almost immediately) had already been evicted from the interval join's state,
so the final join produced zero output no matter how long the job ran.

This function sidesteps that whole bug class by computing every feature for
a card in a single pass over that card's own transaction history, with no
joins and no dependency on Flink's watermark mechanism at all:

- `transactions.raw` is a single partition (see docker-compose.yml's
  kafka-init comment), and ingestion/producer.py sorts the dataset by `Time`
  before replaying it — so events for any one card_id arrive to this
  function already in non-decreasing event-time order. That means a plain
  per-key list, trimmed against each event's own embedded event_time, gives
  exactly the same trailing-window semantics the SQL OVER windows were
  after, without needing watermarks, timers, or a join at all.
- Every incoming transaction produces exactly one output row, computed
  directly from that card's state — so there's no cross-stream matching to
  race or expire.

Run (inside the `flink-taskmanager`/`flink-jobmanager` container, or via
`flink run -py streaming/flink_feature_job.py` against a running cluster —
see docker-compose.yml and streaming/Dockerfile.flink):

    flink run -py streaming/flink_feature_job.py
"""
import json
import os
from datetime import datetime

from pyflink.common import Types
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.watermark_strategy import WatermarkStrategy
from pyflink.datastream import StreamExecutionEnvironment, KeyedProcessFunction, RuntimeContext
from pyflink.datastream.connectors.kafka import (
    KafkaSource,
    KafkaOffsetsInitializer,
    KafkaSink,
    KafkaRecordSerializationSchema,
)
from pyflink.datastream.state import ValueStateDescriptor

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
SOURCE_TOPIC = os.environ.get("SOURCE_TOPIC", "transactions.raw")
SINK_TOPIC = os.environ.get("SINK_TOPIC", "card_features")

ONE_MINUTE_MS = 60_000
FIVE_MINUTES_MS = 300_000
ONE_HOUR_MS = 3_600_000


def _parse_iso_to_millis(event_time: str) -> int:
    # ingestion/producer.py emits event_time via
    # `event_time.isoformat().replace("+00:00", "Z")` — reversing that
    # substitution round-trips cleanly through datetime.fromisoformat().
    dt = datetime.fromisoformat(event_time.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)


class CardFeatureFunction(KeyedProcessFunction):
    """Maintains a trailing 1-hour transaction history per card_id and emits
    an updated feature snapshot on every new transaction for that card."""

    def open(self, runtime_context: RuntimeContext):
        self.state = runtime_context.get_state(
            ValueStateDescriptor("txn_history", Types.PICKLED_BYTE_ARRAY())
        )

    def process_element(self, value, ctx):
        event = json.loads(value)
        card_id = event["card_id"]
        event_time_ms = _parse_iso_to_millis(event["event_time"])
        amount = float(event["amount"])
        merchant = event["merchant_category"]

        history = self.state.value() or []
        # (timestamp_ms, amount, merchant_category) tuples, oldest first.
        history.append((event_time_ms, amount, merchant))
        history = [h for h in history if h[0] >= event_time_ms - ONE_HOUR_MS]
        history.sort(key=lambda h: h[0])  # cheap insurance, see module docstring

        in_1m = [h for h in history if h[0] >= event_time_ms - ONE_MINUTE_MS]
        in_5m = [h for h in history if h[0] >= event_time_ms - FIVE_MINUTES_MS]
        in_1h = history

        txn_count_1h = len(in_1h)
        txn_amount_avg_1h = (sum(h[1] for h in in_1h) / txn_count_1h) if txn_count_1h else 0.0

        if len(history) >= 2:
            prev_time_ms, _, prev_merchant = history[-2]
            time_since_last_txn_sec = (event_time_ms - prev_time_ms) / 1000.0
            last_merchant_category = prev_merchant
        else:
            time_since_last_txn_sec = 0.0
            last_merchant_category = "unknown"

        self.state.update(history)

        out = {
            "card_id": card_id,
            "event_timestamp": event["event_time"],
            "txn_count_1m": len(in_1m),
            "txn_count_5m": len(in_5m),
            "txn_count_1h": txn_count_1h,
            "txn_amount_sum_5m": sum(h[1] for h in in_5m),
            "txn_amount_avg_1h": txn_amount_avg_1h,
            "time_since_last_txn_sec": time_since_last_txn_sec,
            "distinct_merchant_count_1h": len({h[2] for h in in_1h}),
            "last_merchant_category": last_merchant_category,
        }
        yield json.dumps(out)


def build_source() -> KafkaSource:
    return (
        KafkaSource.builder()
        .set_bootstrap_servers(KAFKA_BOOTSTRAP)
        .set_topics(SOURCE_TOPIC)
        .set_group_id("flink-card-feature-job")
        .set_starting_offsets(KafkaOffsetsInitializer.latest())
        .set_value_only_deserializer(SimpleStringSchema())
        .build()
    )


def build_sink() -> KafkaSink:
    serializer = (
        KafkaRecordSerializationSchema.builder()
        .set_topic(SINK_TOPIC)
        .set_value_serialization_schema(SimpleStringSchema())
        .build()
    )
    return (
        KafkaSink.builder()
        .set_bootstrap_servers(KAFKA_BOOTSTRAP)
        .set_record_serializer(serializer)
        .build()
    )


def main() -> None:
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)

    source = env.from_source(build_source(), WatermarkStrategy.no_watermarks(), "kafka-source")
    keyed = source.key_by(lambda raw: json.loads(raw)["card_id"], key_type=Types.STRING())
    features = keyed.process(CardFeatureFunction(), output_type=Types.STRING())
    features.sink_to(build_sink())

    env.execute("fraud-card-feature-job")


if __name__ == "__main__":
    main()
