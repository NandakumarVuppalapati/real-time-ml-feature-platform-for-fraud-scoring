# Real-Time ML Feature Platform for Fraud Scoring

A working, runnable version of the architecture behind most production fraud-scoring
systems: transactions stream in through Kafka, Flink computes rolling behavioral features
in real time, Feast serves those same feature definitions consistently to both training
and serving, and a model trained on point-in-time-correct historical data scores
transactions in under a second through a FastAPI endpoint.

Built to close the gap between "I know what Kafka/Flink/Feast are" and "I've wired them
together and hit the failure modes." Runs entirely on free, open-source tooling via
Docker Compose — no paid cloud services required.

```
Kafka → Flink → (push) → Feast (Redis online / Parquet offline) → FastAPI
                              ↑                         ↓
                    Spark (batch features)      MLflow (registry)
```

## Why this exists

Most fraud pipelines fall into one of two categories: pure batch (nightly Spark jobs,
features that are hours stale by the time a transaction is scored) or an ad-hoc real-time
system where someone re-implements the batch feature logic in a different language for
low-latency serving — and the two implementations quietly drift apart. That drift is
**training/serving skew**, and it's one of the more expensive, hard-to-detect bug classes
in applied ML.

This project's actual point is the plumbing that prevents that: one feature definition
(`feature_repo/definitions.py`), computed by two engines (Flink for the online path, Spark
for offline/training), served through one abstraction (Feast) so both paths agree on what
"features as of this moment" means.

## Architecture

```mermaid
flowchart LR
    subgraph Ingestion
        P[ingestion/producer.py] -->|replays Kaggle dataset| K[(Kafka<br/>transactions.raw)]
    end

    subgraph "Real-time path"
        K --> F[Flink SQL job<br/>rolling windows per card_id]
        F -->|card_features topic| PU[feast_pusher.py]
        PU -->|push ONLINE| R[(Redis<br/>Feast online store)]
    end

    subgraph "Batch / training path"
        RAW[(Kaggle creditcard.csv)] --> SP[Spark batch job<br/>prepare_offline_features.py]
        SP -->|same feature defs| PQ[(Parquet<br/>Feast offline store)]
        PQ --> TR[train.py:<br/>direct parquet read*]
        TR -->|log + register| ML[(MLflow<br/>tracking + registry)]
    end

    subgraph Serving
        API[FastAPI /score] -->|get_online_features| R
        API -->|load models:/fraud-scorer@champion| ML
    end

    style R fill:#c0392b,color:#fff
    style ML fill:#2980b9,color:#fff
    style PQ fill:#27ae60,color:#fff
```

*`train.py` reads the offline parquet directly rather than through Feast's
`get_historical_features()` — a deliberate, documented change, not a shortcut. See
[The OOM that almost forced a downsample](#the-oom-that-almost-forced-a-downsample).

## Stack, and why each piece is here

| Component | Role | Free/local substitute used |
|---|---|---|
| **Kafka** | Event backbone for the raw transaction stream | `apache/kafka` official image, single-node KRaft mode (no ZooKeeper) |
| **Flink** | Real-time windowed feature computation | Open-source Flink cluster (jobmanager + taskmanager) via Docker |
| **Feast** | Single feature-definition contract for online + offline | Open-source Feast, Redis online store, file/Parquet offline store |
| **Spark** | Batch feature computation for training data | Open-source PySpark, `local[*]` — same code runs unmodified on Databricks/EMR |
| **MLflow** | Experiment tracking + model registry | Self-hosted MLflow server, SQLite backend |
| **FastAPI** | Sub-second scoring endpoint | — |
| **Docker Compose** | Orchestrates all of the above locally | — |

The original design referenced Databricks and AWS directly. Those cost money at any real
scale, so this repo substitutes their open-source foundations (Spark instead of managed
Databricks compute, local Kafka/Flink/Redis instead of MSK/KDA/ElastiCache). The code is
structured so that swap is mechanical: change `feature_store.yaml`'s offline/online store
config and `prepare_offline_features.py`'s Spark master, and the same feature definitions
and training/serving code run on managed infrastructure without modification. See
[Cloud deployment path](#cloud-deployment-path) below.

## The dataset

Trained and demoed end-to-end on the full [kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
dataset — **1,296,675 real-schema transactions** (Sparkov-simulated, but with genuine
card numbers, merchants, timestamps, geolocation, and cardholder fields, not PCA-anonymized
components), reflecting the order of magnitude an actual card issuer's fraud team works
against, not a toy sample. `card_id` is derived directly from each transaction's `cc_num`
— no synthetic key assignment needed, unlike the small anonymized
[mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset
this project also supports as a zero-setup fallback (`data/sample/sample_creditcard.csv`,
committed to the repo) for anyone who wants to run it without a Kaggle account.
`ingestion/enrich.py` detects which schema is on disk and normalizes either into the same
`card_id` / `merchant_category` / `amount` / `event_time` / `label` shape the rest of the
pipeline consumes, so nothing downstream needs to know which dataset is in play.

## Results

Trained on the full 1,296,675-row dataset (not a downsampled subset — see
[below](#the-oom-that-almost-forced-a-downsample) for why that mattered):

| Metric | Value |
|---|---|
| AUC-ROC | 0.9749 |
| AUC-PR | 0.5289 |
| Recall | 90.0% |
| Precision | 6.9% |
| F1 | 0.129 |
| Training rows | 972,506 (75% split) |
| Test rows | 324,169 (25% split) |

Precision is intentionally low relative to recall: `scale_pos_weight` is tuned in
`train.py` to catch 9 out of 10 fraud cases, trading off false positives — the standard
cost asymmetry in fraud scoring, where a missed fraud is far more expensive than a
manually-reviewed false alarm.

<table>
<tr><td><img src="docs/screenshots/flink-job-graph.png" width="400"><br/><sub>Flink job graph, live — 9,000 records streamed through the windowed feature job</sub></td>
<td><img src="docs/screenshots/mlflow-run-metrics.png" width="400"><br/><sub>MLflow run <code>silent-donkey-396</code> — real metrics, registered as <code>fraud-scorer v3</code></sub></td></tr>
</table>

<img src="docs/screenshots/api-score-response.png" width="500"><br/>
<sub>A live <code>/score</code> call, scored in 77.91ms using features pulled from Redis at request time</sub>

## The OOM that almost forced a downsample

Feast's local/file-based point-in-time join (`get_historical_features()`) loads its
offline source into memory regardless of how small the training entity set is — it
OOM-killed repeatedly on this dataset's scale, even after reducing the join down to a
matched ~46,000-row slice on both sides, on a machine with ~7.6GB genuinely available to
Docker. The tempting fix was downsampling the training set to fit; that would have quietly
defeated the entire point of using a dataset at production scale.

The actual fix (`training/train.py`): recognize that Feast's asof point-in-time join
exists to handle the general case where a label's timestamp and its feature snapshot's
timestamp differ. Here they don't — `prepare_offline_features.py` emits exactly one
feature row per transaction, computed from that card's trailing history strictly up to and
including that transaction's own timestamp, so the label and the feature snapshot for a
row share an identical timestamp by construction. An asof join against an exact match
degenerates to a direct read. Reading the offline parquet straight into pandas returns the
identical, point-in-time-correct training set — just without the memory blowup — so
training runs against the full dataset, not a sample.

## Repository layout

```
ingestion/       Kafka producer — replays the dataset as a live event stream
streaming/       Flink SQL job (real-time features) + Feast pusher
feature_repo/    Feast feature definitions, config, and bootstrap
training/        Spark batch feature computation + model training (MLflow)
serving/         FastAPI scoring service
data/            Sample dataset (committed), raw/offline data (gitignored)
tests/           pytest suite — runs without Docker, Kafka, or Redis
mlflow/          Dockerfile for the MLflow tracking server
```

## Running it

**Prerequisites:** Docker Desktop (free), ~4GB RAM available to Docker. Optionally a free
[Kaggle account](https://www.kaggle.com) for the real dataset (the repo runs end to end
without one, using a small synthetic sample).

```bash
git clone <this-repo>
cd real-time-ml-feature-platform-for-fraud-scoring

# 1. Bring up Kafka, Redis, MLflow, Flink, Feast, and the API
make up
#    API docs:   http://localhost:8000/docs
#    MLflow UI:  http://localhost:5001
#    Flink UI:   http://localhost:8081

# 2. (Optional but recommended) Get the real ~1.3M-row dataset instead of the tiny sample
#    Needs a free Kaggle account -> Account Settings -> API -> Create New Token
pip install kaggle
python data/download_dataset.py

# 3. Compute offline features (Spark) and train + register a model (MLflow) --
#    on the full dataset by default, no sampling
make offline-features
make train

# 4. Reload the newly trained model into the running API
curl -X POST http://localhost:8000/admin/reload

# 5. Start replaying transactions into Kafka -> Flink -> Feast, live
make demo

# 6. Score a transaction (use a card_id/amount/category actually present in
#    data/raw/fraudTest.csv if you're running against the real dataset)
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"card_id": "card_2291163933867244", "amount": 2.86, "merchant_category": "personal_care"}'
```

Without Docker, each piece also runs directly (`pip install -r requirements.txt`, see
each subfolder's own `requirements.txt` for its minimal dependency set, and `.env.example`
for the environment variables to export).

## Running the tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

The suite (14 tests) validates the Feast feature definitions (apply, push, online read,
point-in-time historical read), the Spark rolling-window feature logic on synthetic data,
the producer's enrichment/serialization functions, and the FastAPI contract — all without
Docker, Kafka, or Redis running. CI (`.github/workflows/ci.yml`) runs this suite plus a
Docker image build check on every push.

## The feature freshness / completeness trade-off

This is the part of the original project description called out as "in progress," and
it's worth being specific about rather than hand-waving:

Flink's windowed aggregation (`streaming/flink_feature_job.py`) uses a watermark
(`MAX_OUT_OF_ORDERNESS`, default 5 seconds) to decide when a window is "done" and safe to
emit. Raise it and features are more complete (a late-arriving event still gets counted in
its window) but staler; lower it and features are fresher but a delayed producer retry or
clock-skewed event can silently miss its window. There's no universally correct value —
it's a trade-off tuned against your actual event-arrival distribution (p99 lateness) and
your fraud model's actual sensitivity to a missed count.

A second, related gap: the online path (Flink, real time) and offline path (Spark, batch)
are two independent implementations of the same feature logic, kept consistent only by
convention (both read `feature_repo/definitions.py`'s docstring and must match its window
definitions by hand). A more mature version of this platform would either (a) run the
same Flink job in a bounded/batch execution mode against historical data for training
features too, or (b) adopt a framework (e.g. Chronon, or Feast's newer on-demand transform
support) that compiles one feature definition into both the streaming and batch execution
plans, removing the "two implementations that must agree" problem entirely.

## Cloud deployment path

Nothing here is architecturally locked to local Docker:

- **Kafka** → Amazon MSK (drop-in — same Kafka protocol)
- **Flink** → Amazon Kinesis Data Analytics or a self-managed Flink cluster on EKS
- **Redis (Feast online store)** → Amazon ElastiCache for Redis, or DynamoDB with Feast's
  DynamoDB online store connector
- **Parquet (Feast offline store)** → S3, with Feast's offline store config pointed at
  `s3://...` paths
- **Spark (`prepare_offline_features.py`)** → runs unmodified on Databricks or EMR; only
  `SparkSession.builder.master(...)` changes
- **MLflow** → point `MLFLOW_TRACKING_URI` at a hosted MLflow server (e.g. on ECS/EKS) or
  a managed offering
- **FastAPI** → any container platform (ECS Fargate, EKS, App Runner)

## License

MIT — see [LICENSE](LICENSE).
