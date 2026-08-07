# Build from the project root: docker build -f training/Dockerfile .
# Runs prepare_offline_features.py (PySpark) and train.py (MLflow) as
# one-shot jobs, e.g.:
#   docker compose run --rm training python prepare_offline_features.py
#   docker compose run --rm training python train.py
FROM python:3.11-slim

# PySpark needs a JVM. openjdk-21, not 17: the python:3.11-slim base image's
# underlying Debian release moves forward over time on rebuilds (it's not
# pinned to a specific point release), and a newer Debian dropped the
# openjdk-17 packages in favor of openjdk-21 -- hit live when this image was
# rebuilt months after it was first written. PySpark 3.5.x supports Java 21
# fine, so this is a same-behavior swap, not a downgrade/workaround.
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-21-jre-headless procps && \
    rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

WORKDIR /app

COPY training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY training/*.py ./
# prepare_offline_features.py imports SAMPLE_PCT_DEFAULT from here (used only
# by its optional --sample-pct escape hatch; see that script's docstring).
# train.py no longer needs this -- it reads the offline parquet directly.
# Copied alone, not producer.py itself, so this image doesn't need
# kafka-python installed just to satisfy an unused import.
COPY ingestion/enrich.py .

# data/, feature_repo/, and mlflow.db are bind-mounted at runtime so
# training output lands in the same places every other service reads from.
