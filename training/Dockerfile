# Build from the project root: docker build -f training/Dockerfile .
# Runs prepare_offline_features.py (PySpark) and train.py (MLflow) as
# one-shot jobs, e.g.:
#   docker compose run --rm training python prepare_offline_features.py
#   docker compose run --rm training python train.py
FROM python:3.11-slim

# PySpark needs a JVM.
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jre-headless procps && \
    rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

WORKDIR /app

COPY training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY training/*.py ./

# data/, feature_repo/, and mlflow.db are bind-mounted at runtime so
# training output lands in the same places every other service reads from.
