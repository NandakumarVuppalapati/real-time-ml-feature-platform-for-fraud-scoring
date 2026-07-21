# One-shot job: registers feature_repo/definitions.py into the Feast
# registry (data/registry.db) and provisions the online store. Runs once at
# stack startup; other services wait on it via
# `condition: service_completed_successfully` in docker-compose.yml.
FROM python:3.11-slim

RUN pip install --no-cache-dir feast==0.65.0 pandas pyarrow

WORKDIR /feature_repo
COPY feature_repo/ .

CMD ["python", "bootstrap.py"]
