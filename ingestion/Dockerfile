# Build from the project root: docker build -f ingestion/Dockerfile .
FROM python:3.11-slim

WORKDIR /app

COPY ingestion/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ingestion/producer.py ingestion/enrich.py .

# data/ is bind-mounted at runtime (see docker-compose.yml) rather than
# baked into the image — the raw Kaggle CSV is ~150MB and gitignored, and
# you shouldn't need to rebuild the image just to point the producer at a
# different file.
ENTRYPOINT ["python", "producer.py"]
