# Build from the project root: docker build -f serving/Dockerfile .
FROM python:3.11-slim

WORKDIR /app

COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serving/ ./serving/

# feature_repo/ is bind-mounted at runtime (see docker-compose.yml) so the
# API always reads the same registry/config the pusher and training jobs
# write to.
EXPOSE 8000
CMD ["uvicorn", "serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
