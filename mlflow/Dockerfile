# Minimal MLflow tracking server + model registry. There's no official
# lightweight MLflow server image, so this builds one: pip install mlflow,
# run `mlflow server`. Backend store and artifact root are bind-mounted
# volumes (see docker-compose.yml) so runs/models survive container
# restarts and are inspectable from the host.
FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==3.14.0

EXPOSE 5000

# --allowed-hosts: MLflow 2.14+ validates the incoming Host header by
# default (DNS-rebinding protection) and rejects anything not explicitly
# listed. Other containers reach this one via the Docker Compose service
# name "mlflow" (e.g. http://mlflow:5000), which isn't in the default
# allowlist -- hit live as "403 Invalid Host header - possible DNS rebinding
# attack detected" from training/train.py. localhost/127.0.0.1 are also
# listed since the healthcheck below calls the server from inside its own
# container, on :5000 -- and again on :5001 since docker-compose.yml maps
# host port 5001 -> container port 5000, so requests made *from the host*
# (browser, curl) arrive with a "localhost:5001" Host header, not :5000.
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:///mlflow-data/mlflow.db", \
     "--default-artifact-root", "/mlflow-data/artifacts", \
     "--allowed-hosts", "mlflow,mlflow:5000,localhost,localhost:5000,localhost:5001,127.0.0.1,127.0.0.1:5000,127.0.0.1:5001"]
