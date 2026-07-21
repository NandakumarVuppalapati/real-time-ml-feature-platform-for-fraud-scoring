# Minimal MLflow tracking server + model registry. There's no official
# lightweight MLflow server image, so this builds one: pip install mlflow,
# run `mlflow server`. Backend store and artifact root are bind-mounted
# volumes (see docker-compose.yml) so runs/models survive container
# restarts and are inspectable from the host.
FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==3.14.0

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:///mlflow-data/mlflow.db", \
     "--default-artifact-root", "/mlflow-data/artifacts"]
