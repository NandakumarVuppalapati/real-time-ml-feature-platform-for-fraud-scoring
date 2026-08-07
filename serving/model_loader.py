"""
Loads the current "champion"-aliased model from the MLflow Model Registry
and the Feast feature store client, and caches both as process-level
singletons so /score doesn't pay reload cost per request.

Call `reload_model()` (exposed via POST /admin/reload) after training a new
model version and re-aliasing it to @champion, to pick it up without
restarting the API process.
"""
import logging
import os
import threading

import mlflow
import mlflow.xgboost
import pandas as pd
from feast import FeatureStore

logger = logging.getLogger("serving.model_loader")

MODEL_NAME = os.environ.get("MODEL_NAME", "fraud-scorer")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")
FEATURE_REPO_PATH = os.environ.get("FEATURE_REPO_PATH", "feature_repo")

FEATURE_REFS = [
    "card_transaction_features:txn_count_1m",
    "card_transaction_features:txn_count_5m",
    "card_transaction_features:txn_count_1h",
    "card_transaction_features:txn_amount_sum_5m",
    "card_transaction_features:txn_amount_avg_1h",
    "card_transaction_features:time_since_last_txn_sec",
    "card_transaction_features:distinct_merchant_count_1h",
]
FEATURE_COLUMNS = [f.split(":")[1] for f in FEATURE_REFS]

_lock = threading.Lock()
_state = {"model": None, "version": None, "store": None}


def _load_feature_store() -> FeatureStore:
    return FeatureStore(repo_path=FEATURE_REPO_PATH)


def _load_model():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    client = mlflow.MlflowClient()
    model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    # Load via the native xgboost flavor (not mlflow.pyfunc) so we get the
    # real XGBClassifier back with .predict_proba() intact. The generic
    # pyfunc wrapper only exposes .predict(), which for a classifier returns
    # hard 0/1 labels — useless for a fraud *score*, and it also enforces a
    # strict input schema that's brittle across pandas dtype variations.
    #
    # Note: this alias URI resolves correctly on its own -- the earlier
    # "MlflowException: No such artifact: ''" wasn't a registry/alias
    # resolution bug. It was that the artifact *file* had never actually
    # been written to a shared location in the first place (see
    # docker-compose.yml's mlflow-data volume mount on the training and api
    # services for the real root cause).
    model = mlflow.xgboost.load_model(model_uri)
    logger.info("Loaded %s v%s (%s)", MODEL_NAME, model_version.version, model_uri)
    return model, str(model_version.version)


def startup() -> None:
    with _lock:
        _state["store"] = _load_feature_store()
        try:
            _state["model"], _state["version"] = _load_model()
        except Exception as exc:  # noqa: BLE001
            # Don't crash the API if no model has been trained/aliased yet —
            # /health will report model_loaded=False and /score will 503.
            logger.warning("No model available at startup: %s", exc)
            _state["model"], _state["version"] = None, None


def reload_model() -> str:
    with _lock:
        _state["model"], _state["version"] = _load_model()
    return _state["version"]


def is_ready() -> bool:
    return _state["model"] is not None


def model_version() -> str | None:
    return _state["version"]


def get_online_features(card_id: str) -> dict:
    store: FeatureStore = _state["store"]
    resp = store.get_online_features(
        features=FEATURE_REFS, entity_rows=[{"card_id": card_id}]
    ).to_dict()
    features = {col: resp[col][0] for col in FEATURE_COLUMNS}
    # Cold start: card_id not seen by the streaming pipeline yet. Treat as
    # "no prior activity" (all zeros) rather than failing the request — a
    # brand-new card is itself a meaningful fraud signal, not an error.
    missing = [k for k, v in features.items() if v is None]
    if missing:
        logger.info("Cold start for card_id=%s, missing=%s -> defaulting to 0", card_id, missing)
        for k in missing:
            features[k] = 0
    return features


def predict(features_row: dict) -> float:
    # dtype should match training (float64 for every column — see the
    # comment in training/train.py); the native xgboost flavor is lenient
    # about this compared to pyfunc's strict schema enforcement, but keeping
    # them consistent avoids any subtle behavior differences.
    df = pd.DataFrame([features_row], columns=FEATURE_COLUMNS).astype("float64")
    proba = _state["model"].predict_proba(df)
    return float(proba[0][1])  # P(Class == 1), i.e. P(fraud)
