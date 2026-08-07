"""
Trains the fraud classifier on point-in-time-correct historical features,
and logs the run (params, metrics, model artifact) to MLflow, registering it
in the MLflow Model Registry so serving/main.py can load
"models:/fraud-scorer@champion" instead of a hardcoded path.

Reads training/prepare_offline_features.py's output parquet directly rather
than through Feast's `get_historical_features()`. That's a deliberate
change, not a shortcut -- see load_training_data()'s docstring for why it's
still point-in-time-correct with zero leakage risk here, and the git history
for the live debugging trail that motivated it (Feast's local/file-based
point-in-time join OOM-killed repeatedly on this dataset's ~1.3M-row scale,
even after reducing the join down to a matched ~46k rows on both sides, on a
machine with ~7.6GB genuinely available to Docker -- a real algorithmic
memory-scaling limitation of that join implementation, not just raw data
volume). Feast still owns the feature *definitions* and still serves these
exact features online (feature_repo/definitions.py, streaming/feast_pusher.py,
serving/main.py) -- this only changes how the offline training pull retrieves
them, and lets training run against the full dataset instead of a sample.

Usage:
    python training/train.py
    python training/train.py --offline-parquet data/offline/card_features.parquet
"""
import argparse
import os

import mlflow
import mlflow.xgboost
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

FEATURE_COLUMNS = [
    "txn_count_1m",
    "txn_count_5m",
    "txn_count_1h",
    "txn_amount_sum_5m",
    "txn_amount_avg_1h",
    "time_since_last_txn_sec",
    "distinct_merchant_count_1h",
]
MODEL_NAME = "fraud-scorer"


def load_training_data(offline_parquet_path: str) -> pd.DataFrame:
    """Loads point-in-time-correct training data straight from the offline
    parquet, bypassing Feast's get_historical_features().

    Why this is still point-in-time-correct, not a shortcut that risks
    leakage: Feast's asof point-in-time join exists to correctly handle the
    *general* case where an entity's label timestamp and its feature
    snapshot's timestamp differ -- e.g. a chargeback confirmed three days
    after the transaction it applies to, where you must be careful to pull
    features as they stood at transaction time, not as they stand now.

    That's not our case. training/prepare_offline_features.py emits exactly
    one feature row per transaction, computed from that card's trailing
    history strictly up to and including that transaction's own
    event_timestamp -- the label for a row and the feature snapshot for that
    same row share the identical timestamp *by construction*, because
    they're the same event. An asof join against an exact timestamp match
    degenerates to precisely this direct read; Feast's general-purpose
    implementation just computes that same answer far more expensively (see
    module docstring). Reading the parquet directly returns the identical
    training set, just without the memory blowup.
    """
    return pd.read_parquet(offline_parquet_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-parquet", default="data/offline/card_features.parquet")
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()

    training_df = load_training_data(args.offline_parquet)
    print(f"Loaded {len(training_df)} labeled transactions "
          f"({training_df['label'].sum()} fraud, {training_df['label'].mean():.3%} rate)")

    # Cold-start rows (a card's very first transaction) have no trailing
    # window history yet — fill rather than drop, since "no prior activity"
    # is itself informative for a fraud model, not missing data to discard.
    training_df[FEATURE_COLUMNS] = training_df[FEATURE_COLUMNS].fillna(0)

    # Cast every feature to float64 up front. Mixed int32/int64/float32
    # dtypes are fine for training, but MLflow's strict schema enforcement
    # at serving time will reject a request DataFrame whose dtypes don't
    # match bit-for-bit. Scoring a rolling count as a float loses nothing a
    # tree model cares about, and it removes an entire class of "works in
    # training, 500s in serving" bugs.
    X = training_df[FEATURE_COLUMNS].astype("float64")
    y = training_df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    scale_pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("fraud-scoring")

    params = dict(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("feature_columns", FEATURE_COLUMNS)

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        auc_roc = roc_auc_score(y_test, y_pred_proba)
        auc_pr = average_precision_score(y_test, y_pred_proba)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        metrics = {
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        }
        mlflow.log_metrics(metrics)
        print("Evaluation:", {k: round(v, 4) if isinstance(v, float) else v
                               for k, v in metrics.items()})

        model_info = mlflow.xgboost.log_model(
            model,
            name="model",
            registered_model_name=MODEL_NAME,
            input_example=X_train.head(3),
        )

        version = model_info.registered_model_version
        MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", version)
        print(
            f"Model registered as '{MODEL_NAME}' v{version} in the MLflow Model "
            f"Registry, aliased '@champion' — serving/main.py loads "
            f"models:/{MODEL_NAME}@champion"
        )


if __name__ == "__main__":
    main()
