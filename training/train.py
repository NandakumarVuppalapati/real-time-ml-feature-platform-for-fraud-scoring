"""
Trains the fraud classifier on point-in-time-correct historical features
pulled from Feast, and logs the run (params, metrics, model artifact) to
MLflow, registering it in the MLflow Model Registry so serving/main.py can
load "models:/fraud-scorer/Production" (or /Staging, /latest) instead of a
hardcoded path.

Why go through Feast here instead of just reading the parquet directly: this
is the step that actually prevents training/serving skew. get_historical_features
performs a point-in-time join — for every (card_id, event_timestamp) in the
label set, it pulls the feature values that were true *as of that moment*,
using the same feature definitions the online store serves at request time.
Skip this and read the parquet with a plain merge, and it's easy to
accidentally leak future feature values into training.

Usage:
    python training/train.py
    MLFLOW_TRACKING_URI=http://localhost:5001 python training/train.py --input data/raw/creditcard_enriched.csv
"""
import argparse
import os

import mlflow
import mlflow.xgboost
import pandas as pd
from feast import FeatureStore
from mlflow import MlflowClient
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

FEATURES = [
    "card_transaction_features:txn_count_1m",
    "card_transaction_features:txn_count_5m",
    "card_transaction_features:txn_count_1h",
    "card_transaction_features:txn_amount_sum_5m",
    "card_transaction_features:txn_amount_avg_1h",
    "card_transaction_features:time_since_last_txn_sec",
    "card_transaction_features:distinct_merchant_count_1h",
]
FEATURE_COLUMNS = [f.split(":")[1] for f in FEATURES]
MODEL_NAME = "fraud-scorer"


def build_entity_df(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    base_time = pd.Timestamp("2013-09-01", tz="UTC")
    df["event_timestamp"] = base_time + pd.to_timedelta(df["Time"], unit="s")
    return df[["card_id", "event_timestamp", "Amount", "Class"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--repo-path", default="feature_repo")
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()

    input_path = args.input
    if input_path is None:
        raw = "data/raw/creditcard_enriched.csv"
        input_path = raw if os.path.exists(raw) else "data/sample/sample_creditcard.csv"
        print(f"--input not given, using {input_path}")

    entity_df = build_entity_df(input_path)
    print(f"Loaded {len(entity_df)} labeled transactions "
          f"({entity_df['Class'].sum()} fraud, {entity_df['Class'].mean():.3%} rate)")

    store = FeatureStore(repo_path=args.repo_path)
    training_df = store.get_historical_features(
        entity_df=entity_df, features=FEATURES
    ).to_df()

    # Cold-start rows (a card's very first transaction) have no trailing
    # window history yet — fill rather than drop, since "no prior activity"
    # is itself informative for a fraud model, not missing data to discard.
    training_df[FEATURE_COLUMNS] = training_df[FEATURE_COLUMNS].fillna(0)

    # Cast every feature to float64 up front. Feast returns Int64 columns as
    # pandas nullable/np.int64 and Float32 columns as float32; left mixed,
    # MLflow's strict schema enforcement at serving time will reject a
    # request DataFrame whose dtypes don't match bit-for-bit (int32 vs
    # int64, float32 vs float64). Scoring a rolling count as a float loses
    # nothing a tree model cares about, and it removes an entire class of
    # "works in training, 500s in serving" bugs.
    X = training_df[FEATURE_COLUMNS].astype("float64")
    y = training_df["Class"]

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
            artifact_path="model",
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
