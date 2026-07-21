"""
Shared pytest fixtures.

`feast_repo` spins up a throwaway Feast repo (sqlite registry + sqlite
online store, instead of the docker-compose stack's Postgres-registry-free
file registry + Redis) in a temp directory, using the *real*
feature_repo/definitions.py from this project. This lets tests exercise the
actual feature definitions — push, online read, historical (point-in-time)
read — without Docker, Redis, or Kafka running. Swapping online_store.type
from "redis" to "sqlite" changes nothing about which code path is under
test; Feast's online store implementations share the same interface.
"""
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

PLACEHOLDER_ROW = {
    "card_id": "placeholder_card",
    "event_timestamp": datetime.now(timezone.utc),
    "created_timestamp": datetime.now(timezone.utc),
    "txn_count_1m": 0,
    "txn_count_5m": 0,
    "txn_count_1h": 0,
    "txn_amount_sum_5m": 0.0,
    "txn_amount_avg_1h": 0.0,
    "time_since_last_txn_sec": 0.0,
    "distinct_merchant_count_1h": 0,
    "last_merchant_category": "none",
    "Amount": 0.0,
    "Class": 0,
    "merchant_category": "none",
}


@pytest.fixture()
def feast_repo(tmp_path: Path):
    repo_dir = tmp_path / "feature_repo"
    repo_dir.mkdir()
    shutil.copy(REPO_ROOT / "feature_repo" / "definitions.py", repo_dir / "definitions.py")

    (repo_dir / "feature_store.yaml").write_text(
        """
project: fraud_feature_platform_test
provider: local
registry: data/registry.db
online_store:
  type: sqlite
  path: data/online_store.db
offline_store:
  type: file
entity_key_serialization_version: 3
auth:
  type: no_auth
"""
    )

    data_dir = tmp_path / "data" / "offline"
    data_dir.mkdir(parents=True)
    pd.DataFrame([PLACEHOLDER_ROW]).to_parquet(data_dir / "card_features.parquet")

    subprocess.run(
        ["feast", "apply"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )

    from feast import FeatureStore

    store = FeatureStore(repo_path=str(repo_dir))
    store.offline_parquet_path = data_dir / "card_features.parquet"  # test convenience
    yield store
