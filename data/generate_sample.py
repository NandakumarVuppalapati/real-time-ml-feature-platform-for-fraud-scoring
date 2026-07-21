"""
Generates data/sample/sample_creditcard.csv — a small synthetic stand-in for
the real Kaggle creditcard.csv, matching its exact schema (Time, V1-V28,
Amount, Class), plus two enrichment columns this project adds: card_id and
merchant_category.

Why the enrichment columns exist: the real Kaggle dataset is PCA-anonymized
and has no customer/card identifier, so there is nothing to key streaming
aggregations on. We assign a synthetic card_id (and merchant_category) per
row to simulate a realistic entity key — a common, openly-acknowledged
workaround when demoing streaming features on an anonymized public dataset.
The real ingestion producer (ingestion/producer.py) does the same assignment
at replay time for the full dataset.

This sample (2,000 rows, committed to git) exists purely so the repo is
runnable end-to-end (tests, CI, local demo) without requiring anyone to sign
up for Kaggle first. For real model training, download the full dataset via
data/download_dataset.py.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_ROWS = 2000
N_CARDS = 120
N_MERCHANTS = 40
FRAUD_RATE = 0.03


def main() -> None:
    time_col = np.sort(RNG.uniform(0, 172800, N_ROWS))  # 48h of seconds, like the real dataset
    v_cols = {f"V{i}": RNG.normal(0, 1, N_ROWS) for i in range(1, 29)}
    amount = np.round(np.abs(RNG.lognormal(mean=3.0, sigma=1.2, size=N_ROWS)), 2)
    label = RNG.choice([0, 1], size=N_ROWS, p=[1 - FRAUD_RATE, FRAUD_RATE])

    card_id = RNG.integers(1, N_CARDS + 1, size=N_ROWS)
    merchant_category = RNG.integers(1, N_MERCHANTS + 1, size=N_ROWS)

    df = pd.DataFrame({"Time": time_col, **v_cols, "Amount": amount, "Class": label})
    df["card_id"] = [f"card_{c:04d}" for c in card_id]
    df["merchant_category"] = [f"mcc_{m:03d}" for m in merchant_category]

    out_path = "data/sample/sample_creditcard.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path} ({df['Class'].sum()} labeled fraud)")


if __name__ == "__main__":
    main()
