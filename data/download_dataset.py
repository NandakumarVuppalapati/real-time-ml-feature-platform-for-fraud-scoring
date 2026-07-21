"""
Downloads the public "Credit Card Fraud Detection" dataset from Kaggle
(mlg-ulb/creditcardfraud, ~285k anonymized European card transactions, Sept 2013).

Run this on your own machine (not inside the sandbox) since it needs your
personal Kaggle API credentials:

    1. Create a free Kaggle account: https://www.kaggle.com
    2. Go to Account settings -> API -> "Create New Token" -> downloads kaggle.json
    3. Place it at ~/.kaggle/kaggle.json  (chmod 600 on Linux/Mac)
    4. pip install kaggle
    5. python data/download_dataset.py

Produces: data/raw/creditcard.csv  (gitignored — not committed, ~150MB)
"""
import os
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
DATASET = "mlg-ulb/creditcardfraud"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise SystemExit(
            "kaggle package not installed. Run: pip install kaggle"
        ) from exc

    api = KaggleApi()
    api.authenticate()  # reads ~/.kaggle/kaggle.json

    print(f"Downloading {DATASET} into {RAW_DIR} ...")
    api.dataset_download_files(DATASET, path=str(RAW_DIR), unzip=True)

    csv_path = RAW_DIR / "creditcard.csv"
    if csv_path.exists():
        print(f"Done: {csv_path} ({csv_path.stat().st_size / 1e6:.1f} MB)")
    else:
        # Kaggle sometimes nests the zip contents; flatten if needed.
        for f in RAW_DIR.glob("**/*.csv"):
            f.rename(RAW_DIR / f.name)
        print("Done. Check data/raw/ for creditcard.csv")


if __name__ == "__main__":
    main()
