"""
Downloads the "Credit Card Transactions Fraud Detection" dataset from Kaggle
(kartik2112/fraud-detection, ~1.85M simulated-but-realistic card transactions
generated with the Sparkov transaction generator: real-shaped fields like
card number, merchant, category, city/state, job, and timestamp — not
anonymized PCA components).

Two files ship in this dataset, and we deliberately use them for two
different roles rather than pooling and re-splitting them ourselves:

    fraudTrain.csv  (~1.30M rows)  -> offline features + model training
                                       ("historical" data)
    fraudTest.csv   (~0.56M rows)  -> replayed live through the producer
                                       ("new" transactions the trained model
                                       has never seen — the same train/serve
                                       split a real fraud team would have)

Uses the `kaggle` pip package's legacy kaggle.json (username + key)
credential flow. (Kaggle's account settings page now also issues a newer
bearer-token credential, but the currently-published `kaggle` package on
PyPI doesn't recognize that format yet — the Legacy API Key, generated from
the same settings page, is what actually works with `pip install kaggle`
today.)

Run this on your own machine (needs real internet access to kaggle.com —
this won't work from a network-restricted sandbox):

    1. https://www.kaggle.com/settings/api -> "Create Legacy API Key"
       (or "Create New Token" under "Legacy API Credentials", depending on
       what the page currently calls it) -> downloads kaggle.json
    2. Place it at ~/.kaggle/kaggle.json
       (on Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json)
    3. pip install kaggle
    4. python data/download_dataset.py

Produces: data/raw/fraudTrain.csv, data/raw/fraudTest.csv
(gitignored — not committed, ~1.5GB combined)
"""
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
DATASET = "kartik2112/fraud-detection"
EXPECTED_FILES = ["fraudTrain.csv", "fraudTest.csv"]


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

    print(f"Downloading {DATASET} into {RAW_DIR} ... (~1.5GB, may take a few minutes)")
    api.dataset_download_files(DATASET, path=str(RAW_DIR), unzip=True)

    # Kaggle sometimes nests the zip contents in a subfolder; flatten if needed.
    for f in RAW_DIR.glob("**/*.csv"):
        if f.parent != RAW_DIR:
            f.rename(RAW_DIR / f.name)

    missing = [name for name in EXPECTED_FILES if not (RAW_DIR / name).exists()]
    if missing:
        print(f"WARNING: expected files not found after download: {missing}")
        print(f"Check {RAW_DIR} for the actual downloaded filenames.")
        return

    for name in EXPECTED_FILES:
        p = RAW_DIR / name
        print(f"Done: {p} ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
