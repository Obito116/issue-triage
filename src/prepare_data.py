# Download the NLBSE'23 issue dataset and cut it down to a laptop-sized slice.
# Full thing is 1.27M issues / 1.6GB so I take a random stratified sample
# (120k train / 30k test), join title+body, save as parquet. Run once.
# data: https://github.com/nlbse2023/issue-report-classification
from __future__ import annotations

import os
import tarfile
import urllib.request

import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
OUT_DIR = "data"

TRAIN_URL = "https://tickettagger.blob.core.windows.net/datasets/nlbse23-issue-classification-train.csv.tar.gz"
TEST_URL = "https://tickettagger.blob.core.windows.net/datasets/nlbse23-issue-classification-test.csv.tar.gz"

TRAIN_CSV = "nlbse23-issue-classification-train.csv"
TEST_CSV = "nlbse23-issue-classification-test.csv"

# How many issues to keep. TF-IDF + LogReg trains in seconds on these sizes.
N_TRAIN = 120_000
N_TEST = 30_000
SEED = 42
MAX_CHARS = 4_000  # truncate very long issue bodies


def _download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        return
    print(f"downloading {os.path.basename(dest)} ...")
    urllib.request.urlretrieve(url, dest)


def _extract(archive: str, member_csv: str) -> str:
    out_path = os.path.join(RAW_DIR, member_csv)
    if os.path.exists(out_path):
        return out_path
    print(f"extracting {os.path.basename(archive)} ...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(RAW_DIR)
    return out_path


def _stratified_subsample(csv_path: str, n_target: int, seed: int) -> pd.DataFrame:
    """Random subsample taken across the whole file via chunked reservoir-style
    Bernoulli sampling, then trimmed to n_target. Robust to any row ordering."""
    rng = np.random.RandomState(seed)
    # Estimate a fraction that slightly overshoots the target.
    frac = min(1.0, (n_target * 1.4) / _estimate_rows(csv_path))
    parts = []
    for chunk in pd.read_csv(
        csv_path,
        usecols=["labels", "title", "body"],
        chunksize=100_000,
        dtype=str,
        keep_default_na=False,
        engine="c",
    ):
        keep = rng.rand(len(chunk)) < frac
        parts.append(chunk[keep])
        if sum(len(p) for p in parts) >= n_target * 1.2:
            break
    df = pd.concat(parts, ignore_index=True)
    df = df.sample(min(n_target, len(df)), random_state=seed).reset_index(drop=True)
    df["text"] = (
        df["title"].fillna("") + " " + df["body"].fillna("")
    ).str.slice(0, MAX_CHARS).str.strip()
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df[["labels", "text"]]


def _estimate_rows(csv_path: str) -> int:
    # Cheap upper-bound estimate by counting newlines (multi-line bodies inflate this,
    # which only makes the sampling fraction conservative -- fine for our purposes).
    size = os.path.getsize(csv_path)
    # ~ 1.3 KB average per issue record across title+body in this corpus.
    return max(1, int(size / 1300))


def main() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    train_archive = os.path.join(RAW_DIR, "train.csv.tar.gz")
    test_archive = os.path.join(RAW_DIR, "test.csv.tar.gz")
    _download(TRAIN_URL, train_archive)
    _download(TEST_URL, test_archive)

    train_csv = _extract(train_archive, TRAIN_CSV)
    test_csv = _extract(test_archive, TEST_CSV)

    print("subsampling train ...")
    train = _stratified_subsample(train_csv, N_TRAIN, SEED)
    print("subsampling test ...")
    test = _stratified_subsample(test_csv, N_TEST, SEED + 1)

    train.to_parquet(os.path.join(OUT_DIR, "train.parquet"), index=False)
    test.to_parquet(os.path.join(OUT_DIR, "test.parquet"), index=False)

    print(f"\nsaved data/train.parquet ({len(train):,} rows)")
    print(f"saved data/test.parquet  ({len(test):,} rows)")
    print("\nclass distribution (train):")
    print(train["labels"].value_counts().to_string())


if __name__ == "__main__":
    main()
