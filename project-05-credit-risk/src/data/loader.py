"""
UCI "default of credit card clients": download, clean, and split.

30,000 Taiwanese credit-card accounts observed April-September 2005, with a
binary outcome for October 2005. Public, no account needed, ~5 MB.

The split is stratified and seeded, and validation is carved out of the
training pool so that model selection never touches test. Test is read once,
by src/evaluation/run_eval.py, after everything else is frozen -- the same
discipline as the previous four projects.

Author: Manuel Corona
"""

import io
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.data.schema import (
    AGE_BANDS, EDUCATION_MAP, ID_COL, MARRIAGE_MAP, TARGET, age_band,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE = DATA_DIR / "cache" / "clients.csv"

URL = ("https://archive.ics.uci.edu/static/public/350/"
       "default+of+credit+card+clients.zip")
XLS_NAME = "default of credit card clients.xls"

# 25% held out for the single test pass, then 20% of the remainder for
# validation -- 18,000 / 4,500 / 7,500. The test split is large on purpose:
# the subgroup audit slices it by sex, age band and education, and the
# smallest of those cells still needs enough defaults for a bootstrap
# interval to mean anything.
TEST_FRACTION = 0.25
VAL_FRACTION = 0.20
SPLIT_SEED = 20260827


def _download() -> Path:
    xls = RAW_DIR / XLS_NAME
    if xls.exists():
        return xls
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    import urllib.request
    with urllib.request.urlopen(URL, timeout=120) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extract(XLS_NAME, RAW_DIR)
    return xls


def load_raw() -> pd.DataFrame:
    """The sheet as published, with the two-row header collapsed."""
    return pd.read_excel(_download(), header=1)


def load_clean(use_cache: bool = True) -> pd.DataFrame:
    """
    The modelling frame: undocumented category codes folded, age banded.

    No rows are dropped and no values are imputed -- the dataset has no
    missing values, and the only defects are administrative coding gaps
    handled in schema.py.
    """
    if use_cache and CACHE.exists():
        return pd.read_csv(CACHE)

    df = load_raw()
    if df[ID_COL].duplicated().any():
        raise ValueError("duplicate IDs in the source sheet")
    if df.isna().to_numpy().any():
        raise ValueError("unexpected missing values in the source sheet")

    df["EDUCATION"] = df["EDUCATION"].map(EDUCATION_MAP).astype(int)
    df["MARRIAGE"] = df["MARRIAGE"].map(MARRIAGE_MAP).astype(int)
    df["AGE_BAND"] = df["AGE"].map(age_band)
    df = df.rename(columns={TARGET: "default"})

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df


def split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/val/test, seeded.

    Stratified on the outcome because it is 22% positive: an unstratified
    25% test split would move the base rate around by enough to shift every
    cost-optimal threshold in this project, and thresholds are the thing
    being measured.
    """
    rng = np.random.default_rng(SPLIT_SEED)
    y = df["default"].to_numpy()
    test_idx, pool_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        cut = int(round(len(idx) * TEST_FRACTION))
        test_idx.extend(idx[:cut])
        pool_idx.extend(idx[cut:])

    pool = np.array(sorted(pool_idx))
    y_pool = y[pool]
    train_idx, val_idx = [], []
    for cls in np.unique(y_pool):
        idx = pool[y_pool == cls]
        rng.shuffle(idx)
        cut = int(round(len(idx) * VAL_FRACTION))
        val_idx.extend(idx[:cut])
        train_idx.extend(idx[cut:])

    take = lambda ix: df.iloc[np.sort(np.array(ix))].reset_index(drop=True)  # noqa: E731
    return take(train_idx), take(val_idx), take(sorted(test_idx))


def get_splits(use_cache: bool = True):
    return split(load_clean(use_cache))


def xy(df: pd.DataFrame, features) -> Tuple[pd.DataFrame, np.ndarray]:
    return df[list(features)], df["default"].to_numpy()


if __name__ == "__main__":
    from src.data.schema import FEATURES
    train, val, test = get_splits()
    total = len(train) + len(val) + len(test)
    print(f"rows      : {total:,}   features: {len(FEATURES)}")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"{name:9s}: {len(part):>6,}  default rate {part['default'].mean():.4f}")
    print()
    print("test split by protected attribute")
    for attr in ["SEX", "AGE_BAND", "EDUCATION"]:
        counts = test.groupby(attr)["default"].agg(["size", "sum", "mean"])
        for key, row in counts.iterrows():
            print(f"  {attr:9s} {str(key):>14s}  n={int(row['size']):>5,}  "
                  f"defaults={int(row['sum']):>4,}  rate={row['mean']:.4f}")
