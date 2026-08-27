"""
Data loading and chronological splitting for the RELIANCE daily price series.

Author: Manuel Corona
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "nse_stocks" / "combined_stock_data.csv"

# Known near-exact ~50% single-day price drops that are corporate-action
# (split / bonus share) artifacts in this unadjusted dataset, NOT real
# market moves -- verified by their ratio being suspiciously close to a
# clean 0.5 (a genuine crash/rally essentially never lands on an exact
# round fraction). Other large single-day moves found in the same scan
# (2004-05-17 -15%, 2006-01-18 -25%, 2008-10-24 -16%, 2009-05-18 +21%)
# correspond to well-documented real events (India general-election market
# swings in 2004/2009, the Oct 2008 financial crisis, and a likely 2006
# corporate demerger) and are deliberately left untouched -- "fixing" a
# real price move would be a worse error than leaving a known artifact.
KNOWN_SPLIT_DATES = ["2009-11-26", "2017-09-07"]


def _adjust_for_splits(df: pd.DataFrame, split_dates=KNOWN_SPLIT_DATES) -> pd.DataFrame:
    """
    Back-adjust OHLC prices for known stock-split/bonus-share dates so the
    series is continuous (same method Yahoo Finance's "Adj Close" uses for
    splits): every price before a split date is scaled down by the same
    ratio the raw price jumped by on that date, leaving the current price
    level and all post-split prices untouched.
    """
    df = df.copy()
    price_cols = ["Open", "High", "Low", "Close"]
    for split_date in split_dates:
        split_date = pd.Timestamp(split_date)
        if split_date not in df.index:
            continue
        loc = df.index.get_loc(split_date)
        if loc == 0:
            continue
        prev_close = df["Close"].iloc[loc - 1]
        curr_close = df["Close"].iloc[loc]
        ratio = curr_close / prev_close
        df.loc[df.index < split_date, price_cols] *= ratio
    return df


def load_series(symbol: str = "RELIANCE", path: Path = RAW_PATH, adjust_splits: bool = True) -> pd.DataFrame:
    """
    Load a single symbol's daily OHLCV series, indexed by date.

    Args:
        symbol: Ticker to filter to (e.g. "RELIANCE").
        path: Path to the combined multi-symbol CSV.
        adjust_splits: If True (default), back-adjust prices for the known
            split dates in KNOWN_SPLIT_DATES so the series doesn't contain
            artificial ~50% discontinuities. See module docstring on
            KNOWN_SPLIT_DATES for why only those specific dates are touched.

    Returns:
        DataFrame indexed by DatetimeIndex (daily, business-day frequency
        as traded -- NOT reindexed/filled), columns: Open, High, Low,
        Close, Volume.
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df[df["Symbol"] == symbol].drop(columns=["Symbol"]).sort_values("Date")
    df = df.set_index("Date")
    if adjust_splits:
        df = _adjust_for_splits(df)
    return df


def chronological_split(
    df: pd.DataFrame, train_size: float = 0.7, val_size: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-indexed DataFrame chronologically (no shuffling -- shuffling
    a time series leaks future information into training).

    Args:
        df: Time-indexed DataFrame, already sorted ascending by date.
        train_size: Fraction of rows for training.
        val_size: Fraction of rows for validation (test gets the remainder).

    Returns:
        (train_df, val_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


if __name__ == "__main__":
    df = load_series()
    train_df, val_df, test_df = chronological_split(df)

    out_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(out_dir / "train.parquet")
    val_df.to_parquet(out_dir / "val.parquet")
    test_df.to_parquet(out_dir / "test.parquet")

    print(f"Train: {len(train_df)} ({train_df.index.min().date()} to {train_df.index.max().date()})")
    print(f"Val:   {len(val_df)} ({val_df.index.min().date()} to {val_df.index.max().date()})")
    print(f"Test:  {len(test_df)} ({test_df.index.min().date()} to {test_df.index.max().date()})")
