"""
Data loading and splitting utilities for the Financial PhraseBank dataset.

Author: Manuel Corona
"""

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# Must match FinancialSentimentClassifier.class_labels order
LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def load_raw_dataset(path: str) -> pd.DataFrame:
    """
    Load the raw Financial PhraseBank CSV (label,text pairs, CR-terminated,
    latin-1 encoded, no header).

    Args:
        path: Path to the raw CSV file.

    Returns:
        DataFrame with columns: label (str), text (str), label_id (int).
    """
    df = pd.read_csv(path, encoding="latin-1", lineterminator="\r", header=None, names=["label", "text"])
    df["label"] = df["label"].str.strip()
    df["text"] = df["text"].str.strip()
    df["label_id"] = df["label"].map(LABEL_TO_ID)
    return df


def create_splits(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val/test splits.

    Args:
        df: Full dataset with a 'label' column to stratify on.
        train_size: Fraction of data for training.
        val_size: Fraction of data for validation (test gets the remainder).
        random_state: Seed for reproducibility.

    Returns:
        (train_df, val_df, test_df)
    """
    test_size = 1.0 - train_size - val_size

    train_df, remainder_df = train_test_split(
        df, train_size=train_size, stratify=df["label"], random_state=random_state
    )
    relative_val_size = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        remainder_df, train_size=relative_val_size, stratify=remainder_df["label"], random_state=random_state
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


class CustomerFeedbackDataset:
    """Lightweight text/label container, indexable like a sequence."""

    def __init__(self, texts: List[str], labels: List[int]):
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        return {"text": self.texts[idx], "label": self.labels[idx]}


if __name__ == "__main__":
    raw_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "financial_phrasebank" / "all-data.csv"
    df = load_raw_dataset(str(raw_path))

    train_df, val_df, test_df = create_splits(df)

    out_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(out_dir / "train.parquet", index=False)
    val_df.to_parquet(out_dir / "val.parquet", index=False)
    test_df.to_parquet(out_dir / "test.parquet", index=False)

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    print("Train class balance:\n", train_df["label"].value_counts(normalize=True))
