"""
Preliminary subgroup / bias analysis utilities.

Phase 1 scope: text-length subgroups only (no reliable sector/domain
labels exist in the raw Financial PhraseBank data, so that dimension is
deferred -- see PHASE_1_SUMMARY.md).

Author: Manuel Corona
"""

from typing import List

import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

LENGTH_BINS = [0, 50, 100, float("inf")]
LENGTH_LABELS = ["short (<50 tok)", "medium (50-100 tok)", "long (>100 tok)"]


def add_length_bucket(df: pd.DataFrame, text_col: str = "text", tokenizer=None) -> pd.DataFrame:
    """
    Add a 'length_bucket' column. Uses word count as a fast proxy for token
    count when no tokenizer is supplied (word count and RoBERTa BPE token
    count are highly correlated for this corpus -- see EDA notebook).
    """
    df = df.copy()
    if tokenizer is not None:
        lengths = df[text_col].apply(lambda t: len(tokenizer.encode(t)))
    else:
        lengths = df[text_col].str.split().str.len()
    df["length_bucket"] = pd.cut(lengths, bins=LENGTH_BINS, labels=LENGTH_LABELS, right=False)
    return df


def subgroup_performance(
    df: pd.DataFrame, y_true_col: str, y_pred_col: str, group_col: str
) -> pd.DataFrame:
    """
    Compute accuracy, weighted F1, and support for each subgroup.

    Args:
        df: DataFrame containing true labels, predictions, and the
            grouping column.
        y_true_col: Column name with ground-truth labels.
        y_pred_col: Column name with predicted labels.
        group_col: Column name to group by (e.g. 'length_bucket').

    Returns:
        DataFrame with one row per subgroup: n, accuracy, f1_weighted.
    """
    rows = []
    for group, sub in df.groupby(group_col, observed=True):
        rows.append({
            "group": group,
            "n": len(sub),
            "accuracy": accuracy_score(sub[y_true_col], sub[y_pred_col]),
            "f1_weighted": f1_score(
                sub[y_true_col], sub[y_pred_col], average="weighted", zero_division=0
            ),
        })
    return pd.DataFrame(rows)


def class_balance_by_group(df: pd.DataFrame, label_col: str, group_col: str) -> pd.DataFrame:
    """Cross-tabulate class distribution within each subgroup (row-normalized %)."""
    return pd.crosstab(df[group_col], df[label_col], normalize="index").round(3) * 100
