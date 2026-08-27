"""
Evaluation utilities for comparing baseline sentiment classifiers.

Author: Manuel Corona
"""

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

LABELS = ["negative", "neutral", "positive"]


def evaluate_predictions(y_true: List[str], y_pred: List[str]) -> Dict:
    """
    Compute the standard metrics suite for a set of predictions.

    Args:
        y_true: Ground-truth string labels.
        y_pred: Predicted string labels.

    Returns:
        Dict with accuracy, weighted precision/recall/f1, per-class f1,
        the confusion matrix, and the full classification report.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0, labels=LABELS),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0, labels=LABELS),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=LABELS),
        "f1_per_class": dict(zip(
            LABELS, f1_score(y_true, y_pred, average=None, zero_division=0, labels=LABELS)
        )),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=LABELS, zero_division=0
        ),
    }


def comparison_table(results: Dict[str, Dict]) -> str:
    """
    Build a markdown comparison table from {model_name: metrics_dict}.
    """
    header = "| Model | Accuracy | F1 (Weighted) | Precision | Recall |\n"
    header += "|-------|----------|----------------|-----------|--------|\n"
    rows = ""
    for name, m in results.items():
        rows += (
            f"| {name} | {m['accuracy']:.3f} | {m['f1_weighted']:.3f} | "
            f"{m['precision_weighted']:.3f} | {m['recall_weighted']:.3f} |\n"
        )
    return header + rows
