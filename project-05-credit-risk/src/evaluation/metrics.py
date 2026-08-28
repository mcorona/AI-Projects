"""
Two families of metric, kept apart on purpose.

Discrimination (AUC, PR-AUC) asks whether the model ranks defaulters above
non-defaulters. Calibration (Brier, ECE) asks whether the number it prints
is a probability. A model can be excellent at the first and useless at the
second, and only the second is what a threshold is applied to -- which is
the whole reason this project separates them.

Author: Manuel Corona
"""

from typing import Dict, List

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def discrimination(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    """Ranking quality. Threshold-free, and therefore decision-free."""
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "base_rate": float(np.mean(y)),
    }


def expected_calibration_error(y: np.ndarray, p: np.ndarray,
                               n_bins: int = 10) -> float:
    """
    ECE with equal-width bins over [0, 1].

    Equal-width rather than equal-count because the decision thresholds in
    this project land in the low-probability region, and equal-count bins
    would pool most of that region into one bin and hide error exactly
    where it costs money.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> List[Dict]:
    """Per-bin observed-vs-predicted, for the reliability plot."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        mask = idx == b
        out.append({
            "bin_low": float(edges[b]), "bin_high": float(edges[b + 1]),
            "n": int(mask.sum()),
            "mean_predicted": float(p[mask].mean()) if mask.any() else None,
            "observed_rate": float(y[mask].mean()) if mask.any() else None,
        })
    return out


def calibration(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    return {
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p, n_bins),
        "mean_predicted": float(np.mean(p)),
        "observed_rate": float(np.mean(y)),
    }


def confusion_at(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, int]:
    """Counts under the policy 'flag as high risk when p >= threshold'."""
    flag = p >= threshold
    pos = y == 1
    return {
        "tp": int((flag & pos).sum()), "fp": int((flag & ~pos).sum()),
        "fn": int((~flag & pos).sum()), "tn": int((~flag & ~pos).sum()),
    }


def rates_at(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, float]:
    """
    The rates a credit reviewer actually argues about.

    flag_rate is the share of applicants declined; recall is the share of
    true defaults caught; precision is how often a decline was justified.
    """
    c = confusion_at(y, p, threshold)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    n = tp + fp + fn + tn
    safe = lambda a, b: float(a / b) if b else float("nan")  # noqa: E731
    return {
        **{k: float(v) for k, v in c.items()},
        "n": float(n),
        "flag_rate": safe(tp + fp, n),
        "recall": safe(tp, tp + fn),          # true positive rate
        "fpr": safe(fp, fp + tn),
        "precision": safe(tp, tp + fp),
        "accuracy": safe(tp + tn, n),
    }


def summarize(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict:
    return {
        **discrimination(y, p),
        **calibration(y, p),
        "threshold": float(threshold),
        "rates": rates_at(y, p, threshold),
    }
