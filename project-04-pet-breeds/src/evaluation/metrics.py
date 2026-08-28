"""
Classification metrics, plus the two breakdowns this project turns on.

Author: Manuel Corona
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import f1_score


def basic_metrics(preds: np.ndarray, labels: np.ndarray,
                  probs: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Accuracy and macro-F1, plus top-5 when probabilities are available.

    Macro-F1 alongside accuracy because the two diverge exactly when a
    model is good on common classes and bad on rare ones. The Pet test
    split is near-balanced (88-100 images per class) so they should track
    closely here -- a large gap would be a signal something is wrong.
    """
    out = {
        "accuracy": float((preds == labels).mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }
    if probs is not None:
        top5 = np.argsort(-probs, axis=1)[:, :5]
        out["top5_accuracy"] = float((top5 == labels[:, None]).any(axis=1).mean())
    return out


def per_class_accuracy(preds: np.ndarray, labels: np.ndarray,
                       n_classes: int) -> np.ndarray:
    """Accuracy within each true class. NaN for classes absent from labels."""
    out = np.full(n_classes, np.nan)
    for c in range(n_classes):
        mask = labels == c
        if mask.any():
            out[c] = (preds[mask] == c).mean()
    return out


def group_accuracy(preds: np.ndarray, labels: np.ndarray,
                   class_mask: np.ndarray) -> Dict[str, float]:
    """
    Accuracy over the images whose TRUE class is in a group.

    Grouping on the true label rather than the prediction: the question is
    "how well is this model doing on these breeds", and conditioning on the
    prediction would let a model look good on a group simply by never
    predicting into it.
    """
    sel = class_mask[labels]
    return {
        "n": int(sel.sum()),
        "accuracy": float((preds[sel] == labels[sel]).mean()) if sel.any() else float("nan"),
    }


def top_confusions(preds: np.ndarray, labels: np.ndarray,
                   class_names: List[str], k: int = 10):
    """The k most frequent (true -> predicted) mistakes."""
    pairs: Dict[tuple, int] = {}
    for t, p in zip(labels, preds):
        if t != p:
            pairs[(int(t), int(p))] = pairs.get((int(t), int(p)), 0) + 1
    ranked = sorted(pairs.items(), key=lambda kv: -kv[1])[:k]
    return [{"true": class_names[t], "predicted": class_names[p], "count": n}
            for (t, p), n in ranked]


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 15) -> Dict[str, float]:
    """
    ECE: mean gap between confidence and accuracy, binned by confidence.

    A model that is 95% confident should be right 95% of the time. Accuracy
    alone cannot tell you whether the model knows when it is wrong, and a
    classifier that is used to trigger an action needs that.
    """
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0, 1, n_bins + 1)
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        acc, avg_conf = correct[m].mean(), conf[m].mean()
        ece += m.mean() * abs(acc - avg_conf)
        bins.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                     "accuracy": float(acc), "confidence": float(avg_conf)})
    return {"ece": float(ece), "mean_confidence": float(conf.mean()),
            "accuracy": float(correct.mean()), "bins": bins}
