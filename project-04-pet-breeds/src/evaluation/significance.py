"""
Paired significance tests for classifier comparisons.

A 1-point accuracy gap on 3,669 test images is not automatically real, and
"model A scored higher" is not a finding until you know whether a different
sample of 3,669 images would have said the same thing.

Both tests here are *paired*: every model is evaluated on the same images,
so the comparison should only use the images where the two models disagree.
An unpaired proportion test throws that structure away and answers a weaker
question.

Author: Manuel Corona
"""

from math import comb
from typing import Dict

import numpy as np


def mcnemar_exact(preds_a: np.ndarray, preds_b: np.ndarray,
                  labels: np.ndarray) -> Dict[str, float]:
    """
    Exact McNemar test on which of two classifiers is right more often.

    Only the discordant items carry information -- images both got right,
    or both got wrong, say nothing about which model is better. The exact
    binomial is used rather than the chi-square approximation because it is
    correct at any discordant count, and cheap at this scale.
    """
    a_ok, b_ok = preds_a == labels, preds_b == labels
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    n = b + c
    if n == 0:
        return {"a_only": 0, "b_only": 0, "discordant": 0, "p_value": 1.0}
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return {"a_only": b, "b_only": c, "discordant": n,
            "p_value": round(min(1.0, 2 * tail), 8)}


def bootstrap_accuracy_diff(preds_a: np.ndarray, preds_b: np.ndarray,
                            labels: np.ndarray, n_resamples: int = 10_000,
                            seed: int = 42) -> Dict[str, float]:
    """
    Paired bootstrap confidence interval on the accuracy difference.

    McNemar answers "is A better than B"; this answers "by how much, and
    how precisely do we know that". Both are reported because a difference
    can be significant and still too small to act on.
    """
    rng = np.random.default_rng(seed)
    diff = (preds_a == labels).astype(float) - (preds_b == labels).astype(float)
    idx = rng.integers(0, len(diff), size=(n_resamples, len(diff)))
    boot = diff[idx].mean(axis=1)
    return {"mean_difference": float(diff.mean()),
            "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5)),
            "n": int(len(diff))}


def compare_on_group(preds_a: np.ndarray, preds_b: np.ndarray,
                     labels: np.ndarray, class_mask: np.ndarray) -> Dict:
    """
    Run both tests restricted to images whose true class is in a group.

    This is what answers the question the project is built around: is the
    fine-tune's advantage over the linear probe concentrated in the breeds
    ImageNet never knew, or spread evenly?
    """
    sel = class_mask[labels]
    return {"n": int(sel.sum()),
            "mcnemar": mcnemar_exact(preds_a[sel], preds_b[sel], labels[sel]),
            "bootstrap": bootstrap_accuracy_diff(preds_a[sel], preds_b[sel], labels[sel])}
