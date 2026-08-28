"""
Whether a difference is a result or a rounding error.

Two paired tests, both over the same test rows for both models, because
comparing independent confidence intervals is a weaker test than comparing
the paired difference -- the models agree on most accounts, and the paired
version gets to use that.

Author: Manuel Corona
"""

from typing import Callable, Dict, Optional

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.evaluation import decision as dec

BOOTSTRAP_N = 2000
SEED = 20260827


def paired_bootstrap(y: np.ndarray, stat: Callable[[np.ndarray], float],
                     n: int = BOOTSTRAP_N, seed: int = SEED) -> Dict[str, float]:
    """
    Percentile interval for any statistic, resampling accounts with
    replacement. The same resampled index is used for both models, so the
    correlation between them is preserved rather than averaged away.
    """
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(y))
    vals = np.empty(n)
    for b in range(n):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        vals[b] = stat(idx)
    return {
        "mean_difference": float(vals.mean()),
        "ci_low": float(np.percentile(vals, 2.5)),
        "ci_high": float(np.percentile(vals, 97.5)),
        "n": int(len(y)),
        "bootstrap_samples": int(n),
    }


def auc_difference(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray) -> Dict:
    """AUC(a) - AUC(b), with a paired bootstrap interval."""
    def stat(idx):
        yi = y[idx]
        if yi.min() == yi.max():
            return 0.0
        return float(roc_auc_score(yi, p_a[idx]) - roc_auc_score(yi, p_b[idx]))
    out = paired_bootstrap(y, stat)
    out["point_estimate"] = float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b))
    return out


def regret_difference(y: np.ndarray, p_a: np.ndarray, t_a: float,
                      p_b: np.ndarray, t_b: float,
                      ratio: float = dec.DEFAULT_RATIO,
                      exposure: Optional[np.ndarray] = None) -> Dict:
    """
    Regret(a) - regret(b) per account, each model at its own threshold.

    Per account rather than in total so the interval is on a scale that
    does not change meaning when the test split changes size.
    """
    w = np.ones_like(y, dtype=float) if exposure is None else np.asarray(exposure, float)

    def cost(idx, p, t):
        yi, pi, wi = y[idx], p[idx], w[idx]
        flag = pi >= t
        return float(wi[flag & (yi == 0)].sum() + wi[(~flag) & (yi == 1)].sum() * ratio)

    def stat(idx):
        return (cost(idx, p_a, t_a) - cost(idx, p_b, t_b)) / len(idx)

    out = paired_bootstrap(y, stat)
    full = np.arange(len(y))
    out["point_estimate"] = stat(full)
    out["regret_a"] = cost(full, p_a, t_a)
    out["regret_b"] = cost(full, p_b, t_b)
    return out


def mcnemar(y: np.ndarray, p_a: np.ndarray, t_a: float,
            p_b: np.ndarray, t_b: float) -> Dict:
    """
    Exact McNemar on the decisions, not the scores.

    Counts only the accounts the two policies decide differently, which is
    the right denominator: agreement carries no evidence either way.
    """
    correct_a = (p_a >= t_a).astype(int) == y
    correct_b = (p_b >= t_b).astype(int) == y
    a_only = int((correct_a & ~correct_b).sum())
    b_only = int((~correct_a & correct_b).sum())
    n = a_only + b_only
    p = float(stats.binomtest(a_only, n, 0.5).pvalue) if n else 1.0
    return {"a_only": a_only, "b_only": b_only, "discordant": n, "p_value": p}
