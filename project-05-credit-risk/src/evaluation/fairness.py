"""
Who the errors land on.

A single global threshold applied to a well-ranked model produces different
error rates for different groups whenever those groups have different
default rates -- which they do here, by construction of the world rather
than of the model. That is not a bug to be patched away silently; it is a
result to be measured and reported, along with what it costs to remove it.

Four definitions are computed, because they disagree and the disagreement
is the finding:

  demographic parity   equal share declined
  equal opportunity    equal recall among those who really default
  equalised odds       equal recall AND equal false-positive rate
  predictive parity    a decline means the same thing in every group

Kleinberg et al. (2016) and Chouldechova (2017) proved these cannot all
hold at once when base rates differ and the model is not perfect. This
module does not resolve that; it prices it.

Author: Manuel Corona
"""

from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.metrics import roc_auc_score

from src.evaluation import decision as dec
from src.evaluation.metrics import expected_calibration_error, rates_at

# Below this many rows -- or this many actual defaults -- a per-group rate
# is noise with a decimal point, and is reported flagged rather than
# quietly averaged in.
MIN_GROUP_N = 300
MIN_GROUP_POSITIVES = 50


def group_report(y: np.ndarray, p: np.ndarray, groups: np.ndarray,
                 threshold: float, ratio: float = dec.DEFAULT_RATIO,
                 exposure: Optional[np.ndarray] = None) -> List[Dict]:
    """Every rate that a disparity argument is ever made from, per group."""
    out = []
    for g in sorted(np.unique(groups), key=str):
        m = groups == g
        yg, pg = y[m], p[m]
        n, pos = int(m.sum()), int(yg.sum())
        row = {
            **rates_at(yg, pg, threshold),
            "group": str(g), "n": n, "positives": pos,
            "base_rate": float(yg.mean()) if n else float("nan"),
            "reliable": bool(n >= MIN_GROUP_N and pos >= MIN_GROUP_POSITIVES),
            "ece": expected_calibration_error(yg, pg),
            "mean_predicted": float(pg.mean()),
            "regret": dec.regret(yg, pg, threshold, ratio,
                                 None if exposure is None else exposure[m]),
        }
        # AUC needs both classes present in the cell.
        row["roc_auc"] = (float(roc_auc_score(yg, pg))
                          if 0 < pos < n else float("nan"))
        row["regret_per_account"] = row["regret"] / n if n else float("nan")
        out.append(row)
    return out


def gaps(report: List[Dict], reliable_only: bool = True) -> Dict[str, float]:
    """
    Max-minus-min across groups, one number per fairness definition.

    Reported as gaps rather than ratios because a ratio between two small
    rates exaggerates: an FPR of 0.02 against 0.01 is a 2x disparity and a
    one-point difference, and only one of those two framings survives
    contact with a decision about 7,500 people.
    """
    rows = [r for r in report if r["reliable"]] if reliable_only else report
    if len(rows) < 2:
        return {}
    pick = lambda k: [r[k] for r in rows if not np.isnan(r[k])]  # noqa: E731
    out = {}
    for key, label in [("flag_rate", "demographic_parity_gap"),
                       ("recall", "equal_opportunity_gap"),
                       ("fpr", "false_positive_rate_gap"),
                       ("precision", "predictive_parity_gap"),
                       ("roc_auc", "auc_gap"),
                       ("base_rate", "base_rate_gap"),
                       ("regret_per_account", "regret_per_account_gap")]:
        vals = pick(key)
        out[label] = float(max(vals) - min(vals)) if vals else float("nan")
    out["n_groups_compared"] = len(rows)
    return out


def equalising_thresholds(y: np.ndarray, p: np.ndarray, groups: np.ndarray,
                          target: float, metric: str = "recall",
                          grid: Optional[Iterable[float]] = None) -> Dict[str, float]:
    """
    Per-group thresholds that bring one metric to a common target.

    This is the intervention people reach for when a disparity shows up.
    It works -- for the metric chosen -- and this module then measures what
    it does to the others and to the money, which is the part that usually
    goes unreported.
    """
    grid = np.linspace(0.001, 0.999, 999) if grid is None else np.asarray(grid)
    out = {}
    for g in np.unique(groups):
        m = groups == g
        yg, pg = y[m], p[m]
        best_t, best_err = None, float("inf")
        for t in grid:
            val = rates_at(yg, pg, float(t))[metric]
            if np.isnan(val):
                continue
            err = abs(val - target)
            if err < best_err:
                best_t, best_err = float(t), err
        out[str(g)] = best_t if best_t is not None else float("nan")
    return out


def apply_group_thresholds(p: np.ndarray, groups: np.ndarray,
                           thresholds: Dict[str, float]) -> np.ndarray:
    """Flag mask under a per-group threshold policy."""
    t = np.array([thresholds[str(g)] for g in groups], dtype=float)
    return p >= t


def cost_of_equalising(y: np.ndarray, p: np.ndarray, groups: np.ndarray,
                       global_threshold: float, metric: str = "recall",
                       ratio: float = dec.DEFAULT_RATIO,
                       exposure: Optional[np.ndarray] = None) -> Dict:
    """
    What equalising one metric costs, in the currency of the cost model.

    Reported as a delta against the single-threshold policy. A positive
    extra_regret means the fairer policy is the more expensive one, which
    is the trade a lender is actually being asked to make -- and which a
    disparity table alone never shows.
    """
    overall = rates_at(y, p, global_threshold)[metric]
    thresholds = equalising_thresholds(y, p, groups, overall, metric)
    flag = apply_group_thresholds(p, groups, thresholds)
    w = np.ones_like(y, dtype=float) if exposure is None else np.asarray(exposure, float)
    fp = flag & (y == 0)
    fn = (~flag) & (y == 1)
    equalised_regret = float(w[fp].sum() + w[fn].sum() * ratio)
    base_regret = dec.regret(y, p, global_threshold, ratio, exposure)

    before = group_report(y, p, groups, global_threshold, ratio, exposure)
    # The before/after gaps must be computed over the same set of groups,
    # or the comparison is between two different tables. Cells too small to
    # report on are excluded from both -- EDUCATION "other/unknown" here is
    # 119 accounts with 4 defaults, and equalising a rate estimated from 4
    # events drives its threshold to the floor and swamps every gap.
    reliable = {r["group"] for r in before if r["reliable"]}
    after = []
    for g in sorted(np.unique(groups), key=str):
        if str(g) not in reliable:
            continue
        m = groups == g
        after.append({"group": str(g),
                      **rates_at(y[m], p[m], thresholds[str(g)])})
    return {
        "metric_equalised": metric,
        "target": float(overall),
        "group_thresholds": thresholds,
        "global_threshold": float(global_threshold),
        "regret_single_threshold": base_regret,
        "regret_group_thresholds": equalised_regret,
        "extra_regret": equalised_regret - base_regret,
        "extra_regret_pct": ((equalised_regret - base_regret) / base_regret * 100
                             if base_regret else float("nan")),
        "gaps_before": gaps(before),
        "gaps_after": _gaps_from_rates(after),
        "groups_compared": sorted(reliable),
        "groups_excluded_as_too_small": sorted(
            {r["group"] for r in before} - reliable),
    }


def _gaps_from_rates(rows: List[Dict]) -> Dict[str, float]:
    out = {}
    for key, label in [("flag_rate", "demographic_parity_gap"),
                       ("recall", "equal_opportunity_gap"),
                       ("fpr", "false_positive_rate_gap"),
                       ("precision", "predictive_parity_gap")]:
        vals = [r[key] for r in rows if not np.isnan(r[key])]
        out[label] = float(max(vals) - min(vals)) if vals else float("nan")
    return out
