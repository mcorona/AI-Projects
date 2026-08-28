"""
Turning a probability into a decision, and pricing the result.

A credit model does not output a score for its own sake. Someone approves
or declines, and the two ways of being wrong cost different amounts. This
module makes that explicit, because the alternative -- thresholding at 0.5
and reporting accuracy -- silently assumes the two errors cost the same,
which in lending they never do.

The economics, per unit of exposure:

    approve, no default   ->  earn m   (the margin on the balance carried)
    approve, default      ->  lose L   (the balance, times loss given default)
    decline               ->  0        (no earnings, no loss)

An oracle earns m on every good account. Any real policy gives some of that
up, and the gap is the only cost worth reporting:

    regret = FP * m + FN * L

Expected value says approve when  (1-p)*m > p*L, i.e. when

    p  <  m / (m + L)  =  1 / (1 + R),       R = L / m

which is the cost-optimal threshold. It falls out of the cost ratio alone,
does not depend on exposure, and for any lending-shaped R is nowhere near
0.5. R = 1 -- the implicit assumption behind a 0.5 cutoff -- means a
declined good customer costs exactly as much as a charged-off balance.

Author: Manuel Corona
"""

from typing import Dict, Iterable, List, Optional

import numpy as np

# Default cost ratio. A revolving card balance earns on the order of 10% a
# year in net margin, and a charged-off balance loses on the order of 75%
# after recoveries -- so R = 7.5, and the optimal threshold is 0.118.
#
# This number is an assumption, not a measurement, and it is the one input
# here that a lender would replace with their own. Every result in this
# project is therefore also reported as a curve over R, so that no
# conclusion rests on the specific value.
MARGIN = 0.10
LOSS_GIVEN_DEFAULT = 0.75
DEFAULT_RATIO = LOSS_GIVEN_DEFAULT / MARGIN


def optimal_threshold(ratio: float = DEFAULT_RATIO) -> float:
    """The break-even probability: approve below it, decline at or above."""
    return 1.0 / (1.0 + ratio)


def regret(y: np.ndarray, p: np.ndarray, threshold: float,
           ratio: float = DEFAULT_RATIO,
           exposure: Optional[np.ndarray] = None) -> float:
    """
    Money left on the table versus a policy that knew every outcome.

    In units of margin (m = 1), so a regret of 100 means "a hundred units
    of exposure-worth of margin was lost, one way or the other". With
    exposure supplied, each account is weighted by the balance actually at
    risk rather than counted as one.
    """
    w = np.ones_like(y, dtype=float) if exposure is None else np.asarray(exposure, dtype=float)
    flag = p >= threshold
    fp = flag & (y == 0)
    fn = (~flag) & (y == 1)
    return float(w[fp].sum() * 1.0 + w[fn].sum() * ratio)


def sweep(y: np.ndarray, p: np.ndarray, ratio: float = DEFAULT_RATIO,
          exposure: Optional[np.ndarray] = None,
          grid: Optional[Iterable[float]] = None) -> List[Dict]:
    """Regret across the whole threshold range, for the cost curve."""
    grid = np.linspace(0.001, 0.999, 999) if grid is None else np.asarray(grid)
    return [{"threshold": float(t),
             "regret": regret(y, p, t, ratio, exposure),
             "flag_rate": float(np.mean(p >= t))} for t in grid]


def best_threshold(y: np.ndarray, p: np.ndarray, ratio: float = DEFAULT_RATIO,
                   exposure: Optional[np.ndarray] = None) -> Dict:
    """
    The empirically cheapest threshold on the data given.

    Only ever called on validation. The theoretical optimum 1/(1+R) is
    correct for a perfectly calibrated model; this finds what actually
    minimises cost for the model in hand, and the distance between the two
    is a calibration diagnostic in its own right.
    """
    rows = sweep(y, p, ratio, exposure)
    best = min(rows, key=lambda r: r["regret"])
    return {"threshold": best["threshold"], "regret": best["regret"],
            "theoretical": optimal_threshold(ratio),
            "flag_rate": best["flag_rate"]}


def ratio_sensitivity(y: np.ndarray, p: np.ndarray,
                      ratios: Iterable[float] = (1, 2, 3, 5, 7.5, 10, 15, 20),
                      exposure: Optional[np.ndarray] = None) -> List[Dict]:
    """
    How the answer moves as the cost ratio moves.

    The point of reporting this is that the headline claim of this project
    -- 0.5 is the wrong threshold -- should not depend on believing R=7.5.
    It holds for every ratio a lender would recognise.
    """
    out = []
    for r in ratios:
        b = best_threshold(y, p, r, exposure)
        out.append({
            "ratio": float(r),
            "theoretical_threshold": optimal_threshold(r),
            "empirical_threshold": b["threshold"],
            "regret_at_empirical": b["regret"],
            "regret_at_half": regret(y, p, 0.5, r, exposure),
            "regret_at_base_rate": regret(y, p, float(np.mean(y)), r, exposure),
        })
    return out


def value_of_thresholding(y: np.ndarray, p: np.ndarray, threshold: float,
                          ratio: float = DEFAULT_RATIO,
                          exposure: Optional[np.ndarray] = None) -> Dict:
    """
    The chosen policy against the two policies that need no model at all.

    'Approve everyone' is what a lender does with no model; 'decline
    everyone' is the degenerate other end. A model that cannot beat both is
    not worth deploying, and reporting regret against them is the honest
    version of a lift chart.
    """
    w = np.ones_like(y, dtype=float) if exposure is None else np.asarray(exposure, dtype=float)
    approve_all = float(w[y == 1].sum() * ratio)
    decline_all = float(w[y == 0].sum() * 1.0)
    policy = regret(y, p, threshold, ratio, exposure)
    best_naive = min(approve_all, decline_all)
    return {
        "regret_policy": policy,
        "regret_approve_all": approve_all,
        "regret_decline_all": decline_all,
        "regret_oracle": 0.0,
        "saved_vs_best_naive": best_naive - policy,
        "fraction_of_oracle_gap_closed": (
            float((best_naive - policy) / best_naive) if best_naive else float("nan")),
    }


def topk_review(y: np.ndarray, p: np.ndarray, exposure: np.ndarray,
                k_fractions: Iterable[float] = (0.01, 0.05, 0.10, 0.20)) -> List[Dict]:
    """
    A capacity-constrained policy: only K accounts can be reviewed.

    Two ways to spend that capacity -- rank by probability of default, or
    by expected loss (probability times balance at risk). They are not the
    same ordering, and which one recovers more money is an empirical
    question that a ranking metric like AUC cannot answer, because AUC does
    not know that some accounts carry ten times the balance of others.
    """
    exposure = np.asarray(exposure, dtype=float)
    by_p = np.argsort(-p)
    by_ev = np.argsort(-(p * exposure))
    total_loss = float(exposure[y == 1].sum())
    out = []
    for f in k_fractions:
        k = max(1, int(round(len(y) * f)))
        row = {"k_fraction": float(f), "k": k, "total_loss_at_risk": total_loss}
        for name, order in [("by_probability", by_p), ("by_expected_loss", by_ev)]:
            sel = order[:k]
            caught = y[sel] == 1
            row[name] = {
                "defaults_caught": int(caught.sum()),
                "loss_prevented": float(exposure[sel][caught].sum()),
                "share_of_loss_prevented": (
                    float(exposure[sel][caught].sum() / total_loss) if total_loss else float("nan")),
            }
        out.append(row)
    return out
