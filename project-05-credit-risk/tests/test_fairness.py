"""
The audit makes claims about who bears the errors, and a claim like that is
worth only as much as the arithmetic under it.
"""

import numpy as np
import pytest

from src.evaluation import fairness as fair
from src.evaluation import significance as sig


@pytest.fixture
def two_groups():
    """Different base rates, same ranking quality -- the situation that
    makes the fairness definitions disagree in the first place."""
    rng = np.random.default_rng(7)
    n = 6000
    g = rng.choice(["A", "B"], n)
    rate = np.where(g == "A", 0.30, 0.15)
    y = rng.binomial(1, rate)
    p = np.clip(rate + 0.35 * (y - rate) + rng.normal(0, 0.10, n), 0.001, 0.999)
    return y, p, g


def test_identical_groups_show_no_gap():
    rng = np.random.default_rng(3)
    n = 4000
    y = rng.binomial(1, 0.22, n)
    p = np.clip(0.22 + 0.3 * (y - 0.22) + rng.normal(0, 0.12, n), 0.001, 0.999)
    g = rng.choice(["A", "B"], n)          # group assigned independently of y and p
    gaps = fair.gaps(fair.group_report(y, p, g, 0.15))
    assert gaps["demographic_parity_gap"] < 0.05
    assert gaps["equal_opportunity_gap"] < 0.10


def test_unequal_base_rates_produce_a_decline_gap(two_groups):
    y, p, g = two_groups
    gaps = fair.gaps(fair.group_report(y, p, g, 0.20))
    assert gaps["base_rate_gap"] > 0.10
    assert gaps["demographic_parity_gap"] > 0.10


def test_small_cells_are_flagged_not_silently_averaged():
    y = np.concatenate([np.zeros(1000, int), np.ones(200, int), np.array([1, 0, 0])])
    p = np.concatenate([np.full(1000, 0.1), np.full(200, 0.6), np.array([0.6, 0.1, 0.1])])
    g = np.array(["big"] * 1200 + ["tiny"] * 3)
    report = fair.group_report(y, p, g, 0.3)
    flags = {r["group"]: r["reliable"] for r in report}
    assert flags["big"] is True
    assert flags["tiny"] is False
    assert fair.gaps(report) == {} or fair.gaps(report)["n_groups_compared"] == 1


def test_equalising_thresholds_actually_equalise(two_groups):
    y, p, g = two_groups
    target = 0.80
    th = fair.equalising_thresholds(y, p, g, target, "recall")
    for group, t in th.items():
        m = g == group
        recall = ((p[m] >= t) & (y[m] == 1)).sum() / (y[m] == 1).sum()
        assert abs(recall - target) < 0.02


def test_equalising_shrinks_the_metric_it_targets(two_groups):
    y, p, g = two_groups
    out = fair.cost_of_equalising(y, p, g, 0.20, "recall")
    assert out["gaps_after"]["equal_opportunity_gap"] < out["gaps_before"]["equal_opportunity_gap"]


def test_before_and_after_gaps_cover_the_same_groups(two_groups):
    """
    A before/after comparison computed over different group sets is not a
    comparison. This guards the fix for exactly that defect.
    """
    y, p, g = two_groups
    small = np.array(["C"] * 20)
    y2 = np.concatenate([y, np.zeros(20, int)])
    p2 = np.concatenate([p, np.full(20, 0.05)])
    g2 = np.concatenate([g, small])
    out = fair.cost_of_equalising(y2, p2, g2, 0.20, "recall")
    assert out["groups_compared"] == ["A", "B"]
    assert out["groups_excluded_as_too_small"] == ["C"]


def test_extra_regret_is_the_difference_it_claims_to_be(two_groups):
    """
    Arithmetic consistency, and a deliberate non-assertion about the sign.

    Per-group thresholds are a strictly larger policy class than a single
    threshold, so an equalised policy can be *cheaper* than the
    cost-optimal single threshold -- which is what happens on this fixture.
    That is not an accounting error: where a model is miscalibrated
    differently across groups, per-group cutoffs correct some of that on
    the way to equalising a rate. Asserting extra_regret >= 0 would be
    encoding a folk belief that fairness always costs money.
    """
    y, p, g = two_groups
    from src.evaluation.decision import best_threshold
    t = best_threshold(y, p)["threshold"]
    out = fair.cost_of_equalising(y, p, g, t, "recall")
    assert out["extra_regret"] == pytest.approx(
        out["regret_group_thresholds"] - out["regret_single_threshold"])
    assert out["regret_single_threshold"] == pytest.approx(
        __import__("src.evaluation.decision", fromlist=["regret"]).regret(y, p, t))


def test_group_thresholds_apply_per_row(two_groups):
    y, p, g = two_groups
    th = {"A": 0.9, "B": 0.0}
    flag = fair.apply_group_thresholds(p, g, th)
    assert flag[g == "B"].all()
    assert flag[g == "A"].mean() < 0.2


def test_identical_models_are_not_significantly_different():
    rng = np.random.default_rng(11)
    y = rng.binomial(1, 0.22, 3000)
    p = np.clip(0.22 + 0.3 * (y - 0.22) + rng.normal(0, 0.12, 3000), 0.001, 0.999)
    auc = sig.auc_difference(y, p, p)
    assert auc["point_estimate"] == pytest.approx(0.0)
    assert auc["ci_low"] <= 0.0 <= auc["ci_high"]
    mc = sig.mcnemar(y, p, 0.2, p, 0.2)
    assert mc["discordant"] == 0 and mc["p_value"] == 1.0


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(13)
    y = rng.binomial(1, 0.22, 3000)
    good = np.clip(0.22 + 0.4 * (y - 0.22) + rng.normal(0, 0.10, 3000), 0.001, 0.999)
    weak = np.clip(0.22 + 0.1 * (y - 0.22) + rng.normal(0, 0.20, 3000), 0.001, 0.999)
    out = sig.auc_difference(y, good, weak)
    assert out["point_estimate"] > 0
    assert out["ci_low"] > 0                       # a real difference
    assert out["ci_low"] <= out["point_estimate"] <= out["ci_high"]
