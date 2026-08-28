"""
The cost model is the load-bearing argument of this project, so its
arithmetic is checked against hand-computed cases rather than trusted.
"""

import numpy as np
import pytest

from src.evaluation import decision as dec
from src.evaluation.metrics import confusion_at, expected_calibration_error


def test_symmetric_costs_give_a_half_threshold():
    """A 0.5 cutoff is exactly the claim that both errors cost the same."""
    assert dec.optimal_threshold(1.0) == pytest.approx(0.5)


def test_threshold_falls_as_defaults_get_more_expensive():
    ts = [dec.optimal_threshold(r) for r in (1, 2, 5, 10, 20)]
    assert ts == sorted(ts, reverse=True)
    assert dec.optimal_threshold(7.5) == pytest.approx(0.1176, abs=1e-4)


def test_regret_matches_the_confusion_matrix():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.8, 0.2, 0.4, 0.95])
    t, r = 0.5, 4.0
    c = confusion_at(y, p, t)
    assert dec.regret(y, p, t, r) == pytest.approx(c["fp"] * 1.0 + c["fn"] * r)


def test_perfect_scores_cost_nothing():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert dec.regret(y, p, 0.5, 7.5) == 0.0


def test_exposure_weighting_prices_big_balances_higher():
    """One missed default on a large balance must cost more than a small one."""
    y = np.array([1, 1])
    p = np.array([0.0, 0.0])                     # both wrongly approved
    small = dec.regret(y, p, 0.5, 5.0, exposure=np.array([100.0, 100.0]))
    large = dec.regret(y, p, 0.5, 5.0, exposure=np.array([100.0, 900.0]))
    assert large > small
    assert large == pytest.approx(5.0 * 1000.0)


def test_naive_policies_bracket_the_oracle():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.22, 2000)
    p = np.clip(0.22 + 0.4 * (y - 0.22) + rng.normal(0, 0.12, 2000), 0.001, 0.999)
    v = dec.value_of_thresholding(y, p, 0.118)
    assert v["regret_oracle"] == 0.0
    assert v["regret_policy"] < min(v["regret_approve_all"], v["regret_decline_all"])
    assert 0.0 < v["fraction_of_oracle_gap_closed"] < 1.0


def test_approve_all_regret_is_all_the_defaults():
    y = np.array([0, 1, 1, 0])
    p = np.zeros(4)
    v = dec.value_of_thresholding(y, p, 0.5, ratio=7.5)
    assert v["regret_approve_all"] == pytest.approx(2 * 7.5)
    assert v["regret_decline_all"] == pytest.approx(2 * 1.0)


def test_best_threshold_never_beaten_by_the_grid():
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.22, 1500)
    p = np.clip(0.22 + 0.35 * (y - 0.22) + rng.normal(0, 0.15, 1500), 0.001, 0.999)
    best = dec.best_threshold(y, p, 7.5)
    for t in np.linspace(0.01, 0.99, 99):
        assert dec.regret(y, p, t, 7.5) >= best["regret"] - 1e-9


def test_topk_orderings_differ_when_exposure_varies():
    """
    Ranking by probability and by expected loss are different policies.
    If this ever stops being true the top-K comparison is vacuous.
    """
    rng = np.random.default_rng(2)
    n = 500
    y = rng.binomial(1, 0.22, n)
    p = rng.uniform(0.01, 0.9, n)
    exposure = rng.lognormal(9, 1.2, n)
    rows = dec.topk_review(y, p, exposure, k_fractions=(0.1,))
    a = rows[0]["by_probability"]["loss_prevented"]
    b = rows[0]["by_expected_loss"]["loss_prevented"]
    assert a != b
    assert rows[0]["total_loss_at_risk"] == pytest.approx(exposure[y == 1].sum())


def test_ece_is_zero_for_a_perfectly_calibrated_constant():
    y = np.concatenate([np.ones(220), np.zeros(780)])
    p = np.full(1000, 0.22)
    assert expected_calibration_error(y, p) == pytest.approx(0.0, abs=1e-3)
