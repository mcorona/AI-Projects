"""
Unit tests for the IR metrics, checked against values computed by hand.

Author: Manuel Corona
"""

import math

from src.evaluation.ir_metrics import (
    evaluate_run,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

RELEVANT = {"d1": 1, "d2": 1, "d3": 1}


def test_perfect_ranking():
    ranked = ["d1", "d2", "d3", "d9", "d8"]
    assert ndcg_at_k(ranked, RELEVANT, 10) == 1.0
    assert recall_at_k(ranked, RELEVANT, 10) == 1.0
    assert mrr_at_k(ranked, RELEVANT, 10) == 1.0
    assert precision_at_k(ranked, RELEVANT, 3) == 1.0


def test_no_relevant_retrieved():
    ranked = ["d7", "d8", "d9"]
    assert ndcg_at_k(ranked, RELEVANT, 10) == 0.0
    assert recall_at_k(ranked, RELEVANT, 10) == 0.0
    assert mrr_at_k(ranked, RELEVANT, 10) == 0.0


def test_ndcg_hand_computed():
    # One relevant document at rank 3 (index 2) -> DCG = 1/log2(4) = 0.5.
    # Ideal ranking puts all three relevant docs first:
    #   IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1 + 0.63093 + 0.5 = 2.13093
    ranked = ["x", "y", "d2", "z"]
    expected = 0.5 / (1 + 1 / math.log2(3) + 0.5)
    assert math.isclose(ndcg_at_k(ranked, RELEVANT, 10), expected, rel_tol=1e-9)


def test_recall_is_capped_by_cutoff_not_inflated():
    # Two of three relevant docs land inside the top 2.
    ranked = ["d1", "d2", "d3"]
    assert math.isclose(recall_at_k(ranked, RELEVANT, 2), 2 / 3)
    assert math.isclose(recall_at_k(ranked, RELEVANT, 3), 1.0)


def test_mrr_uses_first_hit_only():
    assert mrr_at_k(["x", "d3", "d1"], RELEVANT, 10) == 0.5


def test_unanswered_query_scores_zero_rather_than_being_dropped():
    qrels = {"q1": RELEVANT, "q2": RELEVANT}
    run = {"q1": ["d1", "d2", "d3"]}  # q2 deliberately missing
    metrics = evaluate_run(run, qrels, k_values=(10,))
    assert math.isclose(metrics["ndcg@10"], 0.5)  # (1.0 + 0.0) / 2
