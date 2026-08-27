"""
Information-retrieval metrics: nDCG@k, Recall@k, MRR@k, Precision@k.

Implemented here rather than pulled from pytrec_eval so the definitions
are visible and unit-testable (see tests/test_ir_metrics.py, which checks
them against hand-computed values). The conventions match trec_eval:

  - Linear gain (gain = relevance grade), not 2^rel - 1. FiQA relevance is
    binary, where the two are identical, so this choice is not load-bearing
    here -- but it is stated so the numbers stay comparable to published
    BEIR results.
  - IDCG@k is computed over the ideal ranking of *all* known relevant
    documents, truncated to k. A query with more relevant documents than k
    therefore cannot reach nDCG = 1.0 unless every one of the top k is
    relevant.
  - Unjudged documents count as non-relevant.

Author: Manuel Corona
"""

from typing import Dict, List

import numpy as np


def dcg(gains: List[float]) -> float:
    """Discounted cumulative gain of a ranked list of relevance grades."""
    return float(sum(g / np.log2(i + 2) for i, g in enumerate(gains)))


def ndcg_at_k(ranked_ids: List[str], relevant: Dict[str, int], k: int) -> float:
    """nDCG@k for one query. `relevant` maps doc_id -> relevance grade (>0)."""
    if not relevant:
        return 0.0
    gains = [relevant.get(did, 0) for did in ranked_ids[:k]]
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids: List[str], relevant: Dict[str, int], k: int) -> float:
    """Fraction of this query's relevant documents that appear in the top k."""
    if not relevant:
        return 0.0
    hits = sum(1 for did in ranked_ids[:k] if did in relevant)
    return hits / len(relevant)


def precision_at_k(ranked_ids: List[str], relevant: Dict[str, int], k: int) -> float:
    """Fraction of the top k that is relevant."""
    if k == 0:
        return 0.0
    return sum(1 for did in ranked_ids[:k] if did in relevant) / k


def mrr_at_k(ranked_ids: List[str], relevant: Dict[str, int], k: int) -> float:
    """Reciprocal rank of the first relevant document in the top k, else 0."""
    for i, did in enumerate(ranked_ids[:k]):
        if did in relevant:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_run(
    run: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    k_values=(1, 3, 5, 10, 50, 100),
) -> Dict[str, float]:
    """
    Macro-average every metric over all judged queries.

    Args:
        run: {query_id: [doc_id, ...]} ranked best-first.
        qrels: {query_id: {doc_id: relevance}}.
        k_values: cutoffs to report.

    Returns:
        Flat dict, e.g. {"ndcg@10": 0.24, "recall@10": 0.30, ...}.
        Queries in `qrels` that the run did not answer score 0 rather than
        being dropped -- a retriever that returns nothing for hard queries
        should not be rewarded for it.
    """
    scores: Dict[str, List[float]] = {}
    for qid, relevant in qrels.items():
        ranked = run.get(qid, [])
        for k in k_values:
            scores.setdefault(f"ndcg@{k}", []).append(ndcg_at_k(ranked, relevant, k))
            scores.setdefault(f"recall@{k}", []).append(recall_at_k(ranked, relevant, k))
            scores.setdefault(f"precision@{k}", []).append(precision_at_k(ranked, relevant, k))
            scores.setdefault(f"mrr@{k}", []).append(mrr_at_k(ranked, relevant, k))
    return {name: float(np.mean(vals)) for name, vals in scores.items()}


HEADLINE = ["ndcg@10", "recall@10", "recall@100", "mrr@10"]


def print_metrics(name: str, metrics: Dict[str, float], keys=HEADLINE):
    cells = "  ".join(f"{k}={metrics[k]:.4f}" for k in keys if k in metrics)
    print(f"{name:32s} {cells}")
