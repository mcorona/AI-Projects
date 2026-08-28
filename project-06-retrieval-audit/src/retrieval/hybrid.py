"""
Hybrid retrieval by Reciprocal Rank Fusion (RRF).

RRF combines several ranked lists using only rank position, never the raw
scores. That matters here because BM25 scores and cosine similarities live
on incomparable scales -- any weighted-sum fusion needs per-corpus score
normalization that has to be re-tuned whenever either retriever changes,
which is a tuning knob masquerading as a method. RRF has one constant.

    RRF(d) = sum over systems of 1 / (k + rank_of_d_in_that_system)

Cormack et al. (2009) propose k=60 and it is the value used throughout the
retrieval literature; it is kept here rather than tuned on the test set.

Author: Manuel Corona
"""

from typing import Dict, List, Tuple


def rrf_fuse(runs: List[Dict[str, List[Tuple[str, float]]]],
             k: int = 60, top_k: int = 100) -> Dict[str, List[Tuple[str, float]]]:
    """
    Fuse several runs into one.

    Args:
        runs: list of {query_id: [(doc_id, score), ...]}, each best-first.
        k: RRF smoothing constant.
        top_k: how many documents to keep per query after fusion.

    Returns:
        {query_id: [(doc_id, rrf_score), ...]} best-first.
    """
    all_qids = {qid for run in runs for qid in run}
    fused: Dict[str, List[Tuple[str, float]]] = {}
    for qid in all_qids:
        scores: Dict[str, float] = {}
        for run in runs:
            for rank, (doc_id, _) in enumerate(run.get(qid, []), start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        fused[qid] = ranked
    return fused
