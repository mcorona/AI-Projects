"""
Cross-encoder reranking of a first-stage candidate list.

A bi-encoder has to compress a document into one vector before it ever sees
the query. A cross-encoder reads the (query, document) pair jointly, which
is strictly more informative and strictly more expensive -- O(candidates)
forward passes per query instead of one. That is why it runs as a second
stage over a shortlist rather than over the corpus.

The interesting question this module exists to answer is not "does
reranking help" (it usually does) but "does it help enough to justify
adding a second model to the serving path", which the evaluation reports.

Author: Manuel Corona
"""

from typing import Dict, List, Tuple

from sentence_transformers import CrossEncoder
from tqdm import tqdm

from src.retrieval.dense import pick_device


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 device: str = None):
        self.model_name = model_name
        self.device = device or pick_device()
        self.model = CrossEncoder(model_name, device=self.device, max_length=512)

    def rerank(self, query: str, candidates: List[Tuple[str, float]],
               doc_texts: Dict[str, str], top_k: int = 100,
               batch_size: int = 64) -> List[Tuple[str, float]]:
        """Rescore one query's candidate list; returns it re-sorted, best first."""
        if not candidates:
            return []
        doc_ids = [d for d, _ in candidates]
        pairs = [(query, doc_texts[d]) for d in doc_ids]
        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        ranked = sorted(zip(doc_ids, (float(s) for s in scores)), key=lambda kv: -kv[1])
        return ranked[:top_k]

    def rerank_run(self, queries: Dict[str, str],
                   run: Dict[str, List[Tuple[str, float]]],
                   doc_texts: Dict[str, str],
                   candidate_depth: int = 50,
                   batch_size: int = 64) -> Dict[str, List[Tuple[str, float]]]:
        """
        Rerank the top `candidate_depth` of every query in a run.

        Documents below the cutoff keep their first-stage order and are
        appended after the reranked block, so recall@100 can never drop
        below the first stage's -- only the ordering of the head changes.
        """
        out = {}
        for qid, query in tqdm(queries.items(), desc=f"reranking top-{candidate_depth}"):
            cands = run.get(qid, [])
            head = self.rerank(query, cands[:candidate_depth], doc_texts,
                               top_k=candidate_depth, batch_size=batch_size)
            head_ids = {d for d, _ in head}
            tail = [(d, s) for d, s in cands[candidate_depth:] if d not in head_ids]
            out[qid] = head + tail
        return out
