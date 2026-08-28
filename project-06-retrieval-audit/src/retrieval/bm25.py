"""
BM25 lexical retrieval -- the baseline every dense retriever has to beat.

BM25 is not a strawman. On BEIR it still outperforms many neural retrievers
out of the box, and it needs no GPU, no training, and no embedding storage.
Any RAG system that cannot beat it is not earning the complexity it costs.

Author: Manuel Corona
"""

import re
from typing import Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# Minimal English stoplist. Kept short and explicit rather than pulling in
# NLTK: the point is a reproducible baseline, not a tuned one.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "how", "i", "if", "in", "is", "it", "its", "of", "on",
    "or", "that", "the", "to", "was", "were", "what", "when", "where",
    "which", "who", "why", "will", "with", "you", "your",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords and 1-char tokens."""
    return [t for t in _TOKEN_RE.findall(text.lower())
            if t not in STOPWORDS and len(t) > 1]


class BM25Retriever:
    """BM25-Okapi over a fixed corpus, with Robertson/Sparck-Jones defaults."""

    def __init__(self, doc_ids: List[str], doc_texts: List[str], k1: float = 0.9, b: float = 0.4):
        # k1=0.9 / b=0.4 are the BEIR/Anserini defaults, not rank_bm25's
        # own (1.5 / 0.75). Using BEIR's keeps this comparable to published
        # BM25 numbers on the same benchmark.
        self.doc_ids = doc_ids
        tokenized = [tokenize(t) for t in tqdm(doc_texts, desc="tokenizing corpus")]
        self.bm25 = BM25Okapi(tokenized, k1=k1, b=b)

    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Return the top_k (doc_id, score) pairs for one query, best first."""
        scores = self.bm25.get_scores(tokenize(query))
        top_k = min(top_k, len(scores))
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.doc_ids[i], float(scores[i])) for i in idx]

    def search_batch(self, queries: Dict[str, str], top_k: int = 100) -> Dict[str, List[Tuple[str, float]]]:
        return {qid: self.search(q, top_k)
                for qid, q in tqdm(queries.items(), desc="BM25 search")}
