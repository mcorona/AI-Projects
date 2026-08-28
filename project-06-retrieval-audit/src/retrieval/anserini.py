"""
BM25 from Anserini, the implementation the BEIR paper's numbers come from.

The audit's headline result is a relationship between a dense retriever's
advantage over BM25 and what fusion does to it. BM25 sits on both axes, so
the audit's own BM25 is load-bearing -- and `rank_bm25` is not the
implementation the published figures come from. It ran 6.4 points below the
reference on TREC-COVID and 9.8 points *above* it on ArguAna, both in ways
that could move the conclusion.

This module runs the reference implementation instead: Anserini's prebuilt
BEIR Lucene indexes via pyserini, default parameters k1=0.9 and b=0.4 --
the same values `rank_bm25` was given, so the comparison isolates
implementation and tokenization.

Requires a JVM. Kept in a separate module and a separate virtualenv because
pyserini drags in a JDK dependency and half a gigabyte of wheels that the
rest of this project has no use for.

Author: Manuel Corona
"""

from typing import Dict, List, Tuple

# BEIR's prebuilt Anserini indexes, one per corpus in this audit.
PREBUILT = {
    "scifact":    "beir-v1.0.0-scifact.flat",
    "nfcorpus":   "beir-v1.0.0-nfcorpus.flat",
    "arguana":    "beir-v1.0.0-arguana.flat",
    "scidocs":    "beir-v1.0.0-scidocs.flat",
    "trec-covid": "beir-v1.0.0-trec-covid.flat",
    "fiqa":       "beir-v1.0.0-fiqa.flat",
}

K1, B = 0.9, 0.4


class AnseriniBM25:
    def __init__(self, dataset: str, k1: float = K1, b: float = B):
        from pyserini.search.lucene import LuceneSearcher
        self.dataset = dataset
        self.searcher = LuceneSearcher.from_prebuilt_index(PREBUILT[dataset])
        self.searcher.set_bm25(k1, b)
        self.num_docs = self.searcher.num_docs

    def search_batch(self, queries: Dict[str, str],
                     top_k: int = 101) -> Dict[str, List[Tuple[str, float]]]:
        """
        One ranking per query.

        Queries are passed verbatim. Anserini applies its own analyzer --
        which is the entire point of running this: the tokenization
        difference against rank_bm25 is what the replication measures.
        """
        out: Dict[str, List[Tuple[str, float]]] = {}
        for qid, text in queries.items():
            hits = self.searcher.search(text, k=top_k)
            out[qid] = [(h.docid, float(h.score)) for h in hits]
        return out
