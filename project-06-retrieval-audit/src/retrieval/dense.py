"""
Dense (bi-encoder) retrieval with sentence-transformers.

Embeddings are cached to data/processed/ keyed by model name, because
encoding the 57,638-document FiQA corpus is the slowest step in the
pipeline and nothing about it changes between evaluation runs.

Similarity search is a plain normalized matmul rather than FAISS: the 57k
corpus is 84 MB at 384 dimensions and 169 MB at 768, and an exhaustive
search over either takes milliseconds. An approximate index would add a
dependency and a recall/speed tradeoff to solve a problem this corpus does
not have; that call would flip at roughly 10M documents.

Author: Manuel Corona
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def _cache_path(model_name: str, dataset: str = "") -> Path:
    """
    One cache file per (dataset, model).

    Project 3 indexed a single corpus, so the filename only had to identify
    the model. This audit runs the same two models over seven corpora, and
    keeping that filename would have each dataset silently overwrite the
    previous one's embeddings -- producing a matrix of the right shape only
    by coincidence, and wrong numbers with no error.
    """
    stem = f"corpus_emb__{model_name.replace('/', '__')}"
    if dataset:
        stem = f"{dataset}__{stem}"
    return PROCESSED_DIR / f"{stem}.npy"


def pick_device() -> str:
    """Prefer Apple Silicon MPS, then CUDA, then CPU."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class DenseRetriever:
    """
    Bi-encoder retrieval over a corpus embedded once and cached on disk.

    Args:
        model_name: any sentence-transformers model id.
        query_prefix / doc_prefix: instruction prefixes some models require
            (e.g. BGE wants "Represent this sentence for searching relevant
            passages: " on the query side only). Getting these wrong
            silently costs several nDCG points, so they are explicit.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 query_prefix: str = "", doc_prefix: str = "", device: str = None,
                 dataset: str = ""):
        self.model_name = model_name
        self.dataset = dataset
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.device = device or pick_device()
        self.model = SentenceTransformer(model_name, device=self.device)
        self.doc_ids: List[str] = []
        self.embeddings: np.ndarray = None

    def index(self, doc_ids: List[str], doc_texts: List[str],
              batch_size: int = 64, use_cache: bool = True) -> "DenseRetriever":
        """Embed and L2-normalize the corpus (or load a cached matrix)."""
        self.doc_ids = doc_ids
        cache = _cache_path(self.model_name, self.dataset)
        if use_cache and cache.exists():
            emb = np.load(cache)
            if emb.shape[0] == len(doc_ids):
                self.embeddings = emb
                return self
            print(f"Cache at {cache.name} has {emb.shape[0]} rows but corpus has "
                  f"{len(doc_ids)} -- re-encoding.")

        texts = [self.doc_prefix + t for t in doc_texts]
        emb = self.model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=True,
        ).astype(np.float32)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache, emb)
        self.embeddings = emb
        return self

    def search_batch(self, queries: Dict[str, str], top_k: int = 100,
                     batch_size: int = 128) -> Dict[str, List[Tuple[str, float]]]:
        """Rank the corpus for every query by cosine similarity."""
        qids = list(queries)
        q_emb = self.model.encode(
            [self.query_prefix + queries[q] for q in qids],
            batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=True,
        ).astype(np.float32)

        results = {}
        # Chunked so the (n_queries x n_docs) score matrix never has to fit
        # in memory all at once.
        for start in range(0, len(qids), 64):
            chunk = q_emb[start:start + 64]
            sims = chunk @ self.embeddings.T
            k = min(top_k, sims.shape[1])
            idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            for row, qid in enumerate(qids[start:start + 64]):
                cols = idx[row][np.argsort(-sims[row, idx[row]])]
                results[qid] = [(self.doc_ids[c], float(sims[row, c])) for c in cols]
        return results

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode([self.query_prefix + query], convert_to_numpy=True,
                                 normalize_embeddings=True).astype(np.float32)[0]

    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        sims = self.encode_query(query) @ self.embeddings.T
        k = min(top_k, sims.shape[0])
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.doc_ids[i], float(sims[i])) for i in idx]
