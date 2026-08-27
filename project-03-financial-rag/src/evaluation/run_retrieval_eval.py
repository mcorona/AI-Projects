"""
Run every retrieval configuration over the FiQA test split and score them.

Each stage caches its ranked run to data/processed/runs/ so the expensive
parts (corpus embedding, cross-encoder reranking) are paid once. Metrics
land in output/reports/retrieval_metrics.json.

Usage:
    python -m src.evaluation.run_retrieval_eval               # all stages
    python -m src.evaluation.run_retrieval_eval --stages bm25 dense

Author: Manuel Corona
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

from src.data.loader import doc_text, load_split
from src.evaluation.ir_metrics import evaluate_run, print_metrics
from src.retrieval.hybrid import rrf_fuse

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "data" / "processed" / "runs"
REPORTS_DIR = ROOT / "output" / "reports"

TOP_K = 100
RERANK_DEPTH = 50

# BGE asks for an instruction prefix on the query side only; omitting it
# is a silent several-point nDCG loss, so it is spelled out here.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def save_run(name: str, run: Dict[str, List[Tuple[str, float]]]):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNS_DIR / f"{name}.json", "w") as f:
        json.dump({q: [[d, s] for d, s in v] for q, v in run.items()}, f)


def load_run(name: str):
    path = RUNS_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return {q: [(d, s) for d, s in v] for q, v in json.load(f).items()}


def ids_only(run: Dict[str, List[Tuple[str, float]]]) -> Dict[str, List[str]]:
    return {q: [d for d, _ in v] for q, v in run.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="*", default=None,
                        help="subset of: random bm25 dense_minilm dense_bge hybrid "
                             "rerank_hybrid rerank_dense rerank_dense_bge rerank_bm25")
    parser.add_argument("--force", action="store_true", help="ignore cached runs")
    args = parser.parse_args()

    corpus, queries, qrels = load_split("test")
    doc_ids = list(corpus)
    doc_texts_list = [doc_text(corpus[d]) for d in doc_ids]
    doc_texts = dict(zip(doc_ids, doc_texts_list))
    print(f"{len(corpus):,} docs | {len(queries):,} test queries")

    all_stages = ["random", "bm25", "dense_minilm", "dense_bge", "hybrid",
                  "rerank_hybrid", "rerank_dense", "rerank_dense_bge", "rerank_bm25"]
    stages = args.stages or all_stages
    runs: Dict[str, Dict] = {}
    timings: Dict[str, float] = {}

    def get(name: str, build):
        """Load a cached run or build (and cache) it, recording wall time."""
        if not args.force:
            cached = load_run(name)
            if cached is not None:
                print(f"[{name}] loaded from cache")
                runs[name] = cached
                return cached
        t0 = time.time()
        run = build()
        timings[name] = time.time() - t0
        save_run(name, run)
        runs[name] = run
        print(f"[{name}] built in {timings[name]:.1f}s")
        return run

    if "random" in stages:
        def build_random():
            # The floor. With 2.6 relevant docs in a 57,638-doc corpus,
            # random retrieval should score essentially zero -- if a "real"
            # retriever lands near this, something is wired wrong.
            rng = random.Random(42)
            return {qid: [(d, 0.0) for d in rng.sample(doc_ids, TOP_K)] for qid in queries}
        get("random", build_random)

    if "bm25" in stages:
        def build_bm25():
            from src.retrieval.bm25 import BM25Retriever
            return BM25Retriever(doc_ids, doc_texts_list).search_batch(queries, TOP_K)
        get("bm25", build_bm25)

    if "dense_minilm" in stages:
        def build_minilm():
            from src.retrieval.dense import DenseRetriever
            r = DenseRetriever("sentence-transformers/all-MiniLM-L6-v2")
            r.index(doc_ids, doc_texts_list)
            return r.search_batch(queries, TOP_K)
        get("dense_minilm", build_minilm)

    if "dense_bge" in stages:
        def build_bge():
            from src.retrieval.dense import DenseRetriever
            r = DenseRetriever("BAAI/bge-base-en-v1.5", query_prefix=BGE_QUERY_PREFIX)
            r.index(doc_ids, doc_texts_list)
            return r.search_batch(queries, TOP_K)
        get("dense_bge", build_bge)

    if "hybrid" in stages:
        needed = [n for n in ("bm25", "dense_bge") if n not in runs]
        for n in needed:
            cached = load_run(n)
            if cached is None:
                raise SystemExit(f"hybrid needs the '{n}' run -- build it first")
            runs[n] = cached
        get("hybrid", lambda: rrf_fuse([runs["bm25"], runs["dense_bge"]], top_k=TOP_K))

    # Rerank both first stages, not just the fused one. Reranking only the
    # hybrid would confound two changes at once (fusion AND reranking) and
    # make it impossible to say which one moved the number.
    # (stage, first stage to rerank, cross-encoder). Two rerankers, not one:
    # ms-marco-MiniLM is the default every RAG tutorial reaches for and was
    # trained on web search, while bge-reranker-base was trained on more
    # varied retrieval data. Running only the first would leave "reranking
    # hurt" indistinguishable from "that particular reranker didn't transfer".
    MS_MARCO_CE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    BGE_CE = "BAAI/bge-reranker-base"
    rerank_stages = (
        ("rerank_hybrid", "hybrid", MS_MARCO_CE),
        ("rerank_dense", "dense_bge", MS_MARCO_CE),
        ("rerank_dense_bge", "dense_bge", BGE_CE),
        # Diagnostic: rerank a deliberately weak first stage. If the
        # cross-encoder is wired correctly it should lift BM25 substantially
        # -- that is the case reranking is famous for. A reranker that helps
        # BM25 and hurts the dense retriever is not broken, it simply has a
        # quality ceiling below the dense retriever on this domain.
        ("rerank_bm25", "bm25", MS_MARCO_CE),
    )
    for stage, first_stage, ce_model in rerank_stages:
        if stage not in stages:
            continue
        if first_stage not in runs:
            runs[first_stage] = load_run(first_stage)
            if runs[first_stage] is None:
                raise SystemExit(f"{stage} needs the '{first_stage}' run -- build it first")

        def build_rerank(_first=first_stage, _ce=ce_model):
            from src.retrieval.rerank import CrossEncoderReranker
            return CrossEncoderReranker(_ce).rerank_run(
                queries, runs[_first], doc_texts, candidate_depth=RERANK_DEPTH)
        get(stage, build_rerank)

    print("\n=== FiQA-2018 test split (648 queries, 57,638 docs) ===")
    report = {}
    for name in all_stages:
        run = runs.get(name) or load_run(name)
        if run is None:
            continue
        metrics = evaluate_run(ids_only(run), qrels)
        if name in timings:
            metrics["build_seconds"] = round(timings[name], 1)
        report[name] = metrics
        print_metrics(name, metrics)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "retrieval_metrics.json"
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update(report)
    out.write_text(json.dumps(existing, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
