"""
Stage 2: does adding a cross-encoder reranker help, and on what?

The second half of the same guidance Stage 1 audited:

    "Add a cross-encoder reranker after fusion for the biggest single
     precision gain."

Also stated unconditionally. Stage 1 found that the fusion half of that
sentence holds only when the dense retriever is not already ahead of BM25.
This stage asks the same question of the reranker, and includes the
configuration the guidance actually recommends -- fusion *and* reranking --
measured against the plain strong retriever it is supposed to improve on.

One arm exists purely as a control: BM25 + reranker. A cross-encoder
applied to a weak lexical first stage is the textbook case where reranking
shines, so if that arm does not gain substantially, the reranker is
misconfigured and every other number here is void. It is the same instinct
as validating the retrieval harness against published numbers before
reporting a surprising result: prove the instrument works, then use it.

Usage:  python -m src.evaluation.run_stage2

Author: Manuel Corona
"""

import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.data.beir import BY_NAME, load
from src.evaluation.ir_metrics import evaluate_run, ndcg_at_k
from src.evaluation.run_stage1 import (BGE_QUERY_PREFIX, MODELS, REDUCED,
                                       RRF_K, TOP_K, drop_self)
from src.evaluation.significance import paired_bootstrap
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import rrf_fuse
from src.retrieval.rerank import CrossEncoderReranker

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "output" / "reports"
RUNS_DIR = ROOT / "data" / "processed" / "runs"

# Depth 50, matching Project 3, so the two projects' reranking numbers are
# comparable. Deeper reranking costs linearly more and can only change
# nDCG@10 by promoting something from below rank 50.
RERANK_DEPTH = 50

# The default every RAG tutorial reaches for, and a second one trained on
# different data. Project 3 found both degraded a strong first stage; two
# rerankers makes that harder to dismiss as one bad model.
MS_MARCO = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BGE_CE = "BAAI/bge-reranker-base"


def first_stage(name: str) -> Tuple[Dict, Dict, Dict, Dict]:
    """Rebuild (or load) the Stage 1 runs. Cached so stages are cheap."""
    corpus, queries, qrels = load(name, BY_NAME[name].split)
    cache = RUNS_DIR / f"{name}__stage1.pkl"
    if cache.exists():
        with cache.open("rb") as fh:
            return corpus, queries, qrels, pickle.load(fh)

    doc_ids = list(corpus)
    doc_texts = [corpus[d] for d in doc_ids]
    runs: Dict[str, Dict] = {}
    print("  rebuilding first-stage runs ...", flush=True)
    runs["bm25"] = drop_self(BM25Retriever(doc_ids, doc_texts)
                             .search_batch(queries, top_k=TOP_K + 1))
    for short, (model_id, prefix) in MODELS.items():
        r = DenseRetriever(model_id, query_prefix=prefix, dataset=name)
        r.index(doc_ids, doc_texts)
        runs[short] = drop_self(r.search_batch(queries, top_k=TOP_K + 1))
        del r
    for short in MODELS:
        runs[f"rrf_{short}"] = rrf_fuse([runs["bm25"], runs[short]],
                                        k=RRF_K, top_k=TOP_K)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as fh:
        pickle.dump(runs, fh)
    return corpus, queries, qrels, runs


def _per_query(run, qrels, k: int = 10) -> np.ndarray:
    return np.array([ndcg_at_k([d for d, _ in run.get(q, [])], qrels[q], k)
                     for q in sorted(qrels)])


# (arm name, first-stage run to rerank, cross-encoder)
CORE_ARMS: List[Tuple[str, str, str]] = [
    ("rerank_bm25",    "bm25",    MS_MARCO),   # the control
    ("rerank_minilm",  "minilm",  MS_MARCO),
    ("rerank_bge",     "bge",     MS_MARCO),
    ("rerank_rrf_bge", "rrf_bge", MS_MARCO),   # what the guidance recommends
]

# A second reranker on the strong first stage, to show that a degradation
# is not one bad model. bge-reranker-base is roughly nine times slower than
# ms-marco per pair, so it is opt-in: NFCorpus already carries the result
# (both rerankers hurt bge there, p=0.0356 and p<0.0001), and repeating it
# on every corpus costs more time than the extra evidence is worth.
SECOND_RERANKER: Tuple[str, str, str] = ("rerank_bge_ce", "bge", BGE_CE)

ARMS: List[Tuple[str, str, str]] = list(CORE_ARMS)


def run_dataset(name: str, arms: List[Tuple[str, str, str]] = None) -> Dict:
    meta = BY_NAME[name]
    t0 = time.time()
    corpus, queries, qrels, runs = first_stage(name)
    print(f"\n{'='*70}\n{meta.label}  —  {len(corpus):,} docs, {len(queries):,} queries\n{'='*70}")

    arms = arms if arms is not None else ARMS
    rerankers: Dict[str, CrossEncoderReranker] = {}
    for arm, parent, ce_name in arms:
        if ce_name not in rerankers:
            rerankers[ce_name] = CrossEncoderReranker(ce_name)
        t = time.time()
        runs[arm] = rerankers[ce_name].rerank_run(
            queries, runs[parent], corpus, candidate_depth=RERANK_DEPTH)
        print(f"  {arm:<16} {time.time()-t:6.1f}s")

    metrics = {a: evaluate_run({q: [d for d, _ in h] for q, h in r.items()},
                               qrels, k_values=(10, 100))
               for a, r in runs.items()}
    per_q = {a: _per_query(r, qrels) for a, r in runs.items()}

    tests = {}
    for arm, parent, _ in arms:
        tests[f"{arm}_vs_{parent}"] = paired_bootstrap(per_q[arm], per_q[parent])
    # The whole recommended stack against the plain strong retriever.
    tests["rerank_rrf_bge_vs_bge"] = paired_bootstrap(per_q["rerank_rrf_bge"], per_q["bge"])

    print()
    for arm, parent, ce_name in arms:
        t_ = tests[f"{arm}_vs_{parent}"]
        v = ("HELPS" if t_["mean_difference"] > 0 and t_["p_value"] < 0.05 else
             "HURTS" if t_["mean_difference"] < 0 and t_["p_value"] < 0.05 else "n.s.")
        print(f"    {parent:<8} -> {arm:<16} {metrics[parent]['ndcg@10']:.4f} -> "
              f"{metrics[arm]['ndcg@10']:.4f}  {t_['mean_difference']:+.4f} "
              f"p={t_['p_value']:.4f}  {v}")
    t_ = tests["rerank_rrf_bge_vs_bge"]
    print(f"    {'FULL STACK':<8}    rrf+rerank vs bge alone     "
          f"{metrics['bge']['ndcg@10']:.4f} -> {metrics['rerank_rrf_bge']['ndcg@10']:.4f}  "
          f"{t_['mean_difference']:+.4f} p={t_['p_value']:.4f}")

    return {"dataset": name, "label": meta.label,
            "arms": [a for a, _, _ in arms],
            "seconds": round(time.time() - t0, 1),
            "rerank_depth": RERANK_DEPTH,
            "metrics": metrics, "tests": tests}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=REDUCED)
    ap.add_argument("--second-reranker", action="store_true",
                    help="also run bge-reranker-base on the strong first stage "
                         "(roughly 9x slower per pair)")
    args = ap.parse_args()
    arms = list(CORE_ARMS) + ([SECOND_RERANKER] if args.second_reranker else [])

    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "stage2_reranking.json"

    results = []
    for n in args.datasets:
        results.append(run_dataset(n, arms))
        # Written after every corpus, not at the end: an interrupt now costs
        # the corpus in flight, not the whole run.
        path.write_text(json.dumps({
            "claim_under_audit": ("Add a cross-encoder reranker after fusion "
                                  "for the biggest single precision gain."),
            "rerank_depth": RERANK_DEPTH,
            "cross_encoders": {"ms_marco": MS_MARCO, "bge": BGE_CE},
            "complete": False,
            "datasets": results,
        }, indent=2))
        print(f"  [saved {len(results)}/{len(args.datasets)} corpora]", flush=True)

    path.write_text(json.dumps({
        "claim_under_audit": ("Add a cross-encoder reranker after fusion for "
                              "the biggest single precision gain."),
        "rerank_depth": RERANK_DEPTH,
        "cross_encoders": {"ms_marco": MS_MARCO, "bge": BGE_CE},
        "complete": True,
        "datasets": results,
    }, indent=2))

    print(f"\n\n{'='*94}\nSUMMARY — reranking\n{'='*94}")
    print(f"{'corpus':<11} {'first stage':<9} {'before':>8} {'after':>8} "
          f"{'delta':>9} {'p':>8}  verdict")
    for r in results:
        for arm, parent, _ in arms:
            t_ = r["tests"][f"{arm}_vs_{parent}"]
            v = ("HELPS" if t_["mean_difference"] > 0 and t_["p_value"] < 0.05 else
                 "HURTS" if t_["mean_difference"] < 0 and t_["p_value"] < 0.05 else "n.s.")
            tag = parent if arm != "rerank_bge_ce" else "bge (bge-ce)"
            print(f"{r['label']:<11} {tag:<9} {r['metrics'][parent]['ndcg@10']:>8.4f} "
                  f"{r['metrics'][arm]['ndcg@10']:>8.4f} {t_['mean_difference']:>+9.4f} "
                  f"{t_['p_value']:>8.4f}  {v}")
    print(f"\n{'='*94}\nThe recommended stack (BM25+bge fused, then reranked) "
          f"vs plain bge\n{'='*94}")
    for r in results:
        t_ = r["tests"]["rerank_rrf_bge_vs_bge"]
        v = ("BETTER" if t_["mean_difference"] > 0 and t_["p_value"] < 0.05 else
             "WORSE" if t_["mean_difference"] < 0 and t_["p_value"] < 0.05 else "n.s.")
        print(f"{r['label']:<11} {r['metrics']['bge']['ndcg@10']:.4f} -> "
              f"{r['metrics']['rerank_rrf_bge']['ndcg@10']:.4f}  "
              f"{t_['mean_difference']:+.4f}  p={t_['p_value']:.4f}   {v}")
    print(f"\nwrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
