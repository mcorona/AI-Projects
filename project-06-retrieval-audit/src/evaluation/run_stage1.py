"""
Stage 1: does hybrid fusion help, and on what?

The claim under audit, quoted from 2026 practitioner guidance:

    "Hybrid search consistently outperforms either method alone."
    "If your RAG system uses pure vector search, adding BM25 is the single
     highest-impact retrieval upgrade you can make."

Stated without conditions. This script measures it across corpora with two
dense retrievers of deliberately different strength, so the question is not
just whether fusion helps on average but whether the *same corpus* can show
fusion helping a weak retriever and hurting a strong one. That within-dataset
contrast is what separates a mechanism from a coincidence.

Hypothesis: the benefit of fusing BM25 into a dense run is inversely related
to how far that dense run already beats BM25 on the same corpus. If it holds,
a practitioner can predict ex ante whether to fuse, by running the two
retrievers separately first.

Usage:  python -m src.evaluation.run_stage1 [--datasets scifact nfcorpus ...]

Author: Manuel Corona
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.data.beir import BY_NAME, load
from src.evaluation.ir_metrics import evaluate_run
from src.evaluation.significance import paired_bootstrap
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import rrf_fuse

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "output" / "reports"

TOP_K = 100
RRF_K = 60          # the "zero-config default" the guidance recommends

# BGE wants an instruction prefix on the query side only. Omitting it is a
# silent multi-point nDCG loss, so it is spelled out rather than assumed.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

MODELS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "bge":    ("BAAI/bge-base-en-v1.5", BGE_QUERY_PREFIX),
}

REDUCED = ["scifact", "nfcorpus", "arguana", "fiqa"]


def drop_self(run: Dict[str, List[Tuple[str, float]]],
              top_k: int = TOP_K) -> Dict[str, List[Tuple[str, float]]]:
    """
    Remove the query's own document from its ranking.

    Several BEIR corpora reuse document ids as query ids -- 92% of ArguAna's
    judged queries and 8% of FiQA's -- and the query's own text is never
    among its relevant documents. Left in, a dense retriever matches it at
    cosine 1.0 and parks it at rank 1, pushing every real answer down a
    place. The BEIR reference implementation excludes it; this audit has to
    as well or its numbers are not comparable to the published ones.

    It is not a wash across systems, which is why it matters here: a dense
    model self-matches perfectly and BM25 only strongly, so leaving it in
    penalises exactly the retrievers this audit is comparing. Applied
    before fusion -- filtering afterwards would leave the spurious document
    occupying a rank in each input run and shifting every RRF score.
    """
    return {q: [(d, s) for d, s in hits if d != q][:top_k]
            for q, hits in run.items()}


def _ids(run: Dict[str, List[Tuple[str, float]]]) -> Dict[str, List[str]]:
    return {q: [d for d, _ in hits] for q, hits in run.items()}


def _ndcg_per_query(run, qrels, k: int = 10) -> np.ndarray:
    from src.evaluation.ir_metrics import ndcg_at_k
    ids = _ids(run)
    return np.array([ndcg_at_k(ids.get(q, []), qrels[q], k) for q in sorted(qrels)])


def run_dataset(name: str) -> Dict:
    meta = BY_NAME[name]
    t0 = time.time()
    corpus, queries, qrels = load(name, meta.split)
    doc_ids = list(corpus)
    doc_texts = [corpus[d] for d in doc_ids]
    print(f"\n{'='*70}\n{meta.label}  —  {len(corpus):,} docs, {len(queries):,} queries")
    print(f"{meta.note}\n{'='*70}")

    runs: Dict[str, Dict] = {}

    print("  BM25 ...", flush=True)
    t = time.time()
    bm25 = BM25Retriever(doc_ids, doc_texts)
    runs["bm25"] = drop_self(bm25.search_batch(queries, top_k=TOP_K + 1))
    print(f"    {time.time()-t:.1f}s")

    for short, (model_id, prefix) in MODELS.items():
        print(f"  dense: {short} ...", flush=True)
        t = time.time()
        r = DenseRetriever(model_id, query_prefix=prefix, dataset=name)
        r.index(doc_ids, doc_texts)
        runs[short] = drop_self(r.search_batch(queries, top_k=TOP_K + 1))
        print(f"    {time.time()-t:.1f}s")
        del r

    # The improvement under audit, applied to each dense retriever.
    for short in MODELS:
        runs[f"rrf_{short}"] = rrf_fuse([runs["bm25"], runs[short]],
                                        k=RRF_K, top_k=TOP_K)

    metrics = {arm: evaluate_run(_ids(run), qrels, k_values=(10, 100))
               for arm, run in runs.items()}

    # Every comparison that matters, as a paired bootstrap over queries.
    per_q = {arm: _ndcg_per_query(run, qrels) for arm, run in runs.items()}
    tests = {}
    for short in MODELS:
        tests[f"rrf_{short}_vs_{short}"] = paired_bootstrap(
            per_q[f"rrf_{short}"], per_q[short])
        tests[f"{short}_vs_bm25"] = paired_bootstrap(per_q[short], per_q["bm25"])

    self_match = sum(1 for q in qrels if q in corpus)
    out = {
        "dataset": name,
        "self_match_queries": self_match,
        "self_match_share": round(self_match / len(qrels), 4),
        "label": meta.label,
        "domain": meta.domain,
        "note": meta.note,
        "documents": len(corpus),
        "queries": len(queries),
        "seconds": round(time.time() - t0, 1),
        "metrics": metrics,
        "tests": tests,
    }

    print()
    for arm in ("bm25", "minilm", "bge", "rrf_minilm", "rrf_bge"):
        print(f"    {arm:<12} nDCG@10 {metrics[arm]['ndcg@10']:.4f}   "
              f"R@100 {metrics[arm]['recall@100']:.4f}")
    for short in MODELS:
        t_ = tests[f"rrf_{short}_vs_{short}"]
        gap = metrics[short]["ndcg@10"] - metrics["bm25"]["ndcg@10"]
        verdict = "HELPS" if t_["mean_difference"] > 0 and t_["p_value"] < 0.05 else \
                  "HURTS" if t_["mean_difference"] < 0 and t_["p_value"] < 0.05 else "no effect"
        print(f"    fusion on {short:<7} {t_['mean_difference']:+.4f} "
              f"[{t_['ci_low']:+.4f},{t_['ci_high']:+.4f}] p={t_['p_value']:.4f}  "
              f"{verdict:<9} (dense−bm25 gap {gap:+.4f})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=REDUCED)
    args = ap.parse_args()

    results = [run_dataset(n) for n in args.datasets]

    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "stage1_fusion.json"
    payload = {
        "claim_under_audit": (
            "Hybrid search consistently outperforms either method alone; "
            "adding BM25 to pure vector search is the single highest-impact "
            "retrieval upgrade."),
        "rrf_k": RRF_K,
        "top_k": TOP_K,
        "datasets": results,
    }
    path.write_text(json.dumps(payload, indent=2))

    print(f"\n\n{'='*78}\nSUMMARY — does fusing BM25 into a dense run help?\n{'='*78}")
    print(f"{'dataset':<12} {'retriever':<8} {'dense':>8} {'+BM25':>8} "
          f"{'delta':>9} {'p':>8}  {'gap vs bm25':>12}  verdict")
    rows = []
    for r in results:
        for short in MODELS:
            m, t_ = r["metrics"], r["tests"][f"rrf_{short}_vs_{short}"]
            gap = m[short]["ndcg@10"] - m["bm25"]["ndcg@10"]
            verdict = ("HELPS" if t_["mean_difference"] > 0 and t_["p_value"] < 0.05
                       else "HURTS" if t_["mean_difference"] < 0 and t_["p_value"] < 0.05
                       else "n.s.")
            print(f"{r['label']:<12} {short:<8} {m[short]['ndcg@10']:>8.4f} "
                  f"{m['rrf_'+short]['ndcg@10']:>8.4f} {t_['mean_difference']:>+9.4f} "
                  f"{t_['p_value']:>8.4f}  {gap:>+12.4f}  {verdict}")
            rows.append((gap, t_["mean_difference"]))

    gaps = np.array([g for g, _ in rows])
    deltas = np.array([d for _, d in rows])
    if len(gaps) > 2:
        r_ = float(np.corrcoef(gaps, deltas)[0, 1])
        print(f"\nHypothesis: fusion benefit falls as the dense retriever's own "
              f"advantage over BM25 grows.\n"
              f"  correlation(dense−bm25 gap, fusion delta) = {r_:+.3f}  "
              f"over {len(gaps)} (dataset, retriever) pairs")
        payload["hypothesis_correlation"] = r_
        payload["pairs"] = [{"gap": float(g), "fusion_delta": float(d)} for g, d in rows]
        path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
