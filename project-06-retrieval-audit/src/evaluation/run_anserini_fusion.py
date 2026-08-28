"""
Stage 3b: redo the fusion analysis with Anserini's BM25 instead of rank_bm25.

The audit's headline is a relationship between a dense retriever's advantage
over BM25 and what fusion does to it. BM25 appears on both axes, so a
different BM25 could in principle move the whole picture. This recomputes
every fusion comparison against runs produced by the reference
implementation and reports whether the conclusion holds.

Runs in the main virtualenv (needs sentence-transformers); reads the BM25
runs written by run_anserini.py in the pyserini one.

Usage:  ../project-03-financial-rag/venv/bin/python -m src.evaluation.run_anserini_fusion
"""

import json
import pickle
from pathlib import Path

import numpy as np

from src.data.beir import BY_NAME, DATASETS, load
from src.evaluation.ir_metrics import evaluate_run, ndcg_at_k
from src.evaluation.run_stage1 import BGE_QUERY_PREFIX, MODELS, RRF_K, TOP_K, drop_self
from src.evaluation.significance import paired_bootstrap
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import rrf_fuse

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "data" / "processed" / "runs"
REPORTS = ROOT / "output" / "reports"


def per_q(run, qrels, k=10):
    return np.array([ndcg_at_k([d for d, _ in run.get(q, [])], qrels[q], k)
                     for q in sorted(qrels)])


def main():
    prior = json.loads((REPORTS / "stage1_fusion.json").read_text())
    by_ds = {r["dataset"]: r for r in prior["datasets"]}

    rows, out = [], []
    for d in DATASETS:
        pkl = RUNS_DIR / f"{d.name}__anserini_bm25.pkl"
        if not pkl.exists():
            continue
        corpus, queries, qrels = load(d.name, d.split)
        doc_ids = list(corpus)
        doc_texts = [corpus[x] for x in doc_ids]

        runs = {"bm25": pickle.load(pkl.open("rb"))}
        for short, (model_id, prefix) in MODELS.items():
            r = DenseRetriever(model_id, query_prefix=prefix, dataset=d.name)
            r.index(doc_ids, doc_texts)
            runs[short] = drop_self(r.search_batch(queries, top_k=TOP_K + 1))
            del r
        for short in MODELS:
            runs[f"rrf_{short}"] = rrf_fuse([runs["bm25"], runs[short]],
                                            k=RRF_K, top_k=TOP_K)

        metrics = {a: evaluate_run({q: [x for x, _ in h] for q, h in r.items()},
                                   qrels, k_values=(10, 100))
                   for a, r in runs.items()}
        pq = {a: per_q(r, qrels) for a, r in runs.items()}

        entry = {"dataset": d.name, "label": d.label, "metrics": metrics, "tests": {}}
        for short in MODELS:
            t = paired_bootstrap(pq[f"rrf_{short}"], pq[short])
            entry["tests"][f"rrf_{short}_vs_{short}"] = t
            gap = metrics[short]["ndcg@10"] - metrics["bm25"]["ndcg@10"]
            old = by_ds[d.name]
            old_gap = old["metrics"][short]["ndcg@10"] - old["metrics"]["bm25"]["ndcg@10"]
            old_delta = old["tests"][f"rrf_{short}_vs_{short}"]["mean_difference"]
            rows.append({"dataset": d.name, "label": d.label, "retriever": short,
                         "gap": float(gap), "fusion_delta": float(t["mean_difference"]),
                         "p_value": float(t["p_value"]),
                         "gap_rank_bm25": float(old_gap),
                         "fusion_delta_rank_bm25": float(old_delta)})
        out.append(entry)
        print(f"  {d.label} listo")

    g = np.array([r["gap"] for r in rows]); dd = np.array([r["fusion_delta"] for r in rows])
    corr = float(np.corrcoef(g, dd)[0, 1])
    og = np.array([r["gap_rank_bm25"] for r in rows])
    od = np.array([r["fusion_delta_rank_bm25"] for r in rows])
    corr_old = float(np.corrcoef(og, od)[0, 1])

    print(f"\n{'corpus':<12} {'denso':<7} {'brecha':>9} {'Δ fusión':>10} {'p':>8}   "
          f"{'(rank_bm25: brecha':>19} {'Δ)':>9}")
    for r in sorted(rows, key=lambda x: x["gap"]):
        v = ("AYUDA" if r["fusion_delta"] > 0 and r["p_value"] < 0.05 else
             "PERJUDICA" if r["fusion_delta"] < 0 and r["p_value"] < 0.05 else "n.s.")
        print(f"{r['label']:<12} {r['retriever']:<7} {r['gap']:>+9.4f} "
              f"{r['fusion_delta']:>+10.4f} {r['p_value']:>8.4f}   "
              f"{r['gap_rank_bm25']:>+19.4f} {r['fusion_delta_rank_bm25']:>+9.4f}  {v}")
    print(f"\ncorrelación con Anserini  : {corr:+.3f}")
    print(f"correlación con rank_bm25 : {corr_old:+.3f}")

    (REPORTS / "anserini_fusion.json").write_text(json.dumps(
        {"correlation_anserini": corr, "correlation_rank_bm25": corr_old,
         "pairs": rows, "datasets": out}, indent=2))
    print(f"wrote {(REPORTS / 'anserini_fusion.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
