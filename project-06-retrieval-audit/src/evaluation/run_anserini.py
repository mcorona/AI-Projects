"""
Stage 3: replicate the BM25 half of the audit on Anserini.

Two questions, in order.

  1. Does this pipeline reproduce the BEIR paper's BM25 figures when it uses
     the BEIR paper's BM25? If not, something other than the implementation
     is wrong and the whole audit is suspect.

  2. Does the audit's conclusion survive swapping rank_bm25 for Anserini?
     The fusion result depends on BM25 on both axes, so this is the test the
     README named as the highest-value follow-up.

Writes BM25 runs to disk for the fusion step, which runs in the other
virtualenv because it needs sentence-transformers.

Usage:  ./venv-anserini/bin/python -m src.evaluation.run_anserini

Author: Manuel Corona
"""

import json
import pickle
import time
from pathlib import Path

from src.data.beir import BY_NAME, DATASETS
from src.evaluation.ir_metrics import evaluate_run
from src.evaluation.reference import REFERENCE
from src.retrieval.anserini import AnseriniBM25, PREBUILT

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "data" / "processed" / "runs"
REPORTS = ROOT / "output" / "reports"
TOP_K = 100


def drop_self(run, top_k=TOP_K):
    """Same exclusion the rest of the audit applies, for the same reason."""
    return {q: [(d, s) for d, s in hits if d != q][:top_k]
            for q, hits in run.items()}


def main():
    from src.data.beir import load
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in DATASETS:
        if d.name not in PREBUILT:
            continue
        corpus, queries, qrels = load(d.name, d.split)
        t0 = time.time()
        bm25 = AnseriniBM25(d.name)
        run = drop_self(bm25.search_batch(queries, top_k=TOP_K + 1))
        secs = time.time() - t0

        with (RUNS_DIR / f"{d.name}__anserini_bm25.pkl").open("wb") as fh:
            pickle.dump(run, fh)

        m = evaluate_run({q: [x for x, _ in h] for q, h in run.items()},
                         qrels, k_values=(10, 100))
        published = REFERENCE[d.name]["bm25"][0]
        row = {
            "dataset": d.name, "label": d.label,
            "indexed_docs": bm25.num_docs, "corpus_docs": len(corpus),
            "ndcg@10": m["ndcg@10"], "recall@100": m["recall@100"],
            "published": published,
            "delta_vs_published": m["ndcg@10"] - published,
            "seconds": round(secs, 1),
        }
        rows.append(row)
        print(f"{d.label:<12} {m['ndcg@10']:.4f}  publicado {published:.3f}  "
              f"Δ {row['delta_vs_published']:+.4f}   "
              f"({bm25.num_docs:,} docs indexados / {len(corpus):,} en corpus, "
              f"{secs:.0f}s)")

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "anserini_bm25.json").write_text(json.dumps(
        {"k1": 0.9, "b": 0.4, "source": "pyserini prebuilt beir-v1.0.0 *.flat",
         "datasets": rows}, indent=2))
    worst = max(rows, key=lambda r: abs(r["delta_vs_published"]))
    print(f"\nmayor desviación: {worst['label']} {worst['delta_vs_published']:+.4f}")
    print(f"wrote {(REPORTS / 'anserini_bm25.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
