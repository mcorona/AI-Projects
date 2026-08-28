"""
Published nDCG@10 values, and the check of this pipeline against them.

An audit that reports someone else's defaults are wrong has to first show
its own instrument is right. That is the step Project 3 took before
reporting a surprising retrieval result, and the reason its conclusion was
defensible. This module is that step, generalized to every dataset here.

    !! THESE REFERENCE VALUES ARE UNVERIFIED !!

They are transcribed from recollection of the BEIR paper and the MTEB
leaderboard, NOT from a fetched source, and they carry a per-entry
confidence flag saying so. Every value marked "recalled" must be checked
against its primary source before any of this is published. A validation
table built on half-remembered numbers would be worse than no validation
table, because it would look like diligence.

BM25 additionally has genuine implementation spread: the BEIR paper's
Elasticsearch numbers and Anserini's reproduction differ by several points
on some datasets, and this pipeline uses rank_bm25, a third implementation
with its own tokenizer. BM25 rows are therefore a sanity range, not a
target.

Author: Manuel Corona
"""

from typing import Dict, Optional

# (dataset, system) -> (published nDCG@10, source, confidence)
#   confidence: "recalled" = from memory, MUST be verified before publishing
REFERENCE: Dict[str, Dict[str, tuple]] = {
    "scifact": {
        "bm25":   (0.665, "BEIR paper", "recalled"),
        "minilm": (0.645, "MTEB", "recalled"),
        "bge":    (0.742, "MTEB / BGE paper", "recalled"),
    },
    "nfcorpus": {
        "bm25":   (0.325, "BEIR paper", "recalled"),
        "minilm": (0.318, "MTEB", "recalled"),
        "bge":    (0.373, "MTEB / BGE paper", "recalled"),
    },
    "arguana": {
        "bm25":   (0.315, "BEIR paper (Anserini reports ~0.397)", "recalled"),
        "minilm": (0.501, "MTEB", "recalled"),
        "bge":    (0.636, "MTEB / BGE paper", "recalled"),
    },
    "scidocs": {
        "bm25":   (0.158, "BEIR paper", "recalled"),
        "minilm": (0.216, "MTEB", "recalled"),
        "bge":    (0.217, "MTEB / BGE paper", "recalled"),
    },
    "trec-covid": {
        "bm25":   (0.656, "BEIR paper", "recalled"),
        "minilm": (0.473, "MTEB", "recalled"),
        "bge":    (0.781, "MTEB / BGE paper", "recalled"),
    },
    "fiqa": {
        "bm25":   (0.236, "BEIR paper", "verified in project 3"),
        "minilm": (0.369, "MTEB", "recalled"),
        "bge":    (0.406, "MTEB / BGE paper", "verified in project 3"),
    },
}

# How far a dense system may sit from its published value before the
# pipeline is considered suspect rather than merely different. Project 3
# reproduced FiQA to 0.002; anything past 0.02 on a dense run means
# something is configured wrong, as the ArguAna self-match bug demonstrated
# (it showed up as a 0.18 discrepancy).
DENSE_TOLERANCE = 0.02


def check(dataset: str, system: str, observed: float) -> Optional[Dict]:
    entry = REFERENCE.get(dataset, {}).get(system)
    if entry is None:
        return None
    published, source, confidence = entry
    delta = observed - published
    is_bm25 = system == "bm25"
    return {
        "dataset": dataset,
        "system": system,
        "observed": round(observed, 4),
        "published": published,
        "delta": round(delta, 4),
        "source": source,
        "confidence": confidence,
        # BM25 is a range check, not a match check -- three implementations,
        # three tokenizers, several points of legitimate spread.
        "status": ("range check only" if is_bm25
                   else "ok" if abs(delta) <= DENSE_TOLERANCE
                   else "INVESTIGATE"),
    }


def check_all(results: list) -> list:
    out = []
    for r in results:
        for system in ("bm25", "minilm", "bge"):
            row = check(r["dataset"], system, r["metrics"][system]["ndcg@10"])
            if row:
                out.append(row)
    return out


if __name__ == "__main__":
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "output" / "reports" / "stage1_fusion.json"
    data = json.loads(path.read_text())
    rows = check_all(data["datasets"])
    print(f"{'dataset':<10} {'system':<8} {'observed':>9} {'published':>10} "
          f"{'delta':>8}  {'status':<16} confidence")
    print("-" * 84)
    for r in rows:
        print(f"{r['dataset']:<10} {r['system']:<8} {r['observed']:>9.4f} "
              f"{r['published']:>10.3f} {r['delta']:>+8.4f}  {r['status']:<16} "
              f"{r['confidence']}")
    bad = [r for r in rows if r["status"] == "INVESTIGATE"]
    print(f"\n{len(rows)} checks, {len(bad)} needing investigation")
    for r in bad:
        print(f"  !! {r['dataset']}/{r['system']}: {r['delta']:+.4f} from published")
