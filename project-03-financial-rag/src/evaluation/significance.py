"""
Paired bootstrap significance testing for retrieval runs.

648 queries is enough to be confident about a 17-point nDCG gap and not
nearly enough to be confident about a 1-point one. Reporting both as
"model A beats model B" is how leaderboards mislead, so every comparison
that appears in the README is run through this first.

Method: paired bootstrap over queries (10,000 resamples). For each
resample, draw 648 queries with replacement and recompute the difference
in mean nDCG@10 between the two systems on that resample. The two-sided
p-value is the fraction of resamples where the difference has the opposite
sign to (or is equal to) the observed one, doubled. Pairing matters: the
same query is either easy or hard for both systems, and an unpaired test
would drown the signal in that shared query difficulty.

Author: Manuel Corona
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.evaluation.ir_metrics import ndcg_at_k

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "data" / "processed" / "runs"


def per_query_scores(run: Dict[str, List[Tuple[str, float]]],
                     qrels: Dict[str, Dict[str, int]], k: int = 10) -> np.ndarray:
    """nDCG@k for every judged query, in a fixed (sorted) query order."""
    return np.array([ndcg_at_k([d for d, _ in run.get(qid, [])], qrels[qid], k)
                     for qid in sorted(qrels)])


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_resamples: int = 10_000,
                     seed: int = 42) -> Dict[str, float]:
    """
    Test whether system `a` differs from system `b`.

    Returns the observed mean difference, a 95% percentile confidence
    interval for it, and a two-sided p-value.
    """
    rng = np.random.default_rng(seed)
    diff = a - b
    observed = float(diff.mean())
    idx = rng.integers(0, len(diff), size=(n_resamples, len(diff)))
    boot = diff[idx].mean(axis=1)
    # Two-sided p: how often a resample lands on the other side of zero.
    tail = float(np.mean(boot <= 0) if observed > 0 else np.mean(boot >= 0))
    return {
        "mean_difference": observed,
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_value": min(1.0, 2 * tail),
        "n_queries": len(diff),
    }


def compare(name_a: str, name_b: str, qrels, k: int = 10) -> Dict[str, float]:
    def load(name):
        with open(RUNS_DIR / f"{name}.json") as f:
            return {q: [(d, s) for d, s in v] for q, v in json.load(f).items()}
    return paired_bootstrap(per_query_scores(load(name_a), qrels, k),
                            per_query_scores(load(name_b), qrels, k))


def mcnemar_exact(a_correct: Dict[str, bool], b_correct: Dict[str, bool]) -> Dict[str, float]:
    """
    Exact McNemar test on paired binary outcomes.

    The right test for "did system A answer more questions correctly than
    system B" when both were run on the *same* questions. Only the
    discordant pairs carry information -- questions both got right, or both
    got wrong, say nothing about which is better -- so an unpaired
    proportion test on the same data would be answering a different and
    easier question.

    Uses the exact binomial rather than the chi-square approximation
    because the discordant count here is small (tens, not hundreds).
    """
    from math import comb
    keys = [k for k in a_correct if k in b_correct]
    b = sum(1 for k in keys if a_correct[k] and not b_correct[k])
    c = sum(1 for k in keys if not a_correct[k] and b_correct[k])
    n = b + c
    if n == 0:
        return {"a_only": 0, "b_only": 0, "discordant": 0, "p_value": 1.0}
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return {"a_only": b, "b_only": c, "discordant": n,
            "p_value": round(min(1.0, 2 * tail), 6)}


COMPARISONS = [
    # Does the dense retriever earn its GPU over the lexical baseline?
    ("dense_bge", "bm25"),
    # Does the bigger dense model earn its 3.5x indexing cost over MiniLM?
    ("dense_bge", "dense_minilm"),
    # Does hybrid fusion help, as the standard RAG recipe assumes?
    ("hybrid", "dense_bge"),
    # Does reranking help, as the standard RAG recipe assumes?
    ("rerank_dense", "dense_bge"),
    ("rerank_dense_bge", "dense_bge"),
    # Diagnostic: the reranker on a weak first stage. Confirms it works.
    ("rerank_bm25", "bm25"),
]


def main():
    from src.data.loader import load_split
    _, _, qrels = load_split("test")
    report = {}
    print(f"{'comparison':38s} {'Δ nDCG@10':>10s}  {'95% CI':>20s}  {'p':>8s}")
    for a, b in COMPARISONS:
        r = compare(a, b, qrels)
        report[f"{a}_vs_{b}"] = r
        sig = "" if r["p_value"] < 0.05 else "  (n.s.)"
        print(f"{a + ' vs ' + b:38s} {r['mean_difference']:+10.4f}  "
              f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  {r['p_value']:8.4f}{sig}")
    out = ROOT / "output" / "reports" / "retrieval_significance.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
