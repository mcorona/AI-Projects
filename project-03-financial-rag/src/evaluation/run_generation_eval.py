"""
End-to-end RAG evaluation: generate answers under four conditions, judge
them, and report what actually moved the numbers.

The four conditions isolate one variable at a time:

    closed_book   no passages. What the model already knows. RAG has to
                  beat this or retrieval is decoration.
    rag_bm25      top-5 from BM25 (nDCG@10 = 0.237).
    rag_dense     top-5 from bge-base (nDCG@10 = 0.406) -- the best
                  retriever from the Phase 1 benchmark.
    oracle        the human-judged relevant documents. The ceiling: it
                  separates retrieval failure from generation failure.

rag_bm25 vs rag_dense is the question most RAG writeups never ask: a
17-point nDCG gap exists upstream, and the point of this phase is to find
out how much of it survives into answer quality.

Phases are separate and restartable, because batches can take an hour:

    python -m src.evaluation.run_generation_eval --phase pilot --n 4
    python -m src.evaluation.run_generation_eval --phase generate
    python -m src.evaluation.run_generation_eval --phase judge
    python -m src.evaluation.run_generation_eval --phase analyze

Author: Manuel Corona
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import anthropic

from src.data.loader import load_split
from src.generation import batch as batchlib
from src.generation.judge import (
    CROSS_JUDGE_MODEL,
    JUDGE_MODEL,
    citation_precision,
    correctness_params,
    groundedness_params,
    invalid_citation_rate,
)
from src.generation.rag import GENERATOR_MODEL, batch_params, format_context

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "data" / "processed" / "runs"
WORK_DIR = ROOT / "data" / "processed" / "generation"
REPORTS_DIR = ROOT / "output" / "reports"

N_QUESTIONS = 150
TOP_K_CONTEXT = 5
SAMPLE_SEED = 20260827

# (condition, retrieval run to pull context from). None = no context,
# "gold" = the qrels' relevant documents.
CONDITIONS = [
    ("closed_book", None),
    ("rag_bm25", "bm25"),
    ("rag_dense", "dense_bge"),
    ("oracle", "gold"),
]
CONTEXT_CONDITIONS = [c for c, src in CONDITIONS if src is not None]

# Conditions re-judged for self-consistency and by a second model. Kept to
# the two that carry the headline claim (does retrieval beat no retrieval)
# so the calibration checks cost a fraction of the main run.
CHECK_CONDITIONS = ["closed_book", "rag_dense"]


def load_run(name: str):
    with open(RUNS_DIR / f"{name}.json") as f:
        return {q: [d for d, _ in v] for q, v in json.load(f).items()}


def sample_questions(queries: Dict[str, str]) -> List[str]:
    """A fixed, seeded sample so every phase and re-run scores the same items."""
    qids = sorted(queries)
    return sorted(random.Random(SAMPLE_SEED).sample(qids, min(N_QUESTIONS, len(qids))))


def build_contexts(qids, corpus, qrels) -> Dict[str, Dict[str, Optional[dict]]]:
    """
    For each condition, the passage ids shown to the model and their
    rendered text. Returns {condition: {qid: {"ids": [...], "text": str}}}.
    """
    contexts: Dict[str, Dict[str, Optional[dict]]] = {}
    for condition, source in CONDITIONS:
        if source is None:
            contexts[condition] = {qid: None for qid in qids}
            continue
        if source == "gold":
            # Gold documents in a fixed order (sorted by id) so the context
            # is deterministic and does not leak the retriever's ranking.
            picked = {qid: sorted(qrels[qid])[:TOP_K_CONTEXT] for qid in qids}
        else:
            run = load_run(source)
            picked = {qid: run.get(qid, [])[:TOP_K_CONTEXT] for qid in qids}
        contexts[condition] = {
            qid: {"ids": ids, "text": format_context(ids, corpus)}
            for qid, ids in picked.items()
        }
    return contexts


def gold_reference(qid, qrels, corpus) -> str:
    """The human-judged relevant passages, used as the correctness reference."""
    return format_context(sorted(qrels[qid]), corpus)


# --- Phases ---------------------------------------------------------------

def phase_pilot(client, n, queries, corpus, qrels):
    """
    Run a handful of questions synchronously and price the full run from
    measured tokens rather than a guess. Cheap insurance before spending
    real money on 2,000 batched requests.
    """
    from src.generation.rag import answer
    qids = sample_questions(queries)[:n]
    contexts = build_contexts(qids, corpus, qrels)
    totals = Counter()
    for qid in qids:
        for condition, _ in CONDITIONS:
            ctx = contexts[condition][qid]
            a, usage = answer(client, queries[qid], ctx["text"] if ctx else None)
            totals["input"] += usage.input_tokens
            totals["output"] += usage.output_tokens
            totals["n"] += 1
            print(f"  {condition:12s} {qid:>5s} cited={a.citations} "
                  f"sufficient={a.context_sufficient} "
                  f"({usage.input_tokens} in / {usage.output_tokens} out)")

    per_req_in = totals["input"] / totals["n"]
    per_req_out = totals["output"] / totals["n"]
    gen_reqs = N_QUESTIONS * len(CONDITIONS)
    # Judge request shape, measured rather than assumed: a judge reads the
    # answer plus its passages (~1.15x a generation request's input) but
    # emits only a verdict and a sentence of reasoning (~0.55x the output).
    # The first version of this estimator guessed 1.8x on output and
    # overpriced the run by 60%.
    judge_reqs = (N_QUESTIONS * len(CONTEXT_CONDITIONS)      # groundedness
                  + N_QUESTIONS * len(CONDITIONS)            # correctness
                  + N_QUESTIONS * len(CHECK_CONDITIONS))     # self-consistency
    cross_reqs = N_QUESTIONS * len(CHECK_CONDITIONS)

    def usd(reqs, in_tok, out_tok, model):
        return batchlib.estimate_cost(
            {"input_tokens": int(reqs * in_tok), "output_tokens": int(reqs * out_tok)}, model)

    gen = usd(gen_reqs, per_req_in, per_req_out, GENERATOR_MODEL)
    judge = usd(judge_reqs, per_req_in * 1.15, per_req_out * 0.55, JUDGE_MODEL)
    cross = usd(cross_reqs, per_req_in * 1.15, per_req_out * 0.55, CROSS_JUDGE_MODEL)
    print(f"\nMeasured: {per_req_in:.0f} input / {per_req_out:.0f} output tokens per request")
    print(f"Projected full run at N={N_QUESTIONS} (batch pricing):")
    print(f"  generation      {gen_reqs:5d} requests   ~${gen:6.2f}")
    print(f"  judging         {judge_reqs:5d} requests   ~${judge:6.2f}")
    print(f"  cross-model     {cross_reqs:5d} requests   ~${cross:6.2f}")
    print(f"  TOTAL                            ~${gen + judge + cross:6.2f}")


def phase_generate(client, queries, corpus, qrels):
    qids = sample_questions(queries)
    contexts = build_contexts(qids, corpus, qrels)
    requests = []
    for condition, _ in CONDITIONS:
        for qid in qids:
            ctx = contexts[condition][qid]
            requests.append((f"{condition}__{qid}",
                             batch_params(queries[qid], ctx["text"] if ctx else None)))

    results, usage = batchlib.run(client, requests, "generation", GENERATOR_MODEL)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    records = {}
    for condition, _ in CONDITIONS:
        for qid in qids:
            key = f"{condition}__{qid}"
            r = results.get(key, {"error": "missing"})
            ctx = contexts[condition][qid]
            records[key] = {
                "condition": condition, "qid": qid, "question": queries[qid],
                "context_ids": ctx["ids"] if ctx else [],
                "answer": r.get("parsed"), "error": r.get("error"),
            }
    (WORK_DIR / "answers.json").write_text(json.dumps(records, indent=2))
    (WORK_DIR / "generation_usage.json").write_text(json.dumps(usage, indent=2))
    print(f"wrote {len(records)} answers")


def phase_judge(client, queries, corpus, qrels):
    records = json.loads((WORK_DIR / "answers.json").read_text())
    qids = sample_questions(queries)
    contexts = build_contexts(qids, corpus, qrels)
    references = {qid: gold_reference(qid, qrels, corpus) for qid in qids}

    ground_reqs, correct_reqs, repeat_reqs, cross_reqs = [], [], [], []
    for key, rec in records.items():
        if not rec["answer"]:
            continue
        cond, qid, text = rec["condition"], rec["qid"], rec["answer"]["answer"]
        if cond in CONTEXT_CONDITIONS:
            ctx = contexts[cond][qid]["text"]
            ground_reqs.append((key, groundedness_params(queries[qid], ctx, text)))
        correct_reqs.append((key, correctness_params(queries[qid], references[qid], text)))
        if cond in CHECK_CONDITIONS:
            repeat_reqs.append((key, correctness_params(queries[qid], references[qid], text)))
            cross_reqs.append((key, correctness_params(queries[qid], references[qid], text,
                                                      model=CROSS_JUDGE_MODEL)))

    out, usages = {}, {}
    for tag, reqs, model in (
        ("judge_groundedness", ground_reqs, JUDGE_MODEL),
        ("judge_correctness", correct_reqs, JUDGE_MODEL),
        ("judge_correctness_repeat", repeat_reqs, JUDGE_MODEL),
        ("judge_correctness_cross", cross_reqs, CROSS_JUDGE_MODEL),
    ):
        results, usage = batchlib.run(client, reqs, tag, model)
        out[tag] = {k: v.get("parsed") for k, v in results.items()}
        usages[tag] = usage

    (WORK_DIR / "judgments.json").write_text(json.dumps(out, indent=2))
    (WORK_DIR / "judge_usage.json").write_text(json.dumps(usages, indent=2))
    total = sum(u["estimated_usd"] for u in usages.values())
    print(f"judging cost ~${total:.2f}")


def phase_analyze(queries, corpus, qrels):
    records = json.loads((WORK_DIR / "answers.json").read_text())
    judgments = json.loads((WORK_DIR / "judgments.json").read_text())
    ground = judgments["judge_groundedness"]
    correct = judgments["judge_correctness"]
    repeat = judgments["judge_correctness_repeat"]
    cross = judgments["judge_correctness_cross"]

    by_condition = defaultdict(lambda: defaultdict(list))
    for key, rec in records.items():
        cond, qid, ans = rec["condition"], rec["qid"], rec["answer"]
        if not ans:
            by_condition[cond]["generation_errors"].append(1)
            continue
        c = correct.get(key)
        if c:
            by_condition[cond]["correct"].append(c["verdict"] == "correct")
            by_condition[cond]["correct_or_partial"].append(
                c["verdict"] in ("correct", "partially_correct"))
            by_condition[cond]["declined"].append(c["verdict"] == "no_answer")
        g = ground.get(key)
        if g:
            by_condition[cond]["grounded"].append(g["verdict"] == "grounded")
            by_condition[cond]["ungrounded"].append(g["verdict"] == "ungrounded")
        if cond in CONTEXT_CONDITIONS:
            by_condition[cond]["abstained"].append(not ans["context_sufficient"])
            by_condition[cond]["invalid_citations"].append(
                invalid_citation_rate(ans["citations"], len(rec["context_ids"])))
            prec = citation_precision(ans["citations"], rec["context_ids"], qrels[qid])
            if prec is not None:
                by_condition[cond]["citation_precision"].append(prec)

    report = {"n_questions": N_QUESTIONS, "top_k_context": TOP_K_CONTEXT,
              "generator_model": GENERATOR_MODEL, "judge_model": JUDGE_MODEL,
              "conditions": {}}
    for cond, _ in CONDITIONS:
        m = by_condition[cond]
        report["conditions"][cond] = {
            k: round(float(sum(v) / len(v)), 4) for k, v in m.items() if v
        }
        report["conditions"][cond]["n_scored"] = len(m["correct"])

    # Judge calibration. A judge that disagrees with itself, or with a
    # different model, on a large fraction of items cannot resolve small
    # differences between conditions -- and the reader deserves to know
    # that before reading the table above.
    def agreement(a: dict, b: dict) -> Dict[str, float]:
        keys = [k for k in a if a.get(k) and b.get(k)]
        if not keys:
            return {}
        same = sum(a[k]["verdict"] == b[k]["verdict"] for k in keys)
        binary = sum((a[k]["verdict"] == "correct") == (b[k]["verdict"] == "correct")
                     for k in keys)
        return {"n": len(keys), "exact_agreement": round(same / len(keys), 4),
                "binary_agreement": round(binary / len(keys), 4)}

    report["judge_calibration"] = {
        "self_consistency": agreement(correct, repeat),
        "cross_model": agreement(correct, cross),
        "cross_model_id": CROSS_JUDGE_MODEL,
    }

    usage_files = {"generation_usage.json": "generation", "judge_usage.json": "judging"}
    costs = {}
    for fname, label in usage_files.items():
        path = WORK_DIR / fname
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        costs[label] = (round(data["estimated_usd"], 2) if "estimated_usd" in data
                        else round(sum(u["estimated_usd"] for u in data.values()), 2))
    report["estimated_cost_usd"] = costs

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "generation_metrics.json").write_text(json.dumps(report, indent=2))

    print(f"\n=== Answer quality, N={N_QUESTIONS} FiQA test questions ===")
    hdr = f"{'condition':14s} {'correct':>8s} {'+partial':>9s} {'grounded':>9s} " \
          f"{'cite prec':>10s} {'abstained':>10s}"
    print(hdr)
    for cond, _ in CONDITIONS:
        c = report["conditions"][cond]
        def fmt(key):
            return f"{c[key]:.3f}" if key in c else "     -"
        print(f"{cond:14s} {fmt('correct'):>8s} {fmt('correct_or_partial'):>9s} "
              f"{fmt('grounded'):>9s} {fmt('citation_precision'):>10s} "
              f"{fmt('abstained'):>10s}")
    cal = report["judge_calibration"]
    print(f"\njudge self-consistency: {cal['self_consistency']}")
    print(f"judge vs {CROSS_JUDGE_MODEL}: {cal['cross_model']}")
    print(f"estimated cost: {costs}")
    print(f"\nwrote {(REPORTS_DIR / 'generation_metrics.json').relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True,
                        choices=["pilot", "generate", "judge", "analyze"])
    parser.add_argument("--n", type=int, default=3, help="questions for the pilot phase")
    args = parser.parse_args()

    corpus, queries, qrels = load_split("test")
    if args.phase == "analyze":
        phase_analyze(queries, corpus, qrels)
        return

    client = anthropic.Anthropic()
    if args.phase == "pilot":
        phase_pilot(client, args.n, queries, corpus, qrels)
    elif args.phase == "generate":
        phase_generate(client, queries, corpus, qrels)
    elif args.phase == "judge":
        phase_judge(client, queries, corpus, qrels)


if __name__ == "__main__":
    main()
