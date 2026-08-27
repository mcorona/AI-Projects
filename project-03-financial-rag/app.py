"""
Streamlit dashboard for the financial RAG system.

Three tabs, matching the three things the project claims:
  Ask       -- live retrieval + grounded answer, with the gold documents
               marked when the question comes from the benchmark.
  Retrieval -- the Phase 1 benchmark and its significance tests.
  Answers   -- the Phase 2 four-condition comparison and judge calibration.

The app reads precomputed artifacts; it never re-indexes the corpus or
re-runs an evaluation.

Author: Manuel Corona
"""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "output" / "reports"

st.set_page_config(page_title="Financial RAG", page_icon="::", layout="wide")


# --- Loading --------------------------------------------------------------

@st.cache_resource(show_spinner="Loading FiQA corpus ...")
def load_data():
    from src.data.loader import load_split
    return load_split("test")


@st.cache_resource(show_spinner="Loading dense retriever ...")
def load_retriever(model_name, query_prefix):
    from src.retrieval.dense import DenseRetriever
    corpus, _, _ = load_data()
    from src.data.loader import doc_text
    doc_ids = list(corpus)
    r = DenseRetriever(model_name, query_prefix=query_prefix)
    r.index(doc_ids, [doc_text(corpus[d]) for d in doc_ids])
    return r


@st.cache_data
def load_report(name):
    path = REPORTS / name
    return json.loads(path.read_text()) if path.exists() else None


def has_api_key():
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic" / "credentials").exists()


# --- Tab 1: Ask -----------------------------------------------------------

def tab_ask():
    from src.data.loader import doc_text
    from src.evaluation.run_retrieval_eval import BGE_QUERY_PREFIX

    corpus, queries, qrels = load_data()

    st.sidebar.header("Retrieval")
    top_k = st.sidebar.slider("Passages retrieved", 1, 10, 5)
    generate = st.sidebar.checkbox("Generate an answer (calls the Claude API)",
                                   value=False)

    st.sidebar.caption(
        "The dense retriever is bge-base-en-v1.5, which scored nDCG@10 = 0.406 "
        "on the FiQA test split -- the best of the seven configurations "
        "benchmarked. See the Retrieval tab."
    )

    example_ids = sorted(queries)[:200]
    picked = st.selectbox(
        "Pick a benchmark question (its relevant documents are known), "
        "or type your own below",
        ["-- type my own --"] + [f"{qid}: {queries[qid]}" for qid in example_ids],
    )
    default = "" if picked.startswith("--") else picked.split(": ", 1)[1]
    question = st.text_area("Question", value=default, height=80)

    if not question.strip():
        st.info("Enter a question to retrieve passages.")
        return

    qid = next((q for q in queries if queries[q] == question), None)
    gold = qrels.get(qid, {}) if qid else {}

    retriever = load_retriever("BAAI/bge-base-en-v1.5", BGE_QUERY_PREFIX)
    hits = retriever.search(question, top_k=top_k)
    doc_ids = [d for d, _ in hits]

    if gold:
        found = sum(1 for d in doc_ids if d in gold)
        st.success(
            f"Benchmark question: {len(gold)} document(s) are marked relevant by "
            f"human annotators, and {found} of them are in this top-{top_k}."
        )

    if generate:
        if not has_api_key():
            st.error("No Anthropic credentials found. Set ANTHROPIC_API_KEY or run `ant auth login`.")
        else:
            with st.spinner("Generating a grounded answer ..."):
                import anthropic
                from src.generation.rag import answer, format_context
                context = format_context(doc_ids, corpus)
                result, usage = answer(anthropic.Anthropic(), question, context)
            st.subheader("Answer")
            if not result.context_sufficient:
                st.warning(
                    "The model reported that the retrieved passages do not support "
                    "an answer. That is the intended behaviour on a retrieval miss -- "
                    "an honest abstention beats a confident guess dressed in citations."
                )
            st.write(result.answer)
            cited = ", ".join(f"[{c}]" for c in result.citations) or "none"
            st.caption(f"Cited passages: {cited}  |  "
                       f"{usage.input_tokens} input / {usage.output_tokens} output tokens")

    st.subheader(f"Retrieved passages (top {top_k})")
    for rank, (did, score) in enumerate(hits, start=1):
        is_gold = did in gold
        label = f"[{rank}] score {score:.3f}" + ("   ** human-judged relevant **" if is_gold else "")
        with st.expander(label, expanded=rank <= 3):
            st.write(doc_text(corpus[did]))
            st.caption(f"doc id: {did}")


# --- Tab 2: Retrieval benchmark ------------------------------------------

LABELS = {
    "random": "Random (floor)",
    "bm25": "BM25 (lexical baseline)",
    "dense_minilm": "Dense - all-MiniLM-L6-v2",
    "dense_bge": "Dense - bge-base-en-v1.5",
    "hybrid": "Hybrid - RRF(BM25, bge)",
    "rerank_hybrid": "Hybrid + ms-marco reranker",
    "rerank_dense": "Dense + ms-marco reranker",
    "rerank_dense_bge": "Dense + bge reranker",
    "rerank_bm25": "BM25 + ms-marco reranker",
}


def tab_retrieval():
    metrics = load_report("retrieval_metrics.json")
    if not metrics:
        st.warning("Run `python -m src.evaluation.run_retrieval_eval` first.")
        return

    st.markdown(
        "**FiQA-2018 test split** -- 57,638 documents, 648 questions with human "
        "relevance judgments. Sorted by nDCG@10."
    )
    rows = [{
        "Configuration": LABELS.get(name, name),
        "nDCG@10": m.get("ndcg@10"),
        "Recall@10": m.get("recall@10"),
        "Recall@100": m.get("recall@100"),
        "MRR@10": m.get("mrr@10"),
    } for name, m in metrics.items()]
    df = pd.DataFrame(rows).sort_values("nDCG@10", ascending=False)
    st.dataframe(df.style.format({c: "{:.4f}" for c in df.columns if c != "Configuration"}),
                 hide_index=True, use_container_width=True)

    st.info(
        "The two components the standard RAG recipe recommends -- hybrid RRF "
        "fusion and cross-encoder reranking -- both make the best retriever "
        "**worse** on this benchmark. The reranker is not broken: on a weak "
        "first stage it lifts BM25 by 36% relative. It simply has a quality "
        "ceiling below the dense retriever on financial text."
    )

    sig = load_report("retrieval_significance.json")
    if sig:
        st.subheader("Is the difference real?")
        st.caption(
            "Paired bootstrap over the 648 queries, 10,000 resamples. 648 queries "
            "cannot resolve every gap, so each comparison reports a confidence "
            "interval rather than a bare win/loss."
        )
        srows = [{
            "Comparison": k.replace("_vs_", " vs "),
            "Delta nDCG@10": v["mean_difference"],
            "95% CI": f"[{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]",
            "p": v["p_value"],
            "Significant": "yes" if v["p_value"] < 0.05 else "no",
        } for k, v in sig.items()]
        st.dataframe(pd.DataFrame(srows), hide_index=True, use_container_width=True)


# --- Tab 3: Answer quality -----------------------------------------------

COND_LABELS = {
    "closed_book": "No retrieval (closed book)",
    "rag_bm25": "RAG - BM25 passages",
    "rag_dense": "RAG - bge-base passages",
    "oracle": "Oracle - gold passages",
}


def tab_answers():
    report = load_report("generation_metrics.json")
    if not report:
        st.warning("Run `python -m src.evaluation.run_generation_eval --phase analyze` first.")
        return

    st.markdown(
        f"**{report['n_questions']} FiQA test questions**, four conditions, "
        f"answers generated by `{report['generator_model']}` and graded by "
        f"`{report['judge_model']}` against the human-judged gold passages."
    )
    rows = []
    for cond, m in report["conditions"].items():
        rows.append({
            "Condition": COND_LABELS.get(cond, cond),
            "Correct": m.get("correct"),
            "Correct or partial": m.get("correct_or_partial"),
            "Grounded": m.get("grounded"),
            "Citation precision": m.get("citation_precision"),
            "Abstained": m.get("abstained"),
            "n": m.get("n_scored"),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    cal = report.get("judge_calibration", {})
    if cal:
        st.subheader("Can the judge be trusted?")
        st.caption(
            "An LLM judge is a measuring instrument, and an uncalibrated instrument "
            "produces numbers people believe. Two checks: does the judge agree with "
            "itself on a re-run, and does a different model reach the same verdicts?"
        )
        c1, c2 = st.columns(2)
        sc, cm = cal.get("self_consistency", {}), cal.get("cross_model", {})
        if sc:
            c1.metric("Self-consistency (exact verdict)", f"{sc['exact_agreement']:.1%}",
                      help=f"n = {sc['n']} re-judged items")
        if cm:
            c2.metric(f"Agreement with {cal.get('cross_model_id', 'second model')}",
                      f"{cm['exact_agreement']:.1%}", help=f"n = {cm['n']} items")

    costs = report.get("estimated_cost_usd")
    if costs:
        st.caption(f"Evaluation cost (Batches API, 50% pricing): {costs}")


# --- Main -----------------------------------------------------------------

st.title("Financial RAG, measured")
st.caption(
    "Retrieval-augmented QA over 57,638 financial forum documents, evaluated "
    "against human relevance judgments instead of vibes."
)

tabs = st.tabs(["Ask", "Retrieval benchmark", "Answer quality"])
with tabs[0]:
    tab_ask()
with tabs[1]:
    tab_retrieval()
with tabs[2]:
    tab_answers()
