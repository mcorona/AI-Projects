"""
Streamlit demo for the fine-tuned FinancialSentimentClassifier.

Run with:
    streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.api import resolve_checkpoint_path
from src.models.sentiment_classifier import SentimentInference

st.set_page_config(page_title="Financial Sentiment Analyzer", page_icon="\U0001F4B0")
st.title("\U0001F4B0 Financial Sentiment Analyzer")
st.caption("RoBERTa fine-tuned on the Financial PhraseBank dataset (3-class: negative / neutral / positive)")


@st.cache_resource
def load_model() -> SentimentInference:
    ckpt_path = resolve_checkpoint_path()
    if ckpt_path is None:
        return None
    return SentimentInference(ckpt_path)


def show_probabilities(probabilities: dict):
    df = pd.DataFrame(
        {"probability": [probabilities[c] for c in ["negative", "neutral", "positive"]]},
        index=["negative", "neutral", "positive"],
    )
    st.bar_chart(df)


model = load_model()

if model is None:
    st.error(
        "No trained checkpoint found. Set the MODEL_CKPT_PATH environment variable, "
        "or place one at output/models/best-*.ckpt (e.g. by running `python -m src.models.train`)."
    )
    st.stop()

tab_single, tab_batch, tab_metrics = st.tabs(["Single Text", "Batch (CSV)", "Model Performance"])

with tab_single:
    text = st.text_area("Enter financial text to analyze", height=100, placeholder="e.g. Sales increased by 20% this quarter.")
    if st.button("Analyze", type="primary") and text.strip():
        detail = model.predict_batch_with_details([text])[0]
        sentiment = detail["prediction"].lower()
        probabilities = {k.lower(): v for k, v in detail["probabilities"].items()}

        emoji = {"positive": "\U0001F7E2", "neutral": "⚪", "negative": "\U0001F534"}[sentiment]
        st.subheader(f"{emoji} {sentiment.capitalize()} ({detail['confidence']*100:.1f}% confidence)")
        show_probabilities(probabilities)

with tab_batch:
    st.write("Upload a CSV with a `text` column to classify multiple rows at once.")
    uploaded = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        if "text" not in df.columns:
            st.error("CSV must have a column named 'text'.")
        else:
            details = model.predict_batch_with_details(df["text"].tolist())
            df["sentiment"] = [d["prediction"].lower() for d in details]
            df["confidence"] = [d["confidence"] for d in details]
            st.dataframe(df)
            st.download_button(
                "Download results as CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="sentiment_results.csv",
                mime="text/csv",
            )

with tab_metrics:
    results_path = Path("output/reports/roberta_eval_results.json")
    if not results_path.exists():
        st.info(
            "No evaluation report found. Run `python -m src.models.evaluate --model-path <checkpoint>` "
            "to generate output/reports/roberta_eval_results.json."
        )
    else:
        with open(results_path) as f:
            metrics = json.load(f)
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        col2.metric("F1 (weighted)", f"{metrics['f1_weighted']:.3f}")
        col3.metric("F1 (macro)", f"{metrics.get('f1_macro', 0):.3f}")

        if "f1_per_class" in metrics:
            st.write("**Per-class F1:**")
            st.dataframe(pd.DataFrame([metrics["f1_per_class"]]).T.rename(columns={0: "f1"}))
