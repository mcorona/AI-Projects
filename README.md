# AI Projects — Manuel Corona

End-to-end machine learning projects, each taken from raw data to a deployed,
documented, honestly-evaluated system. Every project here includes baselines it
has to beat, a held-out test set it was not tuned on, and a model card stating
what it can't do.

The through-line is **evaluation discipline**: in all three projects the
interesting result came from testing properly, not from picking a fancier
model. In two of them the headline finding is that the sophisticated
approach *lost* — a naive baseline matched ARIMA and an LSTM, and a full RAG
pipeline answered fewer questions correctly than the same model with no
retrieval. Knowing that is worth more than a demo that looks good.

---

## Projects

### 1. [Financial Sentiment Classifier](project-01-customer-feedback-nlp/) — NLP

Fine-tuned RoBERTa-base for 3-class sentiment (negative / neutral / positive)
on short financial text, benchmarked against three honest baselines.

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| TextBlob (rule-based) | 0.562 | 0.547 |
| TF-IDF + Linear SVM | 0.755 | 0.750 |
| DistilBERT (SST-2, no fine-tuning) | 0.310 | 0.217 |
| **RoBERTa (fine-tuned)** | **0.846** | **0.846** |

**Result:** +9.6 F1 points over the best traditional-ML baseline on a 728-sample
held-out test split. The off-the-shelf transformer scored *worse than the
rule-based baseline* — general-purpose sentiment models transfer badly to
financial text, which is the case for fine-tuning in the first place.

Also includes deep error analysis (confused-class pairs, error rate by text
length, confidence calibration), a FastAPI inference server, and a Streamlit
demo — both Dockerized.

**Stack:** PyTorch · PyTorch Lightning · Hugging Face Transformers · scikit-learn · FastAPI · Streamlit · Docker

---

### 2. [Stock Price Forecasting](project-02-timeseries-forecasting/) — Time Series

Three fundamentally different forecasting approaches compared on the same series:
5,306 trading days of Reliance Industries (NSE), 2000–2021. Chronological splits
only — no shuffling, which would leak the future into training.

| Model | MAPE |
|---|---|
| Persistence (naive) | 1.48% |
| **ARIMA(2,1,5)** | **1.48%** |
| **LSTM (log returns)** | **1.48%** |
| Seasonal-naive | 3.48% |
| Prophet (5-day rolling) | 10.67% |

**Result:** for one-day-ahead forecasting of a liquid stock, model sophistication
was not the bottleneck — the naive baseline matched ARIMA and a correctly-specified
LSTM, including through the March 2020 COVID crash. Reporting that is more useful
than hiding it behind a model that appears to win.

The test set also caught a genuine bug: an LSTM trained on raw price level
collapsed to ~20% MAPE because the price nearly tripled beyond its scaler's fitted range,
despite looking healthy in validation (2.85%). Retraining on log returns
(scale-invariant) fixed it. Writeup in
[`MODEL_CARD.md`](project-02-timeseries-forecasting/MODEL_CARD.md).

Includes an interactive Streamlit + Plotly dashboard (forecast explorer, model
comparison, COVID-crash case study), Dockerized.

**Stack:** Statsmodels · Prophet · TensorFlow/Keras · pandas · scikit-learn · Streamlit · Plotly · Docker

---

### 3. [Financial RAG, Measured](project-03-financial-rag/) — Retrieval / LLM

Retrieval-augmented QA over 57,638 financial forum documents, built on a
benchmark with human relevance judgments (BEIR FiQA-2018) so every claim is
measured against ground truth instead of eyeballed.

**Retrieval** — nine configurations, 648 human-judged test queries:

| Configuration | nDCG@10 |
|---|---|
| BM25 (lexical baseline) | 0.2374 |
| Hybrid — RRF(BM25, bge) | 0.3588 |
| Dense + cross-encoder reranker | 0.3851 |
| **Dense — bge-base-en-v1.5** | **0.4062** |

**Answer quality** — 150 questions, four conditions, `claude-opus-5`:

| Condition | Correct | Abstained |
|---|---|---|
| **No retrieval (closed book)** | **0.820** | 0.000 |
| RAG — BM25 passages | 0.487 | 0.407 |
| RAG — bge-base passages | 0.713 | 0.193 |
| Oracle — gold passages | 0.927 | 0.153 |

**Two results, both negative, both the point.** First: hybrid fusion and
cross-encoder reranking — the two components the standard RAG recipe
recommends — each make the best retriever *worse* (p < 0.0001 and p = 0.015).
The reranker isn't broken; it lifts BM25 by 36% relative. It just has a
quality ceiling below the dense retriever on financial text.

Second: **the full RAG pipeline answers fewer questions correctly than the
same model with no retrieval at all** — 71.3% vs 82.0%, exact McNemar
p = 0.020. But restricted to questions it didn't abstain on, RAG is *more*
accurate (86.8%), and given gold passages it's correct every time it speaks
(100%). It abstains 7× more often when retrieval actually failed. The trade
is raw answer rate for attributability — a bad deal on a domain the model
already knows, the whole point for proprietary or post-cutoff documents.

Also includes an implementation validated against published BEIR numbers to
within 0.002 nDCG, paired-bootstrap significance tests, judge self-consistency
calibration (96.6%), and a Streamlit dashboard — Dockerized. The whole
evaluation cost $13.68 through the Batches API.

**Stack:** sentence-transformers · rank-bm25 · cross-encoders · Anthropic Claude (Batches API, structured outputs) · NumPy · Streamlit · Docker

---

## What each project ships

|  | Project 1 | Project 2 | Project 3 |
|---|---|---|---|
| Honest baselines | ✅ 3 | ✅ 2 | ✅ 4 |
| Held-out test evaluation | ✅ | ✅ | ✅ |
| Error / regime analysis | ✅ | ✅ | ✅ |
| Statistical significance testing | — | — | ✅ |
| Validated against published results | — | — | ✅ |
| Model card with limitations | ✅ | ✅ | ✅ |
| Serving layer | FastAPI + Streamlit | Streamlit dashboard | Streamlit dashboard |
| Docker | ✅ | ✅ | ✅ |

---

## Running them

Each project is self-contained with its own `README.md`, `requirements.txt`, and
setup steps — including how to fetch the dataset, which is gitignored in all
three (Project 3's downloads itself on first run).

```bash
cd project-01-customer-feedback-nlp   # or project-02-... / project-03-...
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

See each project's README for dataset download, training, and how to launch the
API / dashboard (locally or via Docker).

---

## Author

**Manuel Corona**
