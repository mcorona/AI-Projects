# AI Projects — Manuel Corona

End-to-end machine learning projects, each taken from raw data to a deployed,
documented, honestly-evaluated system. Every project here includes baselines it
has to beat, a held-out test set it was not tuned on, and a model card stating
what it can't do.

The through-line is **evaluation discipline**: in both projects the interesting
result came from testing properly, not from picking a fancier model.

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

## What each project ships

|  | Project 1 | Project 2 |
|---|---|---|
| Honest baselines | ✅ 3 | ✅ 2 |
| Held-out test evaluation | ✅ | ✅ |
| Error / regime analysis | ✅ | ✅ |
| Model card with limitations | ✅ | ✅ |
| Serving layer | FastAPI + Streamlit | Streamlit dashboard |
| Docker | ✅ | ✅ |

---

## Running them

Each project is self-contained with its own `README.md`, `requirements.txt`, and
setup steps — including how to fetch the dataset, which is gitignored in both.

```bash
cd project-01-customer-feedback-nlp   # or project-02-timeseries-forecasting
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

See each project's README for dataset download, training, and how to launch the
API / dashboard (locally or via Docker).

---

## Author

**Manuel Corona**
