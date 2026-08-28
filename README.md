# AI Projects — Manuel Corona

End-to-end machine learning projects, each taken from raw data to a deployed,
documented, honestly-evaluated system. Every project here includes baselines it
has to beat, a held-out test set it was not tuned on, and a model card stating
what it can't do.

The through-line is **evaluation discipline**: in all five projects the
interesting result came from testing properly, not from picking a fancier
model. In three of them the headline finding is that the sophisticated
approach *lost* — a naive baseline matched ARIMA and an LSTM, a full RAG
pipeline answered fewer questions correctly than the same model with no
retrieval, and fine-tuning a ResNet-50 produced a significantly worse
classifier than a logistic regression on its frozen features. In the fifth,
the metric itself was the problem: a credit model with a respectable AUC
turns out to be worth exactly nothing once its decisions are priced.
Knowing that is worth more than a demo that looks good.

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

### 4. [Pet Breeds, and What ImageNet Already Knew](project-04-pet-breeds/) — Computer Vision

Fine-grained classification of 37 cat and dog breeds (Oxford-IIIT Pet),
built to measure the thing most transfer-learning writeups leave
unexamined: **how much of the task the pretrained model had already been
trained on.**

ImageNet-1k spends ~120 of its 1,000 classes on dog breeds and 5 on
domestic cats. It names **24 of these 37 breeds — 21 of 25 dogs, but only 3
of 12 cats**, covering 64.9% of the test split. Every metric is therefore
reported split on that boundary.

| Model | Accuracy | In ImageNet | Not in ImageNet |
|---|---|---|---|
| Majority class (floor) | 0.0267 | 0.0000 | 0.0762 |
| HOG + colour → linear SVM | 0.0905 | 0.0961 | 0.0801 |
| ImageNet classifier, **zero-shot** | 0.6397 | **0.9849** | **0.0000** |
| **Linear probe on frozen features** | **0.9283** | 0.9610 | 0.8678 |
| Fine-tuned ResNet-50 | 0.9120 | 0.9425 | 0.8554 |

**Two findings.** First: a stock ImageNet classifier, with **zero gradient
steps on this dataset**, is already **98.5%** accurate on the breeds ImageNet
names. "Transfer learning reaches 90%+ on Oxford Pets" is true and
misleading — two thirds of the benchmark was an ImageNet task before anyone
started. (Hand-designed features get 9.1%, so the task itself is genuinely
hard.)

Second: **fine-tuning made the model significantly worse** — 0.9120 vs the
linear probe's 0.9283, paired McNemar p = 1.9 × 10⁻⁴ — and the damage lands
precisely on the breeds ImageNet already knew (p = 4.9 × 10⁻⁶), not on the
ones it didn't (p = 0.24, n.s.). Catastrophic forgetting in miniature.
Validation would not have caught it: the fine-tune *won* on validation
(0.9528) and lost on test. The probe is also 10× better calibrated
(ECE 0.0118 vs 0.1023), so it is the model the app actually serves.

**Stack:** PyTorch · torchvision (ResNet-50) · scikit-learn · scikit-image · Streamlit · Docker

---

### 5. [Credit Decisions, Priced](project-05-credit-risk/) — Tabular / Decision Analysis

Default prediction on 30,000 credit-card accounts (UCI, Taiwan 2005), built
to measure what sits between a model and a business: **the threshold, and
what the two mistakes actually cost.** Declining a customer who would have
paid costs the margin; approving one who defaults costs the balance. A 0.5
cutoff is the buried claim that those are the same number.

| Model | AUC | Cost | Cost at 0.5 | Saved vs no model |
|---|---|---|---|---|
| Delinquency rule (one column) | 0.7205 | 5841 | 8435 | **0** |
| Logistic regression | 0.7377 | 5637 | 9598 | +204 |
| Logistic, `class_weight="balanced"` | 0.7371 | 5663 | 5832 | +179 |
| Random forest | 0.7926 | 4695 | 8061 | +1146 |
| **Gradient boosting** | **0.7881** | **4713** | 7933 | **+1129** |

**Three results.** First: **a model with AUC 0.72 is worth exactly nothing.**
The one-column delinquency rule ranks respectably, but at this cost ratio its
cheapest policy is to decline every applicant — which needs no model. It and
the AUC-0.50 floor are worth identically zero, 0.22 of AUC apart.

Second: **the threshold matters more than the model.** Deployed at 0.5, the
best model costs 68% more than the same model at its own threshold, and loses
to lending to nobody. Picking the model bought 1129; picking the threshold
badly gives back 3220. And that holds across every plausible cost ratio — the
0.5 penalty is +34% at R=5 and +261% at R=20, and vanishes only at R=1, which
is the assumption 0.5 encodes.

Third: **`class_weight="balanced"` does nothing and looks like it does a
lot.** Against the plain logistic regression every difference is null (AUC
−0.0007, regret +0.0034, McNemar p=0.38) — but compared at a 0.5 cutoff it
appears 39% better, purely because the weighting relocated the probabilities.
It bought that appearance with a calibration error four times larger.

The fairness audit prices two interventions instead of assuming them:
dropping the four protected attributes costs ~5% of the model's value and
closes only 19% of the education gap (payment history proxies for it), while
per-group thresholds cost 0.1–3% and close 98%. The reflex fix is the
expensive, ineffective one — and the cheap one is mostly illegal, which is
the trade rather than the answer.

Also includes an exposure-weighted capacity policy (ranking by expected loss
prevents 3.3× the money at 1% review capacity while catching *fewer*
defaults), paired-bootstrap and exact-McNemar tests, and a Streamlit
dashboard where the cost ratio is a slider — Dockerized, and running from
the committed held-out predictions with no dataset or model file.

**Stack:** scikit-learn · pandas · NumPy · SciPy · Streamlit · Plotly · Docker

---

## What each project ships

|  | Project 1 | Project 2 | Project 3 | Project 4 | Project 5 |
|---|---|---|---|---|---|
| Honest baselines | ✅ 3 | ✅ 2 | ✅ 4 | ✅ 3 | ✅ 2 |
| Held-out test evaluation | ✅ | ✅ | ✅ | ✅ | ✅ |
| Error / regime analysis | ✅ | ✅ | ✅ | ✅ | ✅ |
| Statistical significance testing | — | — | ✅ | ✅ | ✅ |
| Validated against published results | — | — | ✅ | — | — |
| Calibration analysis | — | — | — | ✅ | ✅ |
| Cost-sensitive decision analysis | — | — | — | — | ✅ |
| Subgroup / fairness audit | — | — | — | — | ✅ |
| Model card with limitations | ✅ | ✅ | ✅ | ✅ | ✅ |
| Serving layer | FastAPI + Streamlit | Streamlit dashboard | Streamlit dashboard | Streamlit dashboard | Streamlit dashboard |
| Docker | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## Running them

Each project is self-contained with its own `README.md`, `requirements.txt`, and
setup steps — including how to fetch the dataset, which is gitignored in all
five (Projects 3, 4 and 5 download theirs on first run).

```bash
cd project-01-customer-feedback-nlp   # or project-02-... through -05-...
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

See each project's README for dataset download, training, and how to launch the
API / dashboard (locally or via Docker).

---

## Author

**Manuel Corona**
