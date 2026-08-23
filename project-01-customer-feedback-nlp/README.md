# Financial Sentiment Classifier - Production-Grade NLP Pipeline

## 📊 Project Overview

A fine-tuned RoBERTa model for 3-class sentiment classification (negative / neutral / positive) of short financial and business text, built end-to-end following enterprise ML practices: baseline comparison, rigorous held-out evaluation, error analysis, and a deployed API + demo.

**What's actually in this repo:**
- Fine-tuned RoBERTa-base classifier, beating the best traditional-ML baseline by +9.6 points of weighted F1
- 3 baseline models (rule-based, traditional ML, generic pre-trained transformer) for honest comparison
- Full evaluation suite: accuracy, weighted/macro F1, AUROC, per-class metrics, confusion matrix
- Deep error analysis: confused-class pairs, error rate by text length, confidence calibration
- FastAPI inference server + Streamlit demo app, both Dockerized
- A model card documenting scope, limitations, and known failure modes (`MODEL_CARD.md`)

**Note on scope:** this project was originally scoped as a general "customer feedback" system (reviews, surveys, support tickets). It was built and validated against the [Financial PhraseBank](https://arxiv.org/abs/1307.5336) dataset instead (see `PHASE_1_SUMMARY.md` for why) -- the pipeline, code, and practices generalize directly to other feedback domains, but the current trained model and dataset are financial-text-specific. See `MODEL_CARD.md` for exact scope and limitations.

---

## 📈 Results

Evaluated on a 728-sample held-out test split (`src/models/evaluate.py`):

| Model | Accuracy | F1 (Weighted) |
|-------|----------|----------------|
| TextBlob (rule-based) | 0.562 | 0.547 |
| TF-IDF + Linear SVM | 0.755 | 0.750 |
| DistilBERT (SST-2, no fine-tuning) | 0.310 | 0.217 |
| **RoBERTa (fine-tuned, this project)** | **0.846** | **0.846** |

Per-class breakdown for the fine-tuned model:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| negative | 0.76 | 0.91 | 0.83 |
| neutral | 0.88 | 0.87 | 0.88 |
| positive | 0.81 | 0.77 | 0.79 |

Full methodology, baseline details, and error analysis: `PHASE_1_SUMMARY.md` and `PHASE_3_SUMMARY.md`.

---

## 📁 Project Structure

```
project-01-customer-feedback-nlp/
├── data/
│   ├── raw/                        # Raw dataset (gitignored)
│   └── processed/                  # train/val/test parquet splits (gitignored)
├── notebooks/
│   ├── 00_baselines.ipynb          # Baseline training & evaluation
│   └── 01_eda_feedback_data.ipynb  # Exploratory analysis
├── src/
│   ├── api.py                      # FastAPI inference server
│   ├── baselines.py                # TextBlob / TF-IDF+SVM / DistilBERT baselines
│   ├── data/
│   │   ├── loader.py                # Dataset loading & stratified splitting
│   │   └── preprocessing.py         # Text cleaning & tokenized PyTorch Dataset
│   ├── models/
│   │   ├── sentiment_classifier.py  # FinancialSentimentClassifier (PyTorch Lightning)
│   │   ├── train.py                 # Fine-tuning entrypoint
│   │   └── evaluate.py              # Full evaluation on the test set
│   └── evaluation/
│       ├── baseline_evaluation.py   # Shared metrics suite
│       ├── bias_analysis.py         # Text-length subgroup analysis
│       └── error_analysis.py        # Deep error / calibration analysis
├── configs/
│   └── training_config.yaml         # Hyperparameters
├── output/
│   ├── models/                      # Trained checkpoints (gitignored)
│   ├── reports/                     # JSON metrics & CSV predictions
│   └── figures/                     # Plots (confusion matrices, calibration, etc.)
├── app.py                           # Streamlit demo
├── Dockerfile / .dockerignore       # Container build for the API
├── requirements.txt                 # Full dev environment
├── requirements-api.txt             # Minimal deps for the Docker API image
├── MODEL_CARD.md                    # Model scope, evaluation, limitations
├── PHASE_1_SUMMARY.md               # EDA, baselines, preliminary bias analysis
├── PHASE_3_SUMMARY.md               # Fine-tuning results & error analysis
└── setup.py
```

---

## 🎯 Core Objectives

1. **Domain-specific fine-tuning** that measurably beats general-purpose and traditional-ML baselines on the same held-out test set. ✅ Achieved: +9.6 F1 points over the best baseline.
2. **Rigorous evaluation**: stratified splits, weighted/macro/per-class metrics, AUROC, confusion matrices -- not just accuracy. ✅
3. **Honest error analysis**: which classes get confused, whether errors correlate with text length, and whether the model's stated confidence is trustworthy (calibration). ✅ See `PHASE_3_SUMMARY.md`.
4. **Production readiness**: a served API with proper error handling, a usable demo, and a containerized deployment path. ✅

---

## 🛠️ Tech Stack

| Component | Tool | Reason |
|-----------|------|--------|
| **Model** | HuggingFace Transformers (RoBERTa-base) | Pre-trained transformer, fine-tuned for the task |
| **Training** | PyTorch + PyTorch Lightning | Structured, reproducible fine-tuning loop |
| **Data Processing** | Pandas, PyArrow (parquet) | Loading, splitting, tokenized dataset construction |
| **Evaluation** | Scikit-learn | Metrics suite, confusion matrices |
| **Baselines** | TextBlob, Scikit-learn (TF-IDF+SVM), Transformers (DistilBERT) | Honest comparison points |
| **Visualization** | Matplotlib, Seaborn, WordCloud | EDA and evaluation figures |
| **API** | FastAPI + Uvicorn | Inference server |
| **Demo** | Streamlit | Interactive single/batch prediction UI |
| **Deployment** | Docker | Containerized API |
| **Testing** | Pytest | Available for unit tests (not yet populated) |

---

## 📊 Dataset

**[Financial PhraseBank](https://arxiv.org/abs/1307.5336)** (Malo et al., 2014), `sentences_allagree` variant (100% annotator agreement):
- 4,840 unique English sentences (after removing 6 exact duplicates) from Finnish public-company press releases/financial news, circa 2009-2013
- 3-class labels: negative (12.5%), neutral (59.3%), positive (28.1%) -- see `PHASE_1_SUMMARY.md` for the full EDA
- Stratified 70/15/15 split (train 3,392 / val 726 / test 728)

Known limitations of this dataset (single domain/era/region, low sentence-length diversity, no sector metadata for subgroup fairness analysis) are documented in `MODEL_CARD.md` and `PHASE_1_SUMMARY.md`.

---

## ✅ Development Status

### Phase 1: Data & EDA -- Complete
EDA notebook, 3 trained baselines, preliminary bias analysis. See `PHASE_1_SUMMARY.md`.

### Phase 2: Fine-tuning -- Complete
RoBERTa fine-tuned (top 3 of 12 layers unfrozen), best checkpoint at epoch 3 (val F1 = 0.867). See `src/models/train.py`.

### Phase 3: Evaluation & Error Analysis -- Complete
Full test-set evaluation (F1 = 0.846) and deep error analysis (confused pairs, length/confidence subgroups, calibration curve). See `PHASE_3_SUMMARY.md`.

### Phase 4: Deployment -- Complete
FastAPI server, Streamlit demo, Dockerized API, model card.

---

## 🔧 Getting Started

```bash
# 1. Navigate to the project
cd project-01-customer-feedback-nlp

# 2. Set up the environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Get the dataset and build splits (see PHASE_1_SUMMARY.md for the source)
mkdir -p data/raw/financial_phrasebank
curl -L https://raw.githubusercontent.com/isaaccs/sentiment-analysis-for-financial-news/master/all-data.csv \
  -o data/raw/financial_phrasebank/all-data.csv
python -m src.data.loader

# 4. Explore the data / baselines
jupyter notebook notebooks/01_eda_feedback_data.ipynb
jupyter notebook notebooks/00_baselines.ipynb

# 5. Fine-tune the model
python -m src.models.train --config configs/training_config.yaml

# 6. Evaluate on the held-out test set
python -m src.models.evaluate --model-path output/models/best-<epoch>-<f1>.ckpt
python -m src.evaluation.error_analysis

# 7. Serve it
uvicorn src.api:app --reload          # API, http://localhost:8000/docs
streamlit run app.py                  # Interactive demo

# 8. Or run the API in Docker
docker build -t financial-sentiment .
docker run -p 8000:8000 -v "$(pwd)/output/models:/app/output/models" financial-sentiment
```

---

## 📚 References

- **RoBERTa:** [Liu et al., 2019](https://arxiv.org/abs/1907.11692)
- **Financial PhraseBank:** [Malo et al., 2014](https://arxiv.org/abs/1307.5336)
- **HuggingFace Transformers:** [Documentation](https://huggingface.co/docs/transformers/)

---

## 📞 Author

Manuel Corona

*Last Updated: 2026-08-23*
