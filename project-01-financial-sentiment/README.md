# Financial Sentiment Analysis with Fine-tuned Transformers

## 📊 Project Overview

An enterprise-grade sentiment analysis system for financial texts (earnings calls, news articles, investor reports). Designed to detect nuanced emotional tones critical for quantitative finance and investment decisions.

**Key Capabilities:**
- Fine-tuned RoBERTa model for financial domain
- Multi-class emotion detection (Positive, Neutral, Negative, Mixed)
- Bias analysis and fairness metrics
- Cross-dataset evaluation (domain shift analysis)
- Production-ready inference API with confidence scores
- Comprehensive error analysis and interpretability

**Target Audience:** Quant funds, fintech companies, risk management teams

---

## 📁 Project Structure

```
project-01-financial-sentiment/
├── data/
│   ├── raw/                    # Original datasets
│   │   ├── financial_phrasebank/
│   │   ├── sentiment_140/
│   │   └── earnings_call_data/
│   └── processed/              # Cleaned, tokenized data
│       ├── train.parquet
│       ├── val.parquet
│       └── test.parquet
├── notebooks/
│   ├── 01_eda_financial_data.ipynb      # Exploratory analysis
│   ├── 02_dataset_analysis_bias.ipynb   # Bias detection
│   ├── 03_model_training.ipynb          # Training pipeline
│   └── 04_evaluation_ablation.ipynb     # Results & ablation studies
├── src/
│   ├── models/
│   │   ├── sentiment_classifier.py      # Main model class
│   │   └── pretrained_wrapper.py        # HF Transformers wrapper
│   ├── data/
│   │   ├── loader.py                    # Dataset loading & preprocessing
│   │   ├── augmentation.py              # Data augmentation techniques
│   │   └── bias_analyzer.py             # Fairness metrics
│   └── utils/
│       ├── metrics.py                   # Custom evaluation metrics
│       ├── inference.py                 # Batch inference
│       └── logging.py                   # Experiment tracking (MLflow)
├── tests/
│   ├── test_data_loader.py
│   ├── test_model.py
│   └── test_inference.py
├── configs/
│   ├── training_config.yaml             # Hyperparameters
│   └── model_config.yaml                # Model architecture
├── output/
│   ├── models/                          # Trained model checkpoints
│   ├── reports/                         # Analysis reports (PDF/HTML)
│   └── figures/                         # Plots & visualizations
├── requirements.txt
├── .gitignore
├── setup.py
└── Makefile                             # Common commands
```

---

## 🎯 Core Objectives

### Objective 1: Domain-Specific Model
- Fine-tune RoBERTa on financial texts
- Outperform general-purpose sentiment models
- Handle financial jargon and context

### Objective 2: Rigorous Evaluation
- Cross-validation with temporal splits (time-series bias prevention)
- Multiple metrics: Accuracy, F1 (weighted), Precision, Recall, AUROC
- Comparison vs. baselines (TextBlob, DistilBERT, VADER)

### Objective 3: Bias & Fairness Analysis
- Subgroup analysis (by company sector, news source, time period)
- Disparate impact analysis
- Error rate parity across demographic groups

### Objective 4: Error Analysis & Interpretability
- Confusion matrix breakdown
- Hard case analysis (edge cases)
- SHAP/attention visualization for top predictions
- Confidence calibration analysis

### Objective 5: Production Readiness
- Reproducible training pipeline
- FastAPI inference server
- Monitoring & drift detection guidelines
- Comprehensive documentation

---

## 🛠️ Tech Stack

| Component | Tool | Reason |
|-----------|------|--------|
| **Models** | HuggingFace Transformers | Pre-trained, fine-tuned models |
| **Training** | PyTorch + Pytorch-Lightning | Scalable, reproducible training |
| **Data Processing** | Pandas, Polars | Efficient data manipulation |
| **Evaluation** | Scikit-learn, custom metrics | Comprehensive metrics suite |
| **Bias Analysis** | Fairness Indicators, Themis ML | Fairness & bias detection |
| **Visualization** | Plotly, Matplotlib, Seaborn | Interactive & static plots |
| **API** | FastAPI | High-performance inference server |
| **Frontend** | Streamlit | Interactive demo app |
| **Experiment Tracking** | MLflow | Model versioning & comparison |
| **Testing** | Pytest | Code quality & reliability |

---

## 📊 Datasets

### Primary: FinancialPhraseBank
- **Size:** 4,840 sentences from financial news
- **Labels:** Positive, Negative, Neutral (3-class)
- **Source:** Malo et al. (2014)
- **Preprocessing:** Lowercase, special char handling, stop words

### Secondary: SentiCorp (Earnings Calls)
- **Size:** 2,000+ earnings call sentences
- **Labels:** Expert-annotated sentiment
- **Relevance:** Domain-specific financial language
- **Split:** Train/Val/Test (70/15/15)

### Evaluation: Financial Phrase Bank test set
- **Size:** 1,210 sentences
- **Purpose:** Out-of-domain generalization testing
- **Metrics:** Cross-dataset performance, domain shift analysis

---

## 🚀 Development Roadmap

### Phase 1: Data & EDA (Weeks 1-2)
- [ ] Download & explore datasets
- [ ] Conduct bias analysis (class imbalance, demographic representation)
- [ ] Create balanced train/val/test splits
- [ ] Document data quality issues

### Phase 2: Baseline & Fine-tuning (Weeks 2-4)
- [ ] Implement baseline classifiers (TextBlob, DistilBERT)
- [ ] Fine-tune RoBERTa on financial domain
- [ ] Hyperparameter tuning with validation set
- [ ] Training curve analysis

### Phase 3: Evaluation & Analysis (Weeks 4-5)
- [ ] Comprehensive evaluation (metrics, confusion matrix)
- [ ] Error analysis & case studies
- [ ] Ablation studies (different pooling, layer freezing)
- [ ] Bias & fairness metrics

### Phase 4: Deployment & Documentation (Weeks 5-6)
- [ ] FastAPI inference server
- [ ] Streamlit demo application
- [ ] Model card & technical documentation
- [ ] Reproducibility guide (Docker, requirements)

---

## 📈 Key Metrics to Track

| Metric | Threshold | Why Important |
|--------|-----------|---------------|
| **Weighted F1** | >0.85 | Handles class imbalance |
| **Per-class Recall** | >0.80 | Minimize false negatives |
| **AUROC** | >0.90 | Confidence calibration |
| **Fairness Diff** | <5% | Equal performance across groups |
| **Inference Latency** | <500ms | Production requirement |
| **Model Size** | <500MB | Deployment constraint |

---

## 🔍 Evaluation Strategy

### Cross-Validation
- **Temporal Split:** Train on earlier dates, test on recent (no future leakage)
- **K-Fold (5-fold):** Measure variance across folds
- **Stratified:** Maintain class balance in each fold

### Baseline Comparisons
1. **Rule-based:** TextBlob, VADER
2. **Pre-trained:** DistilBERT (generic), FinBERT (financial)
3. **Our model:** Fine-tuned RoBERTa-financial

### Metrics Suite
```
- Accuracy, Precision, Recall, F1 (weighted & per-class)
- AUROC, PR-AUC
- Confusion matrix breakdown
- Error rate by subgroup (fairness)
- Confidence calibration (ECE)
```

---

## 📝 Deliverables

✅ **4 Jupyter Notebooks**
- EDA with visualizations
- Bias analysis report
- Training & results notebook
- Error analysis & interpretation

✅ **Production-Ready Code**
- Modular, type-hinted Python modules
- 80%+ test coverage
- Comprehensive docstrings

✅ **Trained Models**
- Fine-tuned RoBERTa checkpoint
- Baseline models for comparison

✅ **API & Demo**
- FastAPI server for inference
- Streamlit UI for exploration

✅ **Reports & Visualizations**
- Model comparison plots
- Confusion matrices & error analysis
- Bias report (PDF)

✅ **Documentation**
- Technical README (this file)
- Model card with limitations
- Reproducibility guide

---

## 🔧 Getting Started

```bash
# 1. Clone and navigate
cd project-01-financial-sentiment

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download datasets
python src/data/loader.py --download-all

# 4. Run exploratory analysis
jupyter notebook notebooks/01_eda_financial_data.ipynb

# 5. Train model
python -m src.models.train --config configs/training_config.yaml

# 6. Evaluate
python -m src.models.evaluate --model-path output/models/best.pt

# 7. Launch demo
streamlit run app.py
```

---

## 📚 References

- **FinancialPhraseBank:** [Malo et al., 2014](https://www.researchgate.net/publication/260063683_FinancialPhraseBank-a_new_corpus_for_phrase_level_sentiment_analysis)
- **RoBERTa:** [Liu et al., 2019](https://arxiv.org/abs/1907.11692)
- **Fine-tuning Guide:** [HuggingFace Documentation](https://huggingface.co/docs/transformers/)
- **Fairness in ML:** [Fairness Indicators](https://github.com/tensorflow/fairness-indicators)

---

## 📞 Author

Manuel Corona | [LinkedIn](#) | [GitHub](#)

*Last Updated: 2026-08-22*
