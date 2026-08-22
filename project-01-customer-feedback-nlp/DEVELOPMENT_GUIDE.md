# Development Guide: Financial Sentiment Analysis

## 📋 Overview

This guide walks you through developing the Financial Sentiment Analysis project step-by-step. You'll build production-ready code with guidance on best practices, testing, and deployment.

**Time Commitment:** ~3-4 weeks, 12 hours/week

---

## Phase 1: Setup & Exploration (Weeks 1-2)

### Step 1.1: Environment Setup

```bash
# 1. Navigate to project directory
cd project-01-financial-sentiment

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; import transformers; print('✓ All dependencies installed')"
```

### Step 1.2: Dataset Exploration

**Your Task:** Create a Python script or notebook to:
1. Download FinancialPhraseBank dataset (4,840 sentences)
   - Source: https://www.researchgate.net/publication/260063683_FinancialPhraseBank-a_new_corpus_for_phrase_level_sentiment_analysis
   - Alternative: Use Kaggle's Financial Phrase Bank dataset

2. Load and explore the data:
   ```python
   import pandas as pd
   
   # Load dataset
   df = pd.read_csv("data/raw/FinancialPhraseBank.csv")
   
   # Questions to answer:
   # - How many samples per class? (class distribution)
   # - What's the text length distribution?
   # - Any missing values?
   # - Sample texts from each class?
   ```

3. **Create file:** `src/data/loader.py`
   - Write functions to load and preprocess the data
   - Handle class imbalance (document this finding)
   - Create train/val/test splits (70/15/15)

**Checklist:**
- [ ] Dataset downloaded and explored
- [ ] Class distribution visualized (matplotlib/plotly)
- [ ] No NaN values or handle them appropriately
- [ ] Train/val/test splits created (stratified)
- [ ] Notebook saved with EDA findings

---

### Step 1.3: Baseline Evaluation

**Your Task:** Implement baseline models to establish performance benchmarks.

```python
# Create: src/baselines.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# TODO: Implement 3 baselines:
# 1. TextBlob (rule-based sentiment)
# 2. TF-IDF + SVM (traditional ML)
# 3. DistilBERT (pre-trained, no fine-tuning)

# Evaluate all on test set with:
# - Accuracy, Precision, Recall, F1 (weighted)
# - Confusion matrix
# - Per-class metrics
```

**Checklist:**
- [ ] 3 baseline models implemented
- [ ] Evaluation script with comprehensive metrics
- [ ] Baseline results saved (JSON)
- [ ] Comparison table created

---

## Phase 2: Fine-tuning Model (Weeks 2-4)

### Step 2.1: Data Preprocessing

**Your Task:** Create robust data preprocessing pipeline.

```python
# Expand: src/data/loader.py

class FinancialSentimentDataset:
    """
    PyTorch Dataset for financial sentiment texts.
    
    Responsibilities:
    - Tokenization with HF tokenizer
    - Padding/truncation to max_length
    - Label encoding (3-class: Negative, Neutral, Positive)
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        # TODO: Implement __init__, __len__, __getitem__
        pass
```

**Checklist:**
- [ ] FinancialSentimentDataset class created
- [ ] Tokenization working correctly
- [ ] Padding/truncation tested
- [ ] Label encoding verified
- [ ] DataLoader integration tested

### Step 2.2: Model Training

**Your Task:** Use the provided `FinancialSentimentClassifier` to train the model.

```python
# This code is already in: src/models/sentiment_classifier.py

# You'll create a training script:
# Create: src/models/train.py

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# TODO: Implement training loop:
# 1. Load config from configs/training_config.yaml
# 2. Initialize model with FinancialSentimentClassifier
# 3. Setup callbacks (checkpointing, early stopping)
# 4. Train with PyTorch Lightning

# Key considerations:
# - Use learning rate scheduler (2e-5 for fine-tuning)
# - Implement early stopping on val_f1_weighted
# - Save best checkpoint
# - Log metrics to MLflow
```

**Expected Results:**
- Training loss decreases over epochs
- Validation F1 > 0.80
- No overfitting (val loss should plateau, not increase sharply)

**Checklist:**
- [ ] Training script implemented
- [ ] Model training completes successfully
- [ ] Learning curves plotted (loss, F1)
- [ ] Best checkpoint saved
- [ ] Training took ~30-45 minutes on GPU

### Step 2.3: Hyperparameter Tuning

**Your Task:** Experiment with different learning rates and epochs.

```python
# Create: src/models/hyperparameter_search.py

# TODO: Test different combinations:
learning_rates = [1e-5, 2e-5, 5e-5]
epochs = [3, 5, 7]

# For each combination:
# 1. Train model
# 2. Record val_f1_weighted
# 3. Save results
# 4. Plot comparison heatmap
```

**Checklist:**
- [ ] 3+ hyperparameter configurations tested
- [ ] Results compared (table or heatmap)
- [ ] Best config selected and documented
- [ ] Reasoning for final config explained

---

## Phase 3: Evaluation & Analysis (Weeks 4-5)

### Step 3.1: Comprehensive Evaluation

**Your Task:** Evaluate your model across multiple dimensions.

```python
# Create: src/evaluation/evaluator.py

class ModelEvaluator:
    """Comprehensive evaluation of sentiment model."""
    
    def __init__(self, model, test_dataloader):
        pass
    
    def evaluate(self):
        # TODO: Calculate:
        # 1. Per-class metrics (Precision, Recall, F1)
        # 2. Confusion matrix
        # 3. AUROC (One-vs-Rest)
        # 4. Macro vs Weighted metrics
        
        return {
            "accuracy": ...,
            "precision": ...,
            "recall": ...,
            "f1_weighted": ...,
            "f1_macro": ...,
            "confusion_matrix": ...,
            "per_class_metrics": {...}
        }
```

**Checklist:**
- [ ] All metrics calculated correctly
- [ ] Confusion matrix plotted
- [ ] Per-class analysis documented
- [ ] Comparison with baselines created
- [ ] Results saved as JSON/PDF report

### Step 3.2: Error Analysis

**Your Task:** Deep dive into prediction errors.

```python
# Create: src/evaluation/error_analysis.py

# TODO: Analyze errors:
# 1. Which classes are confused most?
# 2. What text properties lead to errors?
#    - Text length
#    - Presence of hedging words
#    - Named entities (company names, ticker symbols)
# 3. Low-confidence predictions (confidence < 0.6)
# 4. Edge cases (sarcasm, mixed sentiment)

# Visualize:
# - Top 20 misclassified examples
# - Error distribution by text length
# - Confidence calibration curve
```

**Checklist:**
- [ ] Top 20 errors analyzed manually
- [ ] Common error patterns identified
- [ ] Confidence calibration assessed
- [ ] Error analysis report created (Jupyter notebook)

### Step 3.3: Bias & Fairness Analysis

**Your Task:** Evaluate model fairness across different domains.

```python
# Create: src/evaluation/fairness_analysis.py

# TODO: Analyze performance by:
# 1. Company sector (tech, finance, healthcare, etc.)
# 2. News source (Reuters, Bloomberg, etc.)
# 3. Text length (short, medium, long)
# 4. Time period (if applicable)

# Calculate disparate impact:
# - Is model biased toward certain sectors?
# - Does performance vary significantly by news source?

# Visualize:
# - F1 score by subgroup (bar chart)
# - Confusion matrices by sector
# - Error rate parity analysis
```

**Checklist:**
- [ ] Performance by subgroups calculated
- [ ] Significant disparities identified
- [ ] Fairness report created
- [ ] Recommendations for addressing bias documented

---

## Phase 4: Deployment & Documentation (Weeks 5-6)

### Step 4.1: Inference API

**Your Task:** Create a FastAPI server for model inference.

```python
# Create: src/api.py

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Financial Sentiment Analyzer")

class TextRequest(BaseModel):
    text: str
    return_confidence: bool = True

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: dict

@app.post("/predict")
async def predict(request: TextRequest) -> SentimentResponse:
    # TODO: Load model
    # TODO: Tokenize text
    # TODO: Get prediction
    # TODO: Return structured response
    pass

# Run with:
# uvicorn src.api:app --reload
```

**Checklist:**
- [ ] FastAPI app created
- [ ] Endpoints defined (/predict, /batch, /health)
- [ ] Proper error handling
- [ ] API tested with curl/Postman
- [ ] Response validation implemented

### Step 4.2: Streamlit Demo App

**Your Task:** Create interactive web demo.

```python
# Create: app.py

import streamlit as st
from src.models.sentiment_classifier import SentimentInference

st.title("💰 Financial Sentiment Analyzer")

# TODO:
# 1. Model loading (cached with @st.cache_resource)
# 2. Text input area
# 3. Single prediction view
# 4. Batch upload (CSV)
# 5. Results visualization (bar chart for probabilities)
# 6. Metrics display (model performance stats)

# Run with:
# streamlit run app.py
```

**Checklist:**
- [ ] App loads model efficiently (caching)
- [ ] Single text prediction works
- [ ] Batch upload working
- [ ] Visualizations are clear
- [ ] App responsive and user-friendly

### Step 4.3: Docker & Reproducibility

**Your Task:** Create Docker setup for reproducibility.

```dockerfile
# Create: Dockerfile

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run:
docker build -t financial-sentiment .
docker run -p 8000:8000 financial-sentiment
```

**Checklist:**
- [ ] Dockerfile created and tested
- [ ] Docker image builds successfully
- [ ] Container runs API
- [ ] All dependencies included

### Step 4.4: Documentation

**Your Task:** Create comprehensive documentation.

**Files to create:**
1. **MODEL_CARD.md** - Model details, limitations, ethical considerations
2. **ARCHITECTURE.md** - System design and components
3. **DEPLOYMENT.md** - Step-by-step deployment instructions
4. **INFERENCE.md** - How to use the model for predictions

**Checklist:**
- [ ] Model card completed
- [ ] Architecture diagram created
- [ ] Deployment guide written
- [ ] All notebooks documented with markdown
- [ ] README updated with results

---

## 📊 Expected Outcomes

By the end of Phase 4, you should have:

✅ **Code Quality**
- 80%+ test coverage
- All code follows PEP 8
- Type hints throughout
- Comprehensive docstrings

✅ **Model Performance**
- Test F1 score: > 0.82
- Weighted Precision/Recall: > 0.80
- Clear improvement over baselines
- Fair performance across subgroups (< 5% disparity)

✅ **Documentation**
- 4 Jupyter notebooks with analysis
- Model card with limitations
- API documentation (Swagger)
- Deployment guide

✅ **Deployment**
- FastAPI server running
- Streamlit demo app
- Docker image ready
- GitHub Pages with results

✅ **GitHub**
- 50+ commits with clear messages
- Professional repo structure
- All code in version control
- Reproducibility guaranteed

---

## 🎯 Success Criteria for Senior Role

**What makes this impressive:**

1. **Scientific Rigor**
   - Proper cross-validation strategy
   - Statistical significance testing
   - Bias analysis and mitigation

2. **Production-Ready Code**
   - Error handling and logging
   - API with proper validation
   - Docker containerization
   - CI/CD setup (GitHub Actions)

3. **Comprehensive Analysis**
   - Error analysis with insights
   - Fairness evaluation
   - Comparison with multiple baselines
   - Ablation studies (e.g., impact of layer freezing)

4. **Documentation**
   - Clear model limitations
   - Assumptions documented
   - Deployment instructions
   - Ethical considerations

5. **Results**
   - Outperforms baselines
   - Generalizes well (cross-dataset evaluation)
   - Low inference latency (< 500ms)
   - Fair across demographic groups

---

## 🚀 Next Steps

After completing this project:
1. Deploy to Hugging Face Spaces (free)
2. Share results on social media/LinkedIn
3. Write blog post on Medium/Dev.to
4. Move to Project 2 (Time Series Analysis)

---

## 📞 Support

When you get stuck or need guidance:
- Ask specific questions (not "is this correct?")
- Show error messages and your code
- We'll debug together, iteratively

Good luck! 🚀
