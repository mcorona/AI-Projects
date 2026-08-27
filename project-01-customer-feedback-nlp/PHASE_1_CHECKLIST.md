# Phase 1: Setup & Exploration Checklist

**Duration:** Weeks 1-2 | **Time per week:** 12 hours  
**Focus:** Dataset exploration, baseline establishment, foundational code

---

## 📦 Task 1: Environment Setup & Dependencies

**Status:** ⏳ TODO | ⏸️ IN PROGRESS | ✅ DONE

- [ ] Create virtual environment
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Verify imports work:
  ```bash
  python -c "import torch; import transformers; print('✓ OK')"
  ```
- [ ] Test GPU availability (if applicable):
  ```bash
  python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
  ```

**Expected Time:** 30 minutes  
**Deliverables:** Working Python environment with all packages installed

---

## 📥 Task 2: Dataset Collection & Exploration

**Status:** ⏳ TODO | ⏸️ IN PROGRESS | ✅ DONE

### Step 2a: Download Customer Feedback Dataset
- [ ] Download dataset from [Kaggle](https://www.kaggle.com/datasets/sbhatti/financial-phrase-bank-sentences)
  - Alternative: Selenium scrape from original paper
  - Expected size: ~4,840 sentences, 3 classes
- [ ] Extract and place in `data/raw/Customer Feedback Dataset/`

### Step 2b: Exploratory Data Analysis (EDA)
- [ ] **Create notebook:** `notebooks/01_eda_feedback_data.ipynb`
- [ ] Load dataset:
  ```python
  import pandas as pd
  df = pd.read_csv("data/raw/Customer Feedback Dataset.csv")
  ```
- [ ] Answer these questions:
  - [ ] Dataset shape? (rows, columns)
  - [ ] Class distribution? (counts + percentages)
  - [ ] Is data imbalanced? (how much?)
  - [ ] Text length statistics? (min, max, mean, median)
  - [ ] Any missing values or duplicates?
  - [ ] Sample texts from each class
  
### Step 2c: Data Visualization
- [ ] Bar chart: Class distribution
  ```python
  import matplotlib.pyplot as plt
  df['label'].value_counts().plot(kind='bar')
  plt.title('Class Distribution')
  plt.show()
  ```
- [ ] Histogram: Text length distribution
- [ ] Word cloud for each class (optional but impressive)

### Step 2d: Create Data Loading Pipeline
- [ ] Create file: `src/data/loader.py`
- [ ] Implement `CustomerFeedbackDataset` class:
  ```python
  class CustomerFeedbackDataset:
      def __init__(self, texts: List[str], labels: List[int]):
          # TODO
      
      def __len__(self) -> int:
          # TODO
      
      def __getitem__(self, idx: int) -> Dict:
          # TODO: Return {'text': str, 'label': int}
  ```
- [ ] Implement data splitting:
  ```python
  def create_splits(df, train_size=0.7, val_size=0.15):
      # TODO: Stratified split
      # Returns: train_df, val_df, test_df
  ```

**Expected Time:** 4-6 hours  
**Deliverables:**
- Jupyter notebook with EDA findings (5+ visualizations)
- Documented observations about class imbalance
- `src/data/loader.py` with loading functions
- Train/val/test splits (70/15/15) saved as parquet files

**Acceptance Criteria:**
- [ ] All 3 classes present in all splits (stratified)
- [ ] No data leakage between splits
- [ ] Train/val/test sizes: ~3400/730/730

---

## 🏗️ Task 3: Data Preprocessing & Feature Preparation

**Status:** ⏳ TODO | ⏸️ IN PROGRESS | ✅ DONE

### Step 3a: Text Preprocessing
- [ ] Create: `src/data/preprocessing.py`
- [ ] Implement text cleaning:
  ```python
  def clean_text(text: str) -> str:
      # TODO:
      # - Lowercase
      # - Remove URLs
      # - Remove mentions (@user)
      # - Remove stock tickers ($AAPL)
      # - Remove extra whitespace
      # - Keep financial terms (don't remove stop words yet)
      return cleaned_text
  ```

### Step 3b: Tokenization
- [ ] Use RoBERTa tokenizer from HuggingFace:
  ```python
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained("roberta-base")
  ```
- [ ] Test tokenization on sample texts
- [ ] Document max token length (target: 128)

### Step 3c: PyTorch Dataset Class
- [ ] Expand `src/data/loader.py` with:
  ```python
  class TokenizedFinancialDataset:
      def __init__(self, texts, labels, tokenizer, max_length=128):
          # TODO: Tokenize all texts
          
      def __getitem__(self, idx):
          # TODO: Return {
          #   'input_ids': Tensor,
          #   'attention_mask': Tensor,
          #   'labels': int
          # }
  ```

**Expected Time:** 3-4 hours  
**Deliverables:**
- `src/data/preprocessing.py` with cleaning functions
- `TokenizedFinancialDataset` class working
- Unit tests for preprocessing (2-3 test cases)

**Acceptance Criteria:**
- [ ] Sample tokenized batch prints without errors
- [ ] Token counts < 128 for 95%+ of texts
- [ ] No data corrupted during preprocessing

---

## 📊 Task 4: Baseline Model Implementation

**Status:** ⏳ TODO | ⏸️ IN PROGRESS | ✅ DONE

Create 3 baseline models for performance comparison.

### Step 4a: Rule-Based Baseline (TextBlob)
- [ ] Create: `src/baselines.py`
- [ ] Implement:
  ```python
  from textblob import TextBlob
  
  def textblob_sentiment(text: str) -> str:
      polarity = TextBlob(text).sentiment.polarity
      if polarity > 0.1:
          return "Positive"
      elif polarity < -0.1:
          return "Negative"
      else:
          return "Neutral"
  ```
- [ ] Evaluate on test set

### Step 4b: Traditional ML Baseline (TF-IDF + SVM)
- [ ] Implement:
  ```python
  from sklearn.pipeline import Pipeline
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.svm import LinearSVC
  
  pipeline = Pipeline([
      ('tfidf', TfidfVectorizer(max_features=5000)),
      ('svm', LinearSVC(max_iter=1000))
  ])
  ```
- [ ] Train on train set, evaluate on test set

### Step 4c: Pre-trained Model Baseline (DistilBERT, no fine-tuning)
- [ ] Use HuggingFace pipeline:
  ```python
  from transformers import pipeline
  classifier = pipeline("sentiment-analysis", 
                       model="distilbert-base-uncased-finetuned-sst-2-english")
  ```
- [ ] Note: This model is trained on SST-2, not financial data
- [ ] Evaluate performance gap

### Step 4d: Evaluate All Baselines
- [ ] Create: `src/evaluation/baseline_evaluation.py`
- [ ] For each baseline, calculate:
  ```python
  from sklearn.metrics import (
      accuracy_score, precision_score, recall_score, f1_score,
      confusion_matrix, classification_report
  )
  
  metrics = {
      'accuracy': accuracy_score(y_true, y_pred),
      'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
      'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
      'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
      'f1_per_class': f1_score(y_true, y_pred, average=None),
      'confusion_matrix': confusion_matrix(y_true, y_pred),
  }
  ```

- [ ] Create baseline comparison table:
  ```
  | Model                    | Accuracy | F1 (Weighted) | Precision | Recall |
  |--------------------------|----------|---------------|-----------|--------|
  | TextBlob                 | 0.XX     | 0.XX          | 0.XX      | 0.XX   |
  | TF-IDF + SVM             | 0.XX     | 0.XX          | 0.XX      | 0.XX   |
  | DistilBERT (no fine-tune)| 0.XX     | 0.XX          | 0.XX      | 0.XX   |
  ```

**Expected Time:** 4-5 hours  
**Deliverables:**
- `src/baselines.py` with 3 baseline implementations
- `src/evaluation/baseline_evaluation.py` with evaluation script
- Baseline results table (markdown format)
- Confusion matrices for each baseline
- Jupyter notebook: `notebooks/00_baselines.ipynb`

**Acceptance Criteria:**
- [ ] All 3 baselines produce predictions on test set
- [ ] Baseline F1 scores between 0.60-0.75 (expected for domain shift)
- [ ] Results saved as JSON for later comparison

---

## 📈 Task 5: Bias Analysis (Preliminary)

**Status:** ⏳ TODO | ⏸️ IN PROGRESS | ✅ DONE

### Step 5a: Subgroup Analysis
- [ ] Create: `src/evaluation/bias_analysis.py`
- [ ] Analyze class distribution by:
  - [ ] Text length (short < 50, medium 50-100, long > 100 tokens)
  - [ ] Financial domain (if identifiable: tech, finance, healthcare)
  
### Step 5b: Document Imbalance
- [ ] Calculate per-class metrics:
  ```python
  for class_name in ['Negative', 'Neutral', 'Positive']:
      class_data = df[df['label'] == class_name]
      print(f"{class_name}: {len(class_data)} samples, "
            f"{class_data['text_length'].mean():.1f} avg tokens")
  ```

**Expected Time:** 1-2 hours  
**Deliverables:**
- Subgroup analysis documenting any disparities
- Recommendations for addressing imbalance (noted for Phase 2)

---

## 📝 Task 6: Documentation & Summary

**Status:** ⏳ TODO | ⏸️ IN PROGRESS | ✅ DONE

- [ ] Summarize Phase 1 findings in `PHASE_1_SUMMARY.md`:
  - Dataset size, class distribution, imbalance
  - Baseline performance (what to beat)
  - Data quality observations
  - Preliminary bias findings
  
- [ ] Create git commits for each milestone:
  ```bash
  git add notebooks/01_eda*.ipynb
  git commit -m "Task 2: Dataset exploration and EDA"
  
  git add src/data/
  git commit -m "Task 3: Data preprocessing and tokenization"
  
  git add src/baselines.py src/evaluation/
  git commit -m "Task 4: Baseline model implementation"
  ```

**Expected Time:** 1-2 hours  
**Deliverables:**
- Phase 1 summary document
- Clean git history with 4-5 commits

---

## 🎯 Phase 1 Completion Criteria

You're ready for Phase 2 when ALL of these are true:

✅ **Environment**
- [ ] Python environment set up with all dependencies
- [ ] GPU access verified (if available)

✅ **Data**
- [ ] Dataset downloaded and explored (1200+ samples per class)
- [ ] No missing values or duplicates
- [ ] Train/val/test splits created (70/15/15, stratified)

✅ **Code**
- [ ] `src/data/loader.py` with data loading
- [ ] Preprocessing pipeline working
- [ ] `TokenizedFinancialDataset` class implemented and tested

✅ **Baselines**
- [ ] 3 baseline models trained and evaluated
- [ ] Baseline F1 scores documented (should be 0.60-0.75)
- [ ] Confusion matrices created

✅ **Documentation**
- [ ] EDA notebook with findings
- [ ] README updated with dataset info
- [ ] Baseline results table in markdown

✅ **Git**
- [ ] 4-5 clean commits with descriptive messages
- [ ] All code in version control

---

## 📞 How to Use This Checklist

1. **Start Here:** Begin with Task 1 (30 min setup)
2. **Work Through Tasks:** In order, one task per session
3. **Mark Progress:** Update status (TODO → IN PROGRESS → DONE)
4. **Ask for Help:** When stuck, show:
   - The specific error message
   - Your code (relevant snippet)
   - What you've already tried
5. **Move Forward:** Once all ✅ are checked, move to Phase 2

---

## ⏱️ Time Breakdown

| Task | Estimated Time | Actual Time |
|------|-----------------|-------------|
| 1. Environment Setup | 0.5 hrs | __ hrs |
| 2. Dataset Exploration | 4-6 hrs | __ hrs |
| 3. Preprocessing | 3-4 hrs | __ hrs |
| 4. Baselines | 4-5 hrs | __ hrs |
| 5. Bias Analysis | 1-2 hrs | __ hrs |
| 6. Documentation | 1-2 hrs | __ hrs |
| **TOTAL** | **14-20 hrs** | **__ hrs** |

**Target:** Complete Phase 1 in ~16 hours (1.5 weeks @ 12h/week)

---

**Good luck! Remember: Quality over speed. Take time to understand each step.** 🚀
