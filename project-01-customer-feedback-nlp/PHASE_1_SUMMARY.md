# Phase 1 Summary: Setup & Exploration

**Status:** Complete | **Dataset:** Financial PhraseBank (Malo et al., 2014)

---

## Dataset

- **Source:** Financial PhraseBank, `sentences_allagree` variant (100% annotator agreement), mirrored from GitHub since Kaggle/HuggingFace Hub were unreachable from this dev sandbox.
- **Size:** 4,846 raw sentences -> **4,840 unique** after dropping 6 exact duplicates.
- **Labels:** 3-class (negative / neutral / positive), no missing values.
- **Splits (stratified, seed=42):** train 3,392 / val 726 / test 728 — matches the ~3400/730/730 target.

### Class Distribution

| Class | Count | % |
|-------|-------|---|
| Neutral | 2,873 | 59.3% |
| Positive | 1,363 | 28.1% |
| Negative | 604 | 12.5% |

**Imbalance ratio (majority:minority): ~4.8:1.** Neutral dominance is expected — most financial-news statements are factual/procedural rather than strongly polarized. This drives the plan to use stratified splitting, weighted/per-class metrics, and (if needed in Phase 2) class-weighted loss.

### Text Length

- Median 21 words, mean 23.1, max 81 words (min 2).
- 100% of the corpus fits under 128 RoBERTa BPE tokens (p95 = 60 tokens, max observed = 103) — confirms `max_length=128` is a safe, non-wasteful budget for Phase 2 fine-tuning.

### Data Quality Notes

- Clean corpus: no nulls, only 6 exact duplicates (removed).
- Single domain/era/language: all sentences are English-translated snippets from Finnish public-company press releases, circa 2009–2013. There is **no sector diversity** (all finance/industrial) and **no sentence-length diversity** (see Bias Analysis below) — a real limitation for generalization claims later.

---

## Baseline Performance (what Phase 2 needs to beat)

Evaluated on the 728-sample test split:

| Model | Accuracy | F1 (Weighted) | Precision | Recall |
|-------|----------|----------------|-----------|--------|
| TextBlob (rule-based) | 0.562 | 0.547 | — | — |
| TF-IDF + Linear SVM | **0.754** | **0.748** | — | — |
| DistilBERT (SST-2, no fine-tune) | 0.310 | 0.217 | — | — |

Full metrics (precision/recall/per-class F1/confusion matrices) in `output/reports/baseline_results.json` and `notebooks/00_baselines.ipynb`.

**Interpretation:**
- **TF-IDF+SVM is the strongest baseline** by a wide margin — trained directly on in-domain vocabulary, it captures financial terms TextBlob's generic lexicon misses entirely.
- **TextBlob** performs near-chance-plus on the minority `negative` class specifically — it frequently reads financially-negative-but-lexically-neutral language (e.g. "decreased", "loss") as neutral or even mildly positive.
- **DistilBERT (SST-2)** underperforms sharply due to a *double* mismatch: (1) it's a **binary** classifier (no native neutral class — we approximate one via a low-confidence threshold, which rarely triggers since SST-2 tends to be over-confident even out-of-domain) and (2) generic movie-review domain vs. financial text. This double gap — task shape *and* domain — is exactly what Phase 2's fine-tuned, natively 3-class RoBERTa model is designed to close.
- **Target for Phase 2:** meaningfully exceed 0.754 weighted F1 (the TF-IDF+SVM ceiling), with particular attention to the minority `negative` class, where all three baselines are weakest.

---

## Preliminary Bias Analysis

Scope for Phase 1: text-length subgroups only (see `src/evaluation/bias_analysis.py`).

| Length bucket | n (test) | Accuracy | F1 (weighted) |
|---|---|---|---|
| short (<50 tokens) | 723 | 0.754 | 0.747 |
| medium (50–100 tokens) | 5 | 0.800 | 0.787 |
| long (>100 tokens) | 0 | — | — |

**Finding:** 723 of 728 test samples (99.3%) fall in the "short" bucket; "long" is empty. This isn't a meaningful bias signal — it's a **coverage limitation of the dataset itself**: Financial PhraseBank sentences are uniformly short, so length-based subgroup analysis has almost no statistical power here.

**Deferred to Phase 2+ (recommendations):**
1. **Sector/domain subgroup analysis** was not possible — the raw data has no sector/company labels. If cross-domain fairness matters for this project's stated goals (reviews, surveys, support tickets — per the README), Phase 2 should incorporate a second, more diverse dataset rather than relying solely on Financial PhraseBank.
2. **Cross-dataset generalization test:** hold out a differently-sourced financial-text sample to check whether performance holds outside this single-source, single-era corpus.
3. Re-run length-subgroup analysis once a more length-diverse dataset is available; current result is inconclusive by sample size alone.

---

## Environment Note (sandbox-specific, not relevant to local/production runs)

This dev sandbox blocks `huggingface.co` and `kaggle.com` at the network-policy level (confirmed via 403s on the outbound proxy). All data/model artifacts here were fetched from GitHub/S3 mirrors instead:
- Dataset: `raw.githubusercontent.com` mirror of Financial PhraseBank.
- RoBERTa tokenizer files (vocab/merges) and DistilBERT-SST2 weights: legacy `s3.amazonaws.com/models.huggingface.co` mirror.
- The installed `transformers` version's *slow* vocab-file tokenizer constructors (`RobertaTokenizer`, `GPT2Tokenizer`, `BertTokenizerFast(vocab_file=...)`) silently produced degenerate tokenization when loading from local vocab files; working around this required building `tokenizer.json` directly via the `tokenizers` library and loading through `PreTrainedTokenizerFast`.

None of this affects the production code in `src/` — it all uses the standard `AutoTokenizer.from_pretrained(...)` / Hub model-id API, which works normally with regular internet access (e.g. on a local machine or standard cloud GPU environment).

---

## Deliverables Completed

- [x] `data/raw/financial_phrasebank/all-data.csv` (gitignored, regenerate via the mirror URL documented in git history)
- [x] `data/processed/{train,val,test}.parquet` (gitignored, regenerate via `python -m src.data.loader`)
- [x] `notebooks/01_eda_feedback_data.ipynb` — executed, 3 figures in `output/figures/`
- [x] `notebooks/00_baselines.ipynb` — executed, results in `output/reports/baseline_results.json`
- [x] `src/data/loader.py`, `src/data/preprocessing.py`
- [x] `src/baselines.py`, `src/evaluation/baseline_evaluation.py`, `src/evaluation/bias_analysis.py`
- [x] This summary document

## Time Breakdown

| Task | Estimated | Actual |
|------|-----------|--------|
| 1. Environment Setup | 0.5 hrs | ~0.5 hrs |
| 2. Dataset Exploration | 4-6 hrs | ~2 hrs (mirror workaround added time, EDA itself was fast) |
| 3. Preprocessing | 3-4 hrs | ~1.5 hrs (tokenizer workaround added time) |
| 4. Baselines | 4-5 hrs | ~2 hrs (DistilBERT mirror workaround added time) |
| 5. Bias Analysis | 1-2 hrs | ~0.5 hrs |
| 6. Documentation | 1-2 hrs | ~0.5 hrs |

---

## Ready for Phase 2

All Phase 1 completion criteria are met: environment set up, data explored and split, preprocessing/tokenization validated end-to-end, 3 baselines trained and evaluated with a clear target to beat (0.754 F1), and a preliminary bias analysis on file with documented limitations. Next: fine-tune RoBERTa (`src/models/sentiment_classifier.py` is already scaffolded) per `DEVELOPMENT_GUIDE.md` Phase 2.
