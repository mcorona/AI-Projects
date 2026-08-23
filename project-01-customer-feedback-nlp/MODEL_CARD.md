# Model Card: Financial Sentiment Classifier

## Model Details

- **Architecture:** RoBERTa-base (125M params), fine-tuned for 3-class sequence classification.
- **Fine-tuning strategy:** the bottom 9 of 12 transformer layers are frozen; only the top 3 layers + classification head (60.9M / 124M params) are trained. See `src/models/sentiment_classifier.py`.
- **Labels:** `negative` (0), `neutral` (1), `positive` (2).
- **Input:** a single sentence or short passage of English financial/business text, max 128 tokens (RoBERTa BPE tokenizer). Longer inputs are truncated.
- **Output:** predicted label + softmax confidence + full 3-class probability distribution.
- **Training config:** LR 2e-5, batch size 32, 5 epochs max with early stopping on validation weighted F1 (patience 3). Full config in `configs/training_config.yaml`.
- **Best checkpoint:** epoch 3, validation weighted F1 = 0.867.

## Intended Use

- **Primary use case:** sentiment classification of short, formal financial-news-style text (earnings statements, press releases, analyst commentary) into negative/neutral/positive.
- **Intended users:** developers building financial text analysis tools, as a component in a larger pipeline (not a standalone investment signal).
- **Out of scope:** general-purpose sentiment analysis (social media, product reviews, casual conversation), languages other than English, sarcasm/irony detection, financial advice or investment recommendations, or any use as the sole basis for an automated decision affecting a person (e.g. credit, employment).

## Training Data

- **Source:** Financial PhraseBank ([Malo et al., 2014](https://arxiv.org/abs/1307.5336)), `sentences_allagree` variant (100% inter-annotator agreement).
- **Size:** 4,840 unique English sentences (6 exact duplicates removed from the raw 4,846) after de-duplication.
- **Provenance:** English-translated snippets from Finnish public-company press releases and financial news, circa 2009-2013.
- **Class balance:** neutral 59.3%, positive 28.1%, negative 12.5% (~4.8:1 majority:minority ratio).
- **Splits:** stratified 70/15/15 (train 3,392 / val 726 / test 728), seed 42. See `PHASE_1_SUMMARY.md`.

**Known data limitations** (see `PHASE_1_SUMMARY.md`, Bias Analysis):
- Single domain (finance/industrial sector), single era (~2009-2013), single source region (Finland), all sentences translated from Finnish to English -- generalization to other sectors, eras, regions, or genuinely native English financial text is untested.
- No sentence-length diversity: 99%+ of sentences are under 50 words, so behavior on longer documents (full articles, reports) is unvalidated.
- No sector/company metadata, so subgroup fairness analysis by sector was not possible.

## Evaluation

Evaluated on the 728-sample held-out test split (never used in training or model selection). Full detail in `PHASE_3_SUMMARY.md`.

| Metric | Value |
|---|---|
| Accuracy | 0.846 |
| F1 (weighted) | 0.846 |
| F1 (macro) | 0.832 |
| AUROC (one-vs-rest, weighted) | 0.948 |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| negative | 0.76 | 0.91 | 0.83 |
| neutral | 0.88 | 0.87 | 0.88 |
| positive | 0.81 | 0.77 | 0.79 |

**Baseline comparison** (same test set):

| Model | F1 (Weighted) |
|-------|----------------|
| TextBlob (rule-based) | 0.547 |
| TF-IDF + Linear SVM | 0.750 |
| DistilBERT (SST-2, no fine-tuning) | 0.217 |
| **This model** | **0.846** |

## Known Limitations & Failure Modes

From error analysis on the test set (112/728 errors, 15.4%; full detail in `PHASE_3_SUMMARY.md`):

1. **positive/neutral confusion dominates errors** (~70% of all mistakes). The model struggles most with subtly positive language that lacks strong lexical sentiment markers (e.g. "shareholders have irrevocably agreed to vote in favor..."). Sign-flip errors (positive↔negative) are rare (~5% of errors).
2. **Confidence is only well-calibrated above ~0.9.** In the 0.6-0.9 confidence range, the model is systematically over-confident (states 65-86% confidence, actually correct only 53-68% of the time). Any downstream system using confidence to gate automated decisions or trigger human review should use a threshold around 0.9, not a lower one, if it needs the stated confidence to be trustworthy.
3. **A known compositional weakness:** the model can key on a directional word (e.g. "increased") without correctly composing it with what it modifies (e.g. "loss increased" is negative, not positive) -- a classic sentiment-analysis negation/composition failure.
4. **Text length does not meaningfully affect the error rate** on this dataset (flat ~14-18% across length buckets) -- but recall this dataset has almost no length diversity, so this finding may not generalize to longer documents.
5. **Not evaluated for adversarial robustness, out-of-domain generalization, or fairness across any subgroup other than text length** (sector/company/demographic subgroup analysis was not possible with the available data -- see Training Data limitations above).

## Ethical Considerations

- **Not a financial advice tool.** Predictions reflect statistical patterns in a 2009-2013 Finnish financial-news corpus and should not be used, alone or combined with other signals, to make investment, lending, or other consequential financial decisions.
- **Training data reflects its source's biases.** The corpus captures the reporting norms, sector mix, and economic conditions of a specific place and time; sentiment patterns learned here may not transfer to other markets, sectors, or periods, and using the model outside this scope without re-validation risks silently misleading outputs.
- **No demographic or protected-attribute data** is present in or was used with this dataset, and the model was not evaluated for disparate impact on any protected group -- it operates purely on financial-news text.

## How to Use

See the "Getting Started" and "Results" sections of `README.md` for running the FastAPI server, the Streamlit demo, or the Docker container.
