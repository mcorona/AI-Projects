# Phase 3 Summary: Fine-tuning Results & Evaluation

**Status:** Complete | **Model:** RoBERTa-base, fine-tuned (top 3 of 12 layers unfrozen)

Covers Phase 2 (fine-tuning) training results and Phase 3 (evaluation & error analysis), since the two are inseparable from a results standpoint: Phase 2 produced the checkpoint, Phase 3 is everything measured about it.

---

## Training (Phase 2)

- **Config:** `configs/training_config.yaml` (`src/models/train.py`) — 5 epochs max, early stopping on `val_f1_weighted` (patience 3), LR 2e-5, batch size 32, layers 0-8 frozen (60.9M / 124M params trainable).
- **Hardware:** trained locally on Apple Silicon (MPS backend).
- **Result:** best checkpoint at **epoch 3, val F1 (weighted) = 0.867**.

## Test Set Evaluation (Phase 3, Step 3.1)

Evaluated on the held-out 728-sample test split (`src/models/evaluate.py`):

| Metric | Value |
|---|---|
| Accuracy | 0.846 |
| F1 (weighted) | 0.846 |
| F1 (macro) | 0.832 |
| AUROC (one-vs-rest, weighted) | 0.948 |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| negative | 0.76 | **0.91** | 0.83 |
| neutral | 0.88 | 0.87 | 0.88 |
| positive | 0.81 | 0.77 | 0.79 |

Note the val→test F1 gap (0.867 → 0.846) is normal: the val set was used for model selection (early stopping / best-checkpoint), so it's mildly optimistic; the test set was never touched during training.

### Comparison vs. Phase 1 Baselines

| Model | Accuracy | F1 (Weighted) | Precision | Recall |
|-------|----------|----------------|-----------|--------|
| TextBlob | 0.562 | 0.547 | 0.542 | 0.562 |
| TF-IDF + SVM | 0.755 | 0.750 | 0.750 | 0.755 |
| DistilBERT (no fine-tune) | 0.310 | 0.217 | 0.561 | 0.310 |
| **RoBERTa (fine-tuned)** | **0.846** | **0.846** | **0.848** | **0.846** |

**+9.6 points of weighted F1 over the best baseline (TF-IDF+SVM).** Fine-tuning delivered a clear, meaningful improvement — and specifically closed the gap on `negative`, the class every Phase 1 baseline struggled with most (recall jumped to 0.91).

---

## Error Analysis (Phase 3, Step 3.2)

112 / 728 test examples misclassified (15.4% error rate). Full detail in `src/evaluation/error_analysis.py`; raw predictions in `output/reports/roberta_test_predictions.csv`.

### Confused class pairs

| True → Pred | Count | % of errors |
|---|---|---|
| positive → neutral | 44 | 39% |
| neutral → positive | 34 | 30% |
| neutral → negative | 22 | 20% |
| negative → neutral | 6 | 5% |
| positive → negative | 4 | 4% |
| negative → positive | 2 | 2% |

**~70% of all errors are the positive↔neutral boundary.** Sign-flip errors (positive↔negative, the worst kind) account for only 5% of errors — the model rarely gets the direction completely backwards; when it's wrong, it's usually about *degree*, not *direction*.

### Error rate by text length

| Length | n | Error rate |
|---|---|---|
| <15 words | 190 | 14.2% |
| 15-25 words | 302 | 15.2% |
| 25-35 words | 142 | 17.6% |
| >35 words | 94 | 14.9% |

Essentially flat — text length is not a meaningful error driver for this model.

### Confidence calibration

| Confidence bin | n | Mean confidence | Accuracy |
|---|---|---|---|
| 0.5-0.6 | 25 | 0.55 | 0.64 (under-confident) |
| 0.6-0.7 | 45 | 0.65 | 0.53 (over-confident) |
| 0.7-0.8 | 45 | 0.75 | 0.53 (over-confident) |
| 0.8-0.9 | 80 | 0.86 | 0.68 (over-confident) |
| 0.9-1.0 | 530 (73% of test set) | 0.98 | 0.94 (well-calibrated) |

The model is well-calibrated only in its very-high-confidence band (>0.9), which is also where the majority of predictions (73%) land, and where accuracy is highest (94%). The 0.6-0.9 "middle" range is systematically over-confident (says 65-86% sure, actually right 53-68% of the time).

**Practical implication:** for a human-in-the-loop review workflow, a confidence threshold of **~0.9** (not the naive 0.6 default) is the more reliable cutoff for "trust the model" vs. "flag for review" — below 0.9, stated confidence is not a very reliable signal of correctness.

### Notable individual errors

- `"This bold spinning 360 red fabric design is set beautifully on Ercols Napoli sofa."` (true=positive, pred=neutral) — this doesn't read as financial text at all; likely a data-quality artifact in the source corpus rather than a genuine model failure.
- `"Operating loss increased to EUR 17mn from EUR 10.8mn"` (true=negative, pred=positive) — the model appears to key on the directional word "increased" without correctly composing it with what's increasing ("loss"). A known class of sentiment-analysis failure: a positive-coded word can modify a negative-polarity noun and flip the true sentiment.
- Most positive→neutral misses involve subtle, non-lexical positive framing with no strong sentiment word present (e.g. "shareholders have irrevocably agreed to vote in favor of the bond issue", "he expects banks to provide alternative financing") — arguably ambiguous even for a human annotator.

---

## Environment Notes (sandbox-specific)

Development/validation of `train.py` and `evaluate.py` happened in a sandbox that blocks `huggingface.co`; a local S3-mirrored `roberta-base` checkpoint was used there purely to smoke-test the code paths (see PHASE_1_SUMMARY.md for the same pattern). Actual training and evaluation reported above ran on the user's Mac (Apple Silicon, MPS) with normal internet access. One real bug was found and fixed in this process:

- **PyTorch ≥2.6 `weights_only` default change** broke `load_from_checkpoint` for our checkpoints (they pickle a custom `ModelConfig` dataclass via `save_hyperparameters()`). Fixed by passing `weights_only=False` explicitly (safe: we only ever load checkpoints this project trained itself).
- **PyTorch/MPS `Unaligned blit request` bug**: loading a checkpoint straight onto an MPS device crashed with an internal ATen assertion. Fixed by loading to CPU first (`map_location="cpu"`) and moving to the target device afterward via the existing `.to(device)` call.

Both fixes are in `src/models/sentiment_classifier.py` and needed by anyone loading these checkpoints on Apple Silicon with a recent PyTorch.

---

## Deliverables Completed

- [x] `src/models/train.py` — fine-tuning entrypoint, dynamic LR schedule computed from actual dataset size
- [x] `src/models/evaluate.py` — full metrics suite, baseline comparison table, confusion matrix, per-example predictions CSV
- [x] `src/evaluation/error_analysis.py` — confusion pairs, length/confidence subgroup analysis, calibration curve, top-20 errors
- [x] `output/models/best-epoch=3-val_f1_weighted=0.867.ckpt` (local only, gitignored — regenerate via `python -m src.models.train`)
- [x] `output/figures/roberta_confusion_matrix.png`, `roberta_calibration_curve.png`, `roberta_error_rate_by_length.png`
- [x] `output/reports/roberta_eval_results.json`, `roberta_errors.json`, `roberta_test_predictions.csv`
- [x] This summary document

## Ready for Phase 4

Model performance is validated and well-understood: strong overall (F1=0.846), specific and explainable failure modes (positive/neutral boundary, a few compositional negation-like misses), and a clear, actionable confidence-calibration finding for any future human-review integration. Next: FastAPI inference server + Streamlit demo (`DEVELOPMENT_GUIDE.md` Phase 4), or further hyperparameter tuning (`DEVELOPMENT_GUIDE.md` Step 2.3) if more headroom is wanted first.
