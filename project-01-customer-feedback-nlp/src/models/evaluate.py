"""
Evaluation entrypoint for a fine-tuned FinancialSentimentClassifier checkpoint.

Usage:
    python -m src.models.evaluate --model-path output/models/best-....ckpt

Author: Manuel Corona
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import f1_score, roc_auc_score

from src.evaluation.baseline_evaluation import LABELS, evaluate_predictions, comparison_table
from src.models.sentiment_classifier import SentimentInference


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to a FinancialSentimentClassifier .ckpt file.")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--device", default=None, help="cuda/mps/cpu; defaults to auto-detect.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    test_df = pd.read_parquet(cfg["data"]["test_path"])

    device = args.device
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    inference = SentimentInference(args.model_path, device=device)
    details = inference.predict_batch_with_details(
        test_df["text"].tolist(), batch_size=cfg["data"]["test_batch_size"]
    )

    y_true = test_df["label"].tolist()
    y_pred = [d["prediction"].lower() for d in details]
    confidences = np.array([d["confidence"] for d in details])

    metrics = evaluate_predictions(y_true, y_pred)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0, labels=LABELS)

    # Multi-class AUROC (one-vs-rest) needs per-class probabilities in LABELS order.
    label_to_idx = {label: i for i, label in enumerate(LABELS)}
    y_true_idx = [label_to_idx[l] for l in y_true]
    proba_matrix = np.array([[d["probabilities"][label.capitalize()] for label in LABELS] for d in details])
    metrics["auroc_ovr_weighted"] = roc_auc_score(y_true_idx, proba_matrix, multi_class="ovr", average="weighted")

    print(f"Accuracy: {metrics['accuracy']:.3f}  F1(weighted): {metrics['f1_weighted']:.3f}  "
          f"F1(macro): {metrics['f1_macro']:.3f}  AUROC(ovr,weighted): {metrics['auroc_ovr_weighted']:.3f}")
    print(metrics["classification_report"])

    # --- Comparison against Phase 1 baselines ---
    baseline_path = Path("output/reports/baseline_results.json")
    all_results = {}
    if baseline_path.exists():
        with open(baseline_path) as f:
            all_results = json.load(f)
    all_results["RoBERTa (fine-tuned)"] = metrics
    print("\n" + comparison_table(all_results))

    # --- Confusion matrix plot ---
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = np.array(metrics["confusion_matrix"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS, ax=ax, cbar=False)
    ax.set_title("RoBERTa (fine-tuned) - Test Set")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    Path("output/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig("output/figures/roberta_confusion_matrix.png", dpi=150)

    # --- Save full report ---
    Path("output/reports").mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in metrics.items() if k != "classification_report"}
    with open("output/reports/roberta_eval_results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    # --- Lightweight error analysis: lowest-confidence wrong predictions ---
    wrong = [
        {"text": t, "true": yt, "pred": d["prediction"].lower(), "confidence": d["confidence"]}
        for t, yt, d in zip(test_df["text"].tolist(), y_true, details) if d["prediction"].lower() != yt
    ]
    wrong.sort(key=lambda r: -r["confidence"])  # most confidently wrong first
    print(f"\nTotal errors: {len(wrong)} / {len(y_true)}")
    print("Top 10 most confidently-wrong predictions:")
    for r in wrong[:10]:
        print(f"  [{r['confidence']:.2f}] true={r['true']:<8} pred={r['pred']:<8} {r['text'][:90]}")
    with open("output/reports/roberta_errors.json", "w") as f:
        json.dump(wrong, f, indent=2)

    print("\nSaved: output/figures/roberta_confusion_matrix.png")
    print("Saved: output/reports/roberta_eval_results.json")
    print("Saved: output/reports/roberta_errors.json")


if __name__ == "__main__":
    main()
