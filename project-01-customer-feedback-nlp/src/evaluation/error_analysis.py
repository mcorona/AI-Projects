"""
Deep-dive error analysis for a fine-tuned model's test-set predictions.

Consumes output/reports/roberta_test_predictions.csv (written by
src/models/evaluate.py) rather than reloading the model, so this can be
re-run quickly and independently of any checkpoint.

Usage:
    python -m src.evaluation.error_analysis

Author: Manuel Corona
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PREDICTIONS_PATH = "output/reports/roberta_test_predictions.csv"
LOW_CONFIDENCE_THRESHOLD = 0.6


def confusion_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Rank (true, pred) mistake pairs by frequency, most common first."""
    errors = df[~df["correct"]]
    pairs = errors.groupby(["true", "pred"]).size().reset_index(name="count")
    return pairs.sort_values("count", ascending=False).reset_index(drop=True)


def error_rate_by_length(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket by word count into quartile-scale bins (finer than Phase 1's
    bias_analysis.py buckets, which were sized for RoBERTa-token coverage
    and are too coarse to show variation on this mostly-short corpus).
    """
    df = df.copy()
    df["n_words"] = df["text"].str.split().str.len()
    df["length_bucket"] = pd.cut(
        df["n_words"], bins=[0, 15, 25, 35, float("inf")],
        labels=["<15 words", "15-25 words", "25-35 words", ">35 words"],
    )
    return df.groupby("length_bucket", observed=True).agg(
        n=("correct", "size"), error_rate=("correct", lambda s: 1 - s.mean())
    ).reset_index()


def confidence_calibration(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    Reliability diagram data: bin predictions by confidence, compare mean
    confidence in each bin to the empirical accuracy in that bin. A
    well-calibrated model has accuracy ~= confidence in every bin.
    """
    df = df.copy()
    df["bin"] = pd.cut(df["confidence"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    grouped = df.groupby("bin", observed=True).agg(
        n=("correct", "size"), mean_confidence=("confidence", "mean"), accuracy=("correct", "mean"),
    ).reset_index()
    return grouped[grouped["n"] > 0]


def main():
    df = pd.read_csv(PREDICTIONS_PATH)
    n = len(df)
    n_errors = (~df["correct"]).sum()
    print(f"Test set: {n} examples, {n_errors} errors ({n_errors/n*100:.1f}%)\n")

    # --- Most confused class pairs ---
    print("=== Most confused class pairs (true -> pred) ===")
    pairs = confusion_pairs(df)
    print(pairs.to_string(index=False))
    print()

    # --- Error rate by text length ---
    print("=== Error rate by text length ===")
    by_length = error_rate_by_length(df)
    print(by_length.to_string(index=False))
    print()

    # --- Low-confidence predictions ---
    low_conf = df[df["confidence"] < LOW_CONFIDENCE_THRESHOLD]
    high_conf = df[df["confidence"] >= LOW_CONFIDENCE_THRESHOLD]
    print(f"=== Confidence threshold analysis (<{LOW_CONFIDENCE_THRESHOLD}) ===")
    print(f"Low-confidence predictions: {len(low_conf)} ({len(low_conf)/n*100:.1f}% of test set), "
          f"accuracy={low_conf['correct'].mean():.3f}" if len(low_conf) else "Low-confidence predictions: 0")
    print(f"High-confidence predictions: {len(high_conf)} ({len(high_conf)/n*100:.1f}% of test set), "
          f"accuracy={high_conf['correct'].mean():.3f}")
    print()

    # --- Calibration curve ---
    calib = confidence_calibration(df)
    print("=== Confidence calibration (mean confidence vs. actual accuracy per bin) ===")
    print(calib.to_string(index=False))
    print()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.5)
    ax.plot(calib["mean_confidence"], calib["accuracy"], "o-", color="#4c72b0", label="Model")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Confidence Calibration - RoBERTa (fine-tuned)")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("output/figures/roberta_calibration_curve.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(by_length["length_bucket"].astype(str), by_length["error_rate"], color="#c44e52")
    ax.set_ylabel("Error rate")
    ax.set_title("Error Rate by Text Length")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("output/figures/roberta_error_rate_by_length.png", dpi=150)

    # --- Top 20 misclassified examples ---
    errors = df[~df["correct"]].copy()
    errors = errors.sort_values("confidence", ascending=False)
    print("=== Top 20 most confidently-wrong predictions ===")
    for _, r in errors.head(20).iterrows():
        print(f"  [{r['confidence']:.2f}] true={r['true']:<8} pred={r['pred']:<8} {r['text'][:90]}")

    print("\nSaved: output/figures/roberta_calibration_curve.png")
    print("Saved: output/figures/roberta_error_rate_by_length.png")


if __name__ == "__main__":
    main()
