"""
Single pass over the held-out test split, for every model at once.

The test split is read here and nowhere else. Model selection happened on
validation (linear probe C, fine-tune checkpoint epoch); this script is the
one place a number that will appear in the README gets produced.

Usage:
    python -m src.evaluation.run_eval

Author: Manuel Corona
"""

import json
import pickle
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from src.data.imagenet_overlap import in_imagenet
from src.data.loader import class_names, get_datasets, labels_of, pick_device, species_of
from src.evaluation.metrics import (
    basic_metrics,
    expected_calibration_error,
    group_accuracy,
    per_class_accuracy,
    top_confusions,
)
from src.evaluation.significance import (
    bootstrap_accuracy_diff,
    compare_on_group,
    mcnemar_exact,
)

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "output" / "models"
REPORTS_DIR = ROOT / "output" / "reports"


def evaluate(preds, labels, names, known, is_cat, probs=None):
    m = basic_metrics(preds, labels, probs)
    m["breeds_in_imagenet"] = group_accuracy(preds, labels, known)
    m["breeds_not_in_imagenet"] = group_accuracy(preds, labels, ~known)
    m["cats"] = group_accuracy(preds, labels, is_cat)
    m["dogs"] = group_accuracy(preds, labels, ~is_cat)
    m["top_confusions"] = top_confusions(preds, labels, names, k=8)
    if probs is not None:
        cal = expected_calibration_error(probs, labels)
        m["calibration"] = {k: v for k, v in cal.items() if k != "bins"}
        m["calibration_bins"] = cal["bins"]
    return m


def main():
    device = pick_device()
    names = class_names()
    known = in_imagenet(names)
    is_cat = species_of(names).astype(bool)
    _, _, test = get_datasets(augment=False)
    labels = labels_of(test)

    report, per_class, preds_by_model = {}, {}, {}

    # Baseline predictions, saved by run_baselines, so the significance
    # tests can compare the whole ladder rather than only the two models
    # trained here.
    for key in ("majority", "classical", "zero_shot"):
        path = MODELS_DIR / f"preds_{key}.npy"
        if path.exists():
            preds_by_model[key] = np.load(path)

    # --- linear probe on frozen features
    probe_path = MODELS_DIR / "linear_probe.pkl"
    if probe_path.exists():
        from src.models.backbone import extract_cached
        loader = DataLoader(test, batch_size=64, shuffle=False, num_workers=4)
        xte, yte = extract_cached(loader, device, "test")
        assert np.array_equal(yte, labels), "cached feature labels drifted from the dataset"
        with open(probe_path, "rb") as f:
            clf = pickle.load(f)
        probs = clf.predict_proba(xte)
        preds = probs.argmax(1)
        report["linear_probe"] = evaluate(preds, labels, names, known, is_cat, probs)
        per_class["linear_probe"] = per_class_accuracy(preds, labels, len(names)).tolist()
        preds_by_model["linear_probe"] = preds
        np.save(MODELS_DIR / "test_probs_linear_probe.npy", probs)
        print(f"linear_probe    acc={report['linear_probe']['accuracy']:.4f}")

    # --- full fine-tune
    ckpt_path = MODELS_DIR / "resnet50_finetuned.pt"
    if ckpt_path.exists():
        from src.models.train import load_finetuned, predict_proba
        model, ckpt = load_finetuned(device)
        loader = DataLoader(test, batch_size=64, shuffle=False, num_workers=4)
        probs = predict_proba(model, loader, device)
        preds = probs.argmax(1)
        report["finetuned"] = evaluate(preds, labels, names, known, is_cat, probs)
        report["finetuned"]["selected_epoch"] = ckpt["epoch"]
        report["finetuned"]["val_accuracy_at_selection"] = round(ckpt["val_accuracy"], 4)
        per_class["finetuned"] = per_class_accuracy(preds, labels, len(names)).tolist()
        preds_by_model["finetuned"] = preds
        np.save(MODELS_DIR / "test_probs_finetuned.npy", probs)
        print(f"finetuned       acc={report['finetuned']['accuracy']:.4f}")

    # --- is the difference real, and where does it come from?
    comparisons = {}
    for a, b in (("finetuned", "linear_probe"),
                 ("linear_probe", "zero_shot"),
                 ("finetuned", "zero_shot")):
        if a not in preds_by_model or b not in preds_by_model:
            continue
        pa, pb = preds_by_model[a], preds_by_model[b]
        comparisons[f"{a}_vs_{b}"] = {
            "overall": {"mcnemar": mcnemar_exact(pa, pb, labels),
                        "bootstrap": bootstrap_accuracy_diff(pa, pb, labels)},
            # The split the project turns on: is the advantage concentrated
            # in the breeds ImageNet never named?
            "breeds_in_imagenet": compare_on_group(pa, pb, labels, known),
            "breeds_not_in_imagenet": compare_on_group(pa, pb, labels, ~known),
        }
    if comparisons:
        (REPORTS_DIR / "significance.json").write_text(json.dumps(comparisons, indent=2))
        for name, c in comparisons.items():
            o = c["overall"]
            print(f"{name:32s} d={o['bootstrap']['mean_difference']:+.4f}  "
                  f"[{o['bootstrap']['ci_low']:+.4f}, {o['bootstrap']['ci_high']:+.4f}]  "
                  f"p={o['mcnemar']['p_value']:.2e}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "model_metrics.json"
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update(report)
    out.write_text(json.dumps(existing, indent=2))

    (REPORTS_DIR / "per_class_accuracy.json").write_text(json.dumps({
        "class_names": names,
        "in_imagenet": known.tolist(),
        "is_cat": is_cat.tolist(),
        "accuracy": per_class,
    }, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
