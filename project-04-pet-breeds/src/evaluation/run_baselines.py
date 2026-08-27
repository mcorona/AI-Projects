"""
Run the three baselines on the official test split and report them.

Usage:
    python -m src.evaluation.run_baselines
    python -m src.evaluation.run_baselines --only zero_shot

Author: Manuel Corona
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

from src.data.imagenet_overlap import in_imagenet
from src.data.loader import (
    DATA_DIR,
    class_names,
    eval_transform,
    get_datasets,
    labels_of,
    pick_device,
    species_of,
)
from src.evaluation.metrics import basic_metrics, group_accuracy, top_confusions
from src.models.baselines import (
    classical_features,
    majority_predict,
    zero_shot_predict,
    zero_shot_report,
)

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "output" / "reports"
CACHE = ROOT / "data" / "cache"

CLASSICAL_SIZE = 128


def _pil_split(split: str):
    """The raw images at a fixed size, for the classical (non-tensor) pipeline."""
    return OxfordIIITPet(
        root=str(DATA_DIR), split=split, target_types="category",
        transform=transforms.Resize((CLASSICAL_SIZE, CLASSICAL_SIZE)), download=False)


def run_majority(train_labels, test_labels):
    preds = majority_predict(train_labels, len(test_labels))
    return preds, {}


def run_classical(train_labels, test_labels):
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"classical_{CLASSICAL_SIZE}.npz"
    if path.exists():
        d = np.load(path)
        xtr, ytr, xte, yte = d["xtr"], d["ytr"], d["xte"], d["yte"]
    else:
        # The classical baseline trains on the full official trainval pool.
        # It has no hyperparameters tuned on validation here, so holding a
        # validation split out would only handicap it for no benefit --
        # the deep models get the same trainval pool.
        tr, te = _pil_split("trainval"), _pil_split("test")
        xtr = classical_features([img for img, _ in tr])
        ytr = np.array([y for _, y in tr])
        xte = classical_features([img for img, _ in te])
        yte = np.array([y for _, y in te])
        np.savez_compressed(path, xtr=xtr, ytr=ytr, xte=xte, yte=yte)

    clf = make_pipeline(StandardScaler(), LinearSVC(C=0.01, dual="auto", max_iter=5000))
    clf.fit(xtr, ytr)
    return clf.predict(xte), {"feature_dim": int(xtr.shape[1])}


def run_zero_shot(names):
    from src.models.backbone import extract_cached
    _, _, test = get_datasets(augment=False)
    loader = DataLoader(test, batch_size=64, shuffle=False, num_workers=4)
    logits, labels = extract_cached(loader, pick_device(), "test", what="logits")
    preds = zero_shot_predict(logits, names)
    return preds, zero_shot_report(preds, labels, names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None,
                        help="subset of: majority classical zero_shot")
    args = parser.parse_args()

    names = class_names()
    is_cat = species_of(names).astype(bool)
    known = in_imagenet(names)
    train, _, test = get_datasets(augment=False)
    train_labels, test_labels = labels_of(train), labels_of(test)

    print(f"{len(names)} classes | train {len(train_labels):,} | test {len(test_labels):,}")
    print(f"ImageNet-1k already names {known.sum()}/{len(names)} of these breeds "
          f"({(known & ~is_cat).sum()}/{(~is_cat).sum()} dogs, "
          f"{(known & is_cat).sum()}/{is_cat.sum()} cats)\n")

    runners = {"majority": lambda: run_majority(train_labels, test_labels),
               "classical": lambda: run_classical(train_labels, test_labels),
               "zero_shot": lambda: run_zero_shot(names)}
    selected = args.only or list(runners)

    REPORTS.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS / "baseline_metrics.json"
    report = json.loads(out_path.read_text()) if out_path.exists() else {}

    for name in selected:
        t0 = time.time()
        preds, extra = runners[name]()
        m = basic_metrics(preds, test_labels)
        m.update(extra)
        m["seconds"] = round(time.time() - t0, 1)
        # The split that the whole project turns on.
        m["breeds_in_imagenet"] = group_accuracy(preds, test_labels, known)
        m["breeds_not_in_imagenet"] = group_accuracy(preds, test_labels, ~known)
        m["cats"] = group_accuracy(preds, test_labels, is_cat)
        m["dogs"] = group_accuracy(preds, test_labels, ~is_cat)
        m["top_confusions"] = top_confusions(preds, test_labels, names, k=8)
        report[name] = m
        np.save(REPORTS.parent / "models" / f"preds_{name}.npy", preds) \
            if (REPORTS.parent / "models").exists() else None
        print(f"{name:12s} acc={m['accuracy']:.4f}  macroF1={m['macro_f1']:.4f}  "
              f"in-IN={m['breeds_in_imagenet']['accuracy']:.4f}  "
              f"not-in-IN={m['breeds_not_in_imagenet']['accuracy']:.4f}  "
              f"({m['seconds']}s)")

    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
