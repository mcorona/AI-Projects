"""
Train the two transfer models. Nothing here touches the test split.

Usage:
    python -m src.models.run_training --stage probe
    python -m src.models.run_training --stage finetune --epochs 10

Author: Manuel Corona
"""

import argparse
import json
import pickle
from pathlib import Path

from torch.utils.data import DataLoader

from src.data.loader import class_names, get_datasets, get_loaders, pick_device
from src.models.backbone import extract_cached
from src.models.train import finetune, train_linear_probe

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "output" / "models"
REPORTS_DIR = ROOT / "output" / "reports"


def frozen_features(device: str):
    """2048-d penultimate features for train / val / test, cached on disk."""
    train, val, test = get_datasets(augment=False)
    mk = lambda ds: DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)  # noqa: E731
    return (extract_cached(mk(train), device, "train"),
            extract_cached(mk(val), device, "val"),
            extract_cached(mk(test), device, "test"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["probe", "finetune"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = pick_device()
    names = class_names()
    print(f"device={device} | {len(names)} classes")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage == "probe":
        (xtr, ytr), (xval, yval), _ = frozen_features(device)
        print(f"features: train {xtr.shape} val {xval.shape}")
        clf, info = train_linear_probe(xtr, ytr, xval, yval)
        with open(MODELS_DIR / "linear_probe.pkl", "wb") as f:
            pickle.dump(clf, f)
        (REPORTS_DIR / "linear_probe_selection.json").write_text(json.dumps(info, indent=2))
        print(f"saved linear_probe.pkl (val acc {info['val_accuracy']:.4f})")

    else:
        train_loader, val_loader, _ = get_loaders(batch_size=args.batch_size)
        finetune(len(names), train_loader, val_loader, device, epochs=args.epochs)


if __name__ == "__main__":
    main()
