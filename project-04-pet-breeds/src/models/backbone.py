"""
Pretrained backbones: penultimate-layer features and raw ImageNet logits.

Both outputs come from the same forward pass shape, and both are needed:
the features feed the linear probe and the fine-tune, and the 1000-way
logits feed the zero-shot baseline that measures how much of this task
ImageNet already solved.

Author: Manuel Corona
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache"

# ResNet-50 with the improved IMAGENET1K_V2 recipe (80.86% ImageNet top-1
# against V1's 76.13%). Using V1 would understate what an off-the-shelf
# pretrained model brings, which would flatter the fine-tuned model.
DEFAULT_BACKBONE = "resnet50"


def build_backbone(name: str = DEFAULT_BACKBONE, num_classes: int = None):
    """
    Load a pretrained torchvision model.

    Args:
        num_classes: if given, the ImageNet head is replaced by a fresh
            linear layer of this width (for fine-tuning). If None, the
            original 1000-way head is kept (for the zero-shot baseline).
    """
    if name != "resnet50":
        raise ValueError(f"unsupported backbone: {name}")
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    if num_classes is not None:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model, weights


@torch.no_grad()
def extract(loader: DataLoader, device: str, what: str = "features",
            name: str = DEFAULT_BACKBONE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the pretrained backbone over a loader.

    Args:
        what: "features" -> 2048-d penultimate activations (the fc layer is
              replaced by identity), or "logits" -> the untouched 1000-way
              ImageNet output.

    Returns (matrix, labels).
    """
    model, _ = build_backbone(name)
    if what == "features":
        model.fc = nn.Identity()
    elif what != "logits":
        raise ValueError(what)
    model.eval().to(device)

    out, ys = [], []
    for x, y in tqdm(loader, desc=f"extracting {what}"):
        out.append(model(x.to(device)).float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(out), np.concatenate(ys)


def extract_cached(loader: DataLoader, device: str, tag: str,
                   what: str = "features") -> Tuple[np.ndarray, np.ndarray]:
    """
    extract(), memoised on disk.

    The frozen backbone is deterministic given a deterministic transform,
    so re-extracting the same split is pure waste -- and the linear probe
    and every analysis downstream read the same matrices repeatedly.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{what}__{tag}.npz"
    if path.exists():
        d = np.load(path)
        return d["x"], d["y"]
    x, y = extract(loader, device, what)
    np.savez_compressed(path, x=x, y=y)
    return x, y
