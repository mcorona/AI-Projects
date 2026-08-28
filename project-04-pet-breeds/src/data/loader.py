"""
Oxford-IIIT Pet: loading, splitting, and the transforms each model needs.

37 breeds (12 cat, 25 dog), ~7,350 images, with an official trainval/test
split published by the dataset authors. That official split is used as-is:
re-splitting the whole dataset randomly would make the numbers here
incomparable to every published result on this benchmark, which is the
same reason Project 3 used BEIR's official qrels rather than its own.

A validation set is carved out of trainval, stratified by class and seeded,
so model selection never touches test.

Author: Manuel Corona
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

VAL_FRACTION = 0.15
SPLIT_SEED = 20260827

# ImageNet channel statistics. Every model here is either pretrained on
# ImageNet or compared against one that is, so they all share the same
# normalization -- a mismatch would silently degrade the pretrained
# features without raising anything.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# The 12 cat breeds in this dataset. Everything else is a dog. Kept
# explicit because the dataset itself does not carry a species label, and
# several analyses below split on it.
CAT_BREEDS = {
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair",
    "Egyptian Mau", "Maine Coon", "Persian", "Ragdoll", "Russian Blue",
    "Siamese", "Sphynx",
}


def train_transform(size: int = 224) -> transforms.Compose:
    """
    Augmentation for fine-tuning.

    Deliberately mild: random resized crop and horizontal flip only. Pet
    photographs are already centred on the animal, and aggressive colour
    jitter would destroy exactly the cue that separates, say, a Russian
    Blue from a British Shorthair -- coat colour is part of the label here,
    not a nuisance variable.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(size: int = 224) -> transforms.Compose:
    """Deterministic resize + centre crop, used for val, test, and features."""
    return transforms.Compose([
        transforms.Resize(int(size * 256 / 224)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _base(split: str, transform=None) -> OxfordIIITPet:
    return OxfordIIITPet(
        root=str(DATA_DIR), split=split, target_types="category",
        transform=transform, download=True,
    )


def class_names() -> List[str]:
    return _base("test").classes


def split_indices(n_items: int, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratified train/val split of the trainval pool.

    Stratified rather than plain random because 37 classes over 3,680
    images is only ~99 per class; an unstratified 15% split would leave
    some classes with a handful of validation examples and make the
    validation metric mostly noise.
    """
    rng = np.random.default_rng(SPLIT_SEED)
    train_idx, val_idx = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        cut = max(1, int(round(len(idx) * VAL_FRACTION)))
        val_idx.extend(idx[:cut])
        train_idx.extend(idx[cut:])
    return np.sort(np.array(train_idx)), np.sort(np.array(val_idx))


def get_datasets(size: int = 224, augment: bool = True):
    """
    Returns (train, val, test) datasets.

    train and val index the same underlying trainval pool but carry
    different transforms -- val must not be augmented, or the validation
    metric measures a different distribution than the one being selected
    on.
    """
    pool_train = _base("trainval", train_transform(size) if augment else eval_transform(size))
    pool_eval = _base("trainval", eval_transform(size))
    labels = np.array([lbl for _, lbl in pool_train._labels_iter()]) \
        if hasattr(pool_train, "_labels_iter") else np.array(pool_train._labels)
    train_idx, val_idx = split_indices(len(labels), labels)
    test = _base("test", eval_transform(size))
    return Subset(pool_train, train_idx), Subset(pool_eval, val_idx), test


def get_loaders(batch_size: int = 32, size: int = 224, augment: bool = True,
                num_workers: int = 4):
    train, val, test = get_datasets(size, augment)
    mk = lambda ds, shuffle: DataLoader(  # noqa: E731
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=False, persistent_workers=num_workers > 0)
    return mk(train, True), mk(val, False), mk(test, False)


def labels_of(dataset) -> np.ndarray:
    """Extract integer labels without decoding a single image."""
    if isinstance(dataset, Subset):
        base = np.array(dataset.dataset._labels)
        return base[np.array(dataset.indices)]
    return np.array(dataset._labels)


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def species_of(names: List[str]) -> np.ndarray:
    """1 for cat, 0 for dog, aligned with the class-index order."""
    return np.array([1 if n in CAT_BREEDS else 0 for n in names])


if __name__ == "__main__":
    train, val, test = get_datasets()
    names = class_names()
    yt, yv, ys = labels_of(train), labels_of(val), labels_of(test)
    print(f"classes : {len(names)}  ({int(species_of(names).sum())} cat, "
          f"{int((1 - species_of(names)).sum())} dog)")
    print(f"train   : {len(train):,}")
    print(f"val     : {len(val):,}")
    print(f"test    : {len(test):,}")
    print(f"per-class train counts: min={np.bincount(yt).min()} "
          f"max={np.bincount(yt).max()}")
    print(f"per-class test counts : min={np.bincount(ys).min()} "
          f"max={np.bincount(ys).max()}")
