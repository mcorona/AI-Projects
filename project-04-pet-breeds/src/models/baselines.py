"""
The three baselines a fine-tuned model has to beat.

Each one exists to rule out a specific, cheap explanation for a good score:

  majority         Rules out "the classes are imbalanced". With 37 roughly
                   balanced classes this floor is ~2.7%, and any model near
                   it is broken rather than weak.

  classical        HOG + colour histogram into a linear SVM -- what image
                   classification looked like before deep learning. Rules
                   out "this task is easy enough that hand-designed
                   features solve it".

  zero_shot        The pretrained ImageNet classifier used directly, with
                   its 1000-way output restricted to the classes that name
                   these breeds. No fine-tuning, no training data touched
                   at all. This is the baseline that matters most here,
                   because 24 of the 37 breeds ARE ImageNet classes: it
                   measures how much of the task was already solved before
                   this project started.

Author: Manuel Corona
"""

from typing import Dict, List

import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog
from tqdm import tqdm

from src.data.imagenet_overlap import PET_TO_IMAGENET


def majority_predict(train_labels: np.ndarray, n_test: int) -> np.ndarray:
    """Always predict the most frequent training class."""
    return np.full(n_test, np.bincount(train_labels).argmax(), dtype=int)


# --- Classical computer vision ------------------------------------------

def classical_features(images) -> np.ndarray:
    """
    HOG plus a coarse 3-D colour histogram, per image.

    HOG captures shape and edge structure; the colour histogram captures
    coat colour, which for this dataset is genuinely part of the label
    (a Russian Blue is defined partly by being blue-grey). Together they
    are a fair representative of a strong pre-deep-learning pipeline --
    not a strawman built to lose.
    """
    feats = []
    for img in tqdm(images, desc="classical features"):
        arr = np.asarray(img, dtype=np.float32) / 255.0
        gray = rgb2gray(arr)
        h = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                cells_per_block=(2, 2), feature_vector=True)
        # 6x6x6 = 216 colour bins. Coarse on purpose: finer bins mostly add
        # sensitivity to lighting, which is a nuisance variable here.
        hist, _ = np.histogramdd(
            arr.reshape(-1, 3), bins=(6, 6, 6),
            range=((0, 1), (0, 1), (0, 1)))
        hist = (hist / hist.sum()).ravel()
        feats.append(np.concatenate([h, hist]))
    return np.stack(feats)


# --- Zero-shot ImageNet --------------------------------------------------

def zero_shot_predict(imagenet_logits: np.ndarray,
                      class_names: List[str]) -> np.ndarray:
    """
    Map a pretrained ImageNet classifier's output onto the 37 pet breeds.

    For each pet class that has ImageNet counterparts, its score is the max
    logit over those counterparts. Classes with no ImageNet counterpart get
    -inf and can never be predicted -- which is the honest representation
    of what this baseline is: a classifier that structurally cannot name 13
    of the 37 breeds. Reporting only its accuracy on the 24 it can name
    would be quoting a number for a different, easier task, so
    `zero_shot_report` returns both.
    """
    n = imagenet_logits.shape[0]
    scores = np.full((n, len(class_names)), -np.inf, dtype=np.float32)
    for c, name in enumerate(class_names):
        idxs = PET_TO_IMAGENET.get(name)
        if idxs:
            scores[:, c] = imagenet_logits[:, list(idxs)].max(axis=1)
    return scores.argmax(axis=1)


def zero_shot_report(preds: np.ndarray, labels: np.ndarray,
                     class_names: List[str]) -> Dict[str, float]:
    """Overall accuracy, plus accuracy on the subset the baseline can express."""
    reachable = np.array([class_names[y] in PET_TO_IMAGENET for y in labels])
    return {
        "accuracy": float((preds == labels).mean()),
        "accuracy_on_reachable_classes": float(
            (preds[reachable] == labels[reachable]).mean()),
        "reachable_share_of_test": float(reachable.mean()),
        "n_reachable": int(reachable.sum()),
    }
