"""
Guards on the dataset contract.

These are cheap and they catch the class of bug that silently poisons every
downstream number: a split that leaks, a class list that drifted, or a
species mapping that stopped matching the dataset's own naming.

Author: Manuel Corona
"""

import numpy as np

from src.data.loader import (
    CAT_BREEDS,
    class_names,
    get_datasets,
    labels_of,
    species_of,
    split_indices,
)


def test_thirty_seven_classes_twelve_of_them_cats():
    names = class_names()
    assert len(names) == 37
    # The published dataset is 12 cat breeds and 25 dog breeds. An earlier
    # version of CAT_BREEDS used underscores and silently matched only 8,
    # which would have mislabelled every per-species analysis.
    assert species_of(names).sum() == 12
    assert CAT_BREEDS <= set(names)


def test_train_and_val_do_not_overlap():
    labels = np.repeat(np.arange(37), 100)
    train_idx, val_idx = split_indices(len(labels), labels)
    assert set(train_idx).isdisjoint(val_idx)
    assert len(train_idx) + len(val_idx) == len(labels)


def test_split_is_stratified_and_every_class_reaches_validation():
    labels = np.repeat(np.arange(37), 100)
    _, val_idx = split_indices(len(labels), labels)
    counts = np.bincount(labels[val_idx], minlength=37)
    assert counts.min() >= 1
    assert counts.max() - counts.min() <= 1


def test_split_is_deterministic():
    labels = np.repeat(np.arange(37), 100)
    a, _ = split_indices(len(labels), labels)
    b, _ = split_indices(len(labels), labels)
    assert np.array_equal(a, b)


def test_real_splits_are_disjoint_and_test_is_untouched():
    train, val, test = get_datasets(augment=False)
    assert set(train.indices).isdisjoint(val.indices)
    assert len(train) + len(val) == 3680      # official trainval size
    assert len(test) == 3669                  # official test size
    assert len(np.unique(labels_of(test))) == 37


def test_mcnemar_ignores_concordant_items():
    """A test that counted agreements would call these two models different."""
    import numpy as np
    from src.evaluation.significance import mcnemar_exact
    labels = np.zeros(100, dtype=int)
    a = np.zeros(100, dtype=int)      # right on everything
    b = np.zeros(100, dtype=int)
    b[:2] = 1                          # wrong on exactly 2
    r = mcnemar_exact(a, b, labels)
    assert r["discordant"] == 2 and r["a_only"] == 2 and r["b_only"] == 0
    assert r["p_value"] == 0.5         # 2 discordant, all one way: 2 * (1/4)


def test_mcnemar_is_symmetric_in_p():
    import numpy as np
    from src.evaluation.significance import mcnemar_exact
    labels = np.zeros(50, dtype=int)
    a, b = np.zeros(50, dtype=int), np.zeros(50, dtype=int)
    a[:5] = 1
    b[5:12] = 1
    assert mcnemar_exact(a, b, labels)["p_value"] == \
           mcnemar_exact(b, a, labels)["p_value"]
