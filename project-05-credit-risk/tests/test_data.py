"""
Contract tests for the dataset and the split.

These guard the things that would silently invalidate every number in the
project: a row appearing in two splits, a base rate drifting between them,
or an undocumented category code surviving into the feature matrix.
"""

import numpy as np
import pytest

from src.data.loader import get_splits, load_clean
from src.data.schema import (
    CATEGORICAL_FEATURES, EDUCATION_LABELS, FEATURES, MARRIAGE_LABELS,
    NUMERIC_FEATURES, REPAY_COLS, age_band,
)


@pytest.fixture(scope="module")
def splits():
    return get_splits()


@pytest.fixture(scope="module")
def clean():
    return load_clean()


def test_row_count(clean):
    assert len(clean) == 30_000


def test_no_missing_values(clean):
    assert not clean.isna().to_numpy().any()


def test_ids_unique(clean):
    assert clean["ID"].is_unique


def test_split_sizes(splits):
    train, val, test = splits
    assert (len(train), len(val), len(test)) == (18_000, 4_500, 7_500)
    assert len(train) + len(val) + len(test) == 30_000


def test_splits_are_disjoint(splits):
    train, val, test = splits
    ids = [set(part["ID"]) for part in (train, val, test)]
    assert ids[0] & ids[1] == set()
    assert ids[0] & ids[2] == set()
    assert ids[1] & ids[2] == set()
    assert len(ids[0] | ids[1] | ids[2]) == 30_000


def test_stratification_holds(splits, clean):
    """Every split must carry the same base rate -- thresholds depend on it."""
    overall = clean["default"].mean()
    for part in splits:
        assert abs(part["default"].mean() - overall) < 0.002


def test_split_is_deterministic():
    a = get_splits()[2]["ID"].to_numpy()
    b = get_splits()[2]["ID"].to_numpy()
    assert np.array_equal(a, b)


def test_undocumented_education_codes_are_folded(clean):
    assert set(clean["EDUCATION"].unique()) <= set(EDUCATION_LABELS)


def test_undocumented_marriage_codes_are_folded(clean):
    assert set(clean["MARRIAGE"].unique()) <= set(MARRIAGE_LABELS)


def test_folding_preserves_every_row(clean):
    """Folding is a relabelling, not a filter."""
    assert len(clean) == 30_000
    assert clean["EDUCATION"].notna().all()
    assert clean["MARRIAGE"].notna().all()


def test_age_bands_cover_the_observed_range(clean):
    assert clean["AGE_BAND"].notna().all()
    assert age_band(21) == "21-29"
    assert age_band(79) == "50+"


def test_feature_list_matches_the_frame(clean):
    assert set(FEATURES) <= set(clean.columns)
    assert set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES) == set()


def test_target_is_not_a_feature():
    assert "default" not in FEATURES


def test_repayment_columns_are_the_six_documented_months(clean):
    assert REPAY_COLS == ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    assert "PAY_1" not in clean.columns


def test_subgroup_cells_are_large_enough_to_report(splits):
    """
    The fairness audit slices test by protected attribute. Any cell it
    reports on has to be big enough for the number to mean something.
    """
    test = splits[2]
    for attr in ("SEX", "AGE_BAND"):
        counts = test.groupby(attr)["default"].agg(["size", "sum"])
        assert counts["size"].min() >= 500
        assert counts["sum"].min() >= 100
