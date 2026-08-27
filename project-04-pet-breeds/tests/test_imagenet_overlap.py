"""
Guards on the Pet -> ImageNet mapping.

The central finding of this project rests on this table being right, so it
is checked against the actual ImageNet-1k category list rather than
trusted. A silently wrong entry here would move the headline number.

Author: Manuel Corona
"""

import pytest

from src.data.imagenet_overlap import NOT_IN_IMAGENET, PET_TO_IMAGENET, in_imagenet
from src.data.loader import class_names


@pytest.fixture(scope="module")
def imagenet_categories():
    from torchvision.models import ResNet50_Weights
    return ResNet50_Weights.IMAGENET1K_V2.meta["categories"]


def test_mapping_partitions_every_pet_class_exactly_once():
    names = set(class_names())
    mapped = set(PET_TO_IMAGENET) | set(NOT_IN_IMAGENET)
    assert mapped == names, mapped.symmetric_difference(names)
    assert set(PET_TO_IMAGENET).isdisjoint(NOT_IN_IMAGENET)


def test_every_index_is_a_valid_imagenet_class(imagenet_categories):
    for breed, idxs in PET_TO_IMAGENET.items():
        for i in idxs:
            assert 0 <= i < len(imagenet_categories), (breed, i)


def test_indices_are_unique_across_breeds():
    seen = set()
    for breed, idxs in PET_TO_IMAGENET.items():
        for i in idxs:
            assert i not in seen, f"{breed} reuses ImageNet index {i}"
            seen.add(i)


@pytest.mark.parametrize("breed,expected_substring", [
    ("Beagle", "beagle"),
    ("Japanese Chin", "Japanese spaniel"),
    ("Leonberger", "Leonberg"),
    ("Scottish Terrier", "Scotch terrier"),
    ("English Cocker Spaniel", "cocker spaniel"),
    ("German Shorthaired", "short-haired pointer"),
    ("Egyptian Mau", "Egyptian cat"),
])
def test_spot_check_the_non_obvious_renames(breed, expected_substring, imagenet_categories):
    idx = PET_TO_IMAGENET[breed][0]
    assert expected_substring.lower() in imagenet_categories[idx].lower()


def test_imagenet_has_no_class_for_the_absent_breeds(imagenet_categories):
    # The distinctive token of each absent breed must not name an ImageNet
    # class -- this is what would break if ImageNet were re-listed.
    joined = " | ".join(imagenet_categories).lower()
    for token in ("abyssinian", "ragdoll", "sphynx", "bengal", "birman",
                  "bombay", "maine coon", "shorthair", "russian blue",
                  "havanese", "shiba"):
        assert token not in joined, token


def test_coverage_is_lopsided_toward_dogs():
    names = class_names()
    known = in_imagenet(names)
    assert known.sum() == 24
    assert len(names) - known.sum() == 13
