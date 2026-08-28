"""
Which Oxford-IIIT Pet breeds already exist as ImageNet-1k classes.

This is the load-bearing table of the whole project, so it is written out
by hand and unit-tested rather than produced by fuzzy string matching.
Automatic matching gets this wrong in both directions: it pairs "American
Bulldog" with ImageNet's "French bulldog" (a different breed) and misses
"Japanese Chin" -> "Japanese spaniel" and "Leonberger" -> "Leonberg",
which are the same breeds under older names.

Why it matters: every "pretrained on ImageNet" model in this project was
trained on a label set that already contains most of this task. Reporting
transfer-learning accuracy without saying so is reporting a number that is
partly memorisation. The split below makes it measurable -- accuracy on
breeds ImageNet knows, against breeds it has never been given a name for.

Sources: the ImageNet-1k category list shipped in torchvision's weight
metadata, checked breed by breed against the Oxford-IIIT Pet class list.

Author: Manuel Corona
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

# Pet breed -> the ImageNet-1k class indices naming that same breed.
# A breed absent from this dict has no ImageNet-1k class.
PET_TO_IMAGENET: Dict[str, Tuple[int, ...]] = {
    "Basset Hound": (161,),                  # basset
    "Beagle": (162,),                        # beagle
    "Boxer": (242,),                         # boxer
    "Chihuahua": (151,),                     # Chihuahua
    "Egyptian Mau": (285,),                  # Egyptian cat
    "English Cocker Spaniel": (219,),        # cocker spaniel
    "English Setter": (212,),                # English setter
    "German Shorthaired": (210,),            # German short-haired pointer
    "Great Pyrenees": (257,),                # Great Pyrenees
    "Japanese Chin": (152,),                 # Japanese spaniel (same breed, older name)
    "Keeshond": (261,),                      # keeshond
    "Leonberger": (255,),                    # Leonberg (same breed, older name)
    "Miniature Pinscher": (237,),            # miniature pinscher
    "Newfoundland": (256,),                  # Newfoundland
    "Persian": (283,),                       # Persian cat
    "Pomeranian": (259,),                    # Pomeranian
    "Pug": (254,),                           # pug
    "Saint Bernard": (247,),                 # Saint Bernard
    "Samoyed": (258,),                       # Samoyed
    "Scottish Terrier": (199,),              # Scotch terrier (same breed, older name)
    "Siamese": (284,),                       # Siamese cat
    # Two indices: ImageNet splits the Staffordshire Bull Terrier and the
    # American Staffordshire Terrier, which most registries treat as
    # separate breeds descended from the same stock. Both are accepted for
    # this class rather than arbitrarily picking one.
    "Staffordshire Bull Terrier": (179, 180),
    "Wheaten Terrier": (202,),               # soft-coated wheaten terrier
    "Yorkshire Terrier": (187,),             # Yorkshire terrier
}

# Breeds with no ImageNet-1k class, listed explicitly so the count is
# auditable rather than implied by absence.
#
# One judgment call is recorded here rather than buried: American Pit Bull
# Terrier is treated as ABSENT. ImageNet has "American Staffordshire
# terrier", which shares ancestry and is often confused with the APBT, but
# the UKC and AKC register them as different breeds. Counting it as present
# would inflate the "ImageNet knows this breed" group with a case where the
# label is genuinely contested.
NOT_IN_IMAGENET: Tuple[str, ...] = (
    "Abyssinian",                 # cat
    "American Bulldog",           # dog -- ImageNet has French bulldog, a different breed
    "American Pit Bull Terrier",  # dog -- see note above
    "Bengal",                     # cat
    "Birman",                     # cat
    "Bombay",                     # cat
    "British Shorthair",          # cat
    "Havanese",                   # dog
    "Maine Coon",                 # cat
    "Ragdoll",                    # cat
    "Russian Blue",               # cat
    "Shiba Inu",                  # dog
    "Sphynx",                     # cat
)


def in_imagenet(names: List[str]) -> np.ndarray:
    """Boolean mask over the class-index order: does ImageNet name this breed?"""
    return np.array([n in PET_TO_IMAGENET for n in names])


def imagenet_indices_for(name: str) -> Optional[Tuple[int, ...]]:
    return PET_TO_IMAGENET.get(name)


def coverage_summary(names: List[str], is_cat: np.ndarray) -> Dict[str, float]:
    """
    How much of this task ImageNet-1k already contains, split by species.

    ImageNet-1k devotes about 120 of its 1,000 classes to dog breeds and
    only 5 to domestic cats, so the coverage here is wildly lopsided -- and
    that lopsidedness is what makes the dataset a natural experiment.
    """
    known = in_imagenet(names)
    return {
        "breeds_total": int(len(names)),
        "breeds_in_imagenet": int(known.sum()),
        "dog_breeds_total": int((~is_cat.astype(bool)).sum()),
        "dog_breeds_in_imagenet": int((known & ~is_cat.astype(bool)).sum()),
        "cat_breeds_total": int(is_cat.sum()),
        "cat_breeds_in_imagenet": int((known & is_cat.astype(bool)).sum()),
    }


if __name__ == "__main__":
    from src.data.loader import class_names, species_of
    names = class_names()
    is_cat = species_of(names)
    s = coverage_summary(names, is_cat)
    print(f"breeds with an ImageNet-1k class : {s['breeds_in_imagenet']}/{s['breeds_total']}")
    print(f"  dogs : {s['dog_breeds_in_imagenet']}/{s['dog_breeds_total']}")
    print(f"  cats : {s['cat_breeds_in_imagenet']}/{s['cat_breeds_total']}")
