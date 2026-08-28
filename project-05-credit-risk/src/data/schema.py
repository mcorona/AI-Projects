"""
What the columns of the UCI credit-default dataset actually mean.

The dataset ships with a codebook that is incomplete in three places, and
every one of those gaps is a decision someone has to make before modelling.
They are made here, in one file, with the reasoning attached -- rather than
inside a preprocessing function where they would read as arithmetic.

Author: Manuel Corona
"""

from typing import Dict, List, Tuple

TARGET = "default payment next month"
ID_COL = "ID"

# The six repayment-status columns, most recent first. The dataset names
# the September column PAY_0 and then jumps to PAY_2 for August -- there is
# no PAY_1. That is an error in the original upload, preserved here because
# every published result on this dataset uses these names and renaming them
# would silently break comparability.
#
# The billing period runs April..September 2005; the target is default in
# October 2005. So every feature below is observable strictly before the
# outcome, and nothing here leaks.
REPAY_COLS: List[str] = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS: List[str] = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT_COLS: List[str] = [f"PAY_AMT{i}" for i in range(1, 7)]

MONTH_OF: Dict[str, str] = {
    "PAY_0": "2005-09", "PAY_2": "2005-08", "PAY_3": "2005-07",
    "PAY_4": "2005-06", "PAY_5": "2005-05", "PAY_6": "2005-04",
}

# Repayment status codes. The codebook documents -1 and 1..9 only.
#
#   -2  no consumption / no balance to repay that month
#   -1  balance paid in full and on time
#    0  revolving credit -- the minimum was paid, a balance was carried
#  1..9 payment delayed by that many months
#
# -2 and 0 are undocumented; the readings above are the consensus in the
# literature and are consistent with the billing columns (rows with -2 have
# a zero bill). They matter because a naive reader treats the column as
# ordinal and concludes that -2 is "better" than -1, which is a different
# claim: no debt is not the same as debt repaid. Both are treated as
# non-delinquent here, and DELINQUENT_FROM marks where delinquency starts.
DELINQUENT_FROM = 1

# Undocumented category codes, and how they are folded.
#
# EDUCATION is documented as 1=graduate school, 2=university, 3=high
# school, 4=others. The data also contains 0, 5 and 6 -- 445 rows between
# them, with no codebook entry. They are folded into 4 ("others") rather
# than dropped: dropping rows on the basis of an administrative coding gap
# would quietly delete a subpopulation, and 4/5/6 already share a residual
# meaning.
EDUCATION_MAP: Dict[int, int] = {0: 4, 1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 4}
EDUCATION_LABELS: Dict[int, str] = {
    1: "Graduate school", 2: "University", 3: "High school", 4: "Other/unknown",
}

# MARRIAGE is documented as 1=married, 2=single, 3=others; 54 rows carry 0.
MARRIAGE_MAP: Dict[int, int] = {0: 3, 1: 1, 2: 2, 3: 3}
MARRIAGE_LABELS: Dict[int, str] = {1: "Married", 2: "Single", 3: "Other/unknown"}

SEX_LABELS: Dict[int, str] = {1: "Male", 2: "Female"}

# Age bands for the subgroup audit. Chosen before looking at any outcome by
# band, so the cut points are not selected to produce a gap.
AGE_BANDS: List[Tuple[str, int, int]] = [
    ("21-29", 21, 29), ("30-39", 30, 39), ("40-49", 40, 49), ("50+", 50, 200),
]

# Attributes the fairness audit reports on. SEX and AGE are protected under
# most credit regulation (in the US, ECOA covers both); EDUCATION is not
# protected but is a strong proxy for socioeconomic status and is included
# because a disparity there is worth seeing even when it is legal.
PROTECTED: List[str] = ["SEX", "AGE_BAND", "EDUCATION"]

# Everything the models are allowed to see. SEX, EDUCATION and MARRIAGE are
# included by default -- the audit later measures what happens when they
# are removed, which is only a meaningful experiment if they were in to
# begin with.
NUMERIC_FEATURES: List[str] = ["LIMIT_BAL", "AGE"] + REPAY_COLS + BILL_COLS + PAYAMT_COLS
CATEGORICAL_FEATURES: List[str] = ["SEX", "EDUCATION", "MARRIAGE"]
FEATURES: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def age_band(age: int) -> str:
    for label, lo, hi in AGE_BANDS:
        if lo <= age <= hi:
            return label
    return AGE_BANDS[-1][0]


def group_labels(attribute: str) -> Dict:
    return {
        "SEX": SEX_LABELS,
        "EDUCATION": EDUCATION_LABELS,
        "MARRIAGE": MARRIAGE_LABELS,
        "AGE_BAND": {b[0]: b[0] for b in AGE_BANDS},
    }[attribute]
