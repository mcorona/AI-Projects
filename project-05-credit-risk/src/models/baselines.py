"""
The two things a lender can do without training anything.

Both produce genuine probabilities rather than labels, so they can be
scored, calibrated and thresholded on exactly the same footing as the
models -- which is the only way a comparison against them means anything.

Author: Manuel Corona
"""

from typing import Dict

import numpy as np
import pandas as pd

from src.data.schema import DELINQUENT_FROM


class BaseRate:
    """
    Predict the training default rate for everyone.

    The floor. Its AUC is 0.5 by construction, but it is perfectly
    calibrated in aggregate, which is a useful reminder that calibration
    alone is not evidence of a useful model.
    """

    name = "base_rate"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "BaseRate":
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.rate_, dtype=float)


class DelinquencyRule:
    """
    One column, one lookup table: the September repayment status.

    This is the decision a credit officer makes from the account screen
    without any model at all -- is this customer already behind? Fitted as
    the empirical default rate within each PAY_0 bucket, so it returns a
    probability rather than a flag and can be thresholded like the rest.

    It is the baseline that matters in this project. Any model has to
    justify itself against the single fact that a customer who is already
    two months late will probably be late again.
    """

    name = "delinquency_rule"
    column = "PAY_0"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "DelinquencyRule":
        v = X[self.column].to_numpy()
        self.prior_ = float(np.mean(y))
        self.table_: Dict[int, float] = {}
        for code in np.unique(v):
            mask = v == code
            # Buckets with almost no support get shrunk toward the prior
            # rather than trusted: PAY_0 = 8 has 19 rows in the full
            # dataset, and an unshrunk estimate there is noise with a
            # decimal point.
            n = int(mask.sum())
            obs = float(y[mask].mean())
            k = 20.0
            self.table_[int(code)] = (n * obs + k * self.prior_) / (n + k)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        v = X[self.column].to_numpy()
        return np.array([self.table_.get(int(c), self.prior_) for c in v])

    def as_flag(self, X: pd.DataFrame) -> np.ndarray:
        """The rule in its native form: flag anyone at least one month late."""
        return (X[self.column].to_numpy() >= DELINQUENT_FROM).astype(int)


BASELINES = {"base_rate": BaseRate, "delinquency_rule": DelinquencyRule}
