"""
The model ladder, from a logistic regression to gradient boosting.

Two deliberate choices are recorded here rather than left to be inferred.

Gradient boosting is scikit-learn's HistGradientBoostingClassifier, not
LightGBM or XGBoost. It is the same histogram-boosting algorithm -- sklearn
built it after LightGBM -- and it drops a compiled OpenMP dependency that
makes the other two awkward to install on macOS. Nothing in this project's
conclusions turns on which of the three is used.

class_weight="balanced" is included as its own model rather than applied
everywhere, because it is the reflex fix for class imbalance and it is a
trap here: it leaves ranking roughly unchanged while destroying
calibration, and calibration is what the threshold in this project is
applied to. Having it as a labelled variant makes that measurable instead
of arguable.

Author: Manuel Corona
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES

SEED = 20260827


def _preprocessor(features: List[str]) -> ColumnTransformer:
    """
    Scale the numerics, one-hot the three categoricals.

    SEX, EDUCATION and MARRIAGE are one-hot encoded rather than passed as
    integers because their codes are nominal -- EDUCATION 3 is not "more"
    than EDUCATION 1 in any sense a linear model should be allowed to use.
    The repayment-status columns are left as integers: those genuinely are
    ordered, in months of delay.
    """
    num = [c for c in NUMERIC_FEATURES if c in features]
    cat = [c for c in CATEGORICAL_FEATURES if c in features]
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"), cat),
    ], remainder="drop")


def build(name: str, features: List[str], **kw):
    """One entry point so every model is constructed the same way."""
    if name == "logistic":
        return Pipeline([
            ("prep", _preprocessor(features)),
            ("clf", LogisticRegression(max_iter=2000, C=kw.get("C", 1.0),
                                       random_state=SEED)),
        ])
    if name == "logistic_balanced":
        return Pipeline([
            ("prep", _preprocessor(features)),
            ("clf", LogisticRegression(max_iter=2000, C=kw.get("C", 1.0),
                                       class_weight="balanced", random_state=SEED)),
        ])
    if name == "random_forest":
        return Pipeline([
            ("prep", _preprocessor(features)),
            ("clf", RandomForestClassifier(
                n_estimators=kw.get("n_estimators", 400),
                min_samples_leaf=kw.get("min_samples_leaf", 20),
                n_jobs=-1, random_state=SEED)),
        ])
    if name == "hist_gbm":
        return Pipeline([
            ("prep", _preprocessor(features)),
            ("clf", HistGradientBoostingClassifier(
                max_iter=kw.get("max_iter", 300),
                learning_rate=kw.get("learning_rate", 0.06),
                max_leaf_nodes=kw.get("max_leaf_nodes", 31),
                l2_regularization=kw.get("l2_regularization", 1.0),
                early_stopping=False, random_state=SEED)),
        ])
    if name == "hist_gbm_calibrated":
        # Isotonic calibration fitted by 5-fold cross-validation *inside the
        # training set*. Calibrating on the validation split instead would
        # spend the same rows twice -- once to learn the probability
        # mapping, once to choose the threshold applied to it -- and the
        # threshold is the number this project reports.
        base = build("hist_gbm", features, **kw)
        return CalibratedClassifierCV(base, method="isotonic", cv=5)
    raise ValueError(f"unknown model: {name}")


MODELS = ["logistic", "logistic_balanced", "random_forest",
          "hist_gbm", "hist_gbm_calibrated"]


def fit_predict(name: str, features: List[str],
                train: pd.DataFrame, evals: Dict[str, pd.DataFrame],
                **kw):
    """Fit on train, score the given frames. Returns (model, {split: probs})."""
    model = build(name, features, **kw)
    model.fit(train[features], train["default"].to_numpy())
    probs = {k: model.predict_proba(df[features])[:, 1] for k, df in evals.items()}
    return model, probs


def feature_importance(model, features: List[str]) -> List[Dict]:
    """
    Permutation-free importance, read straight off the fitted estimator.

    Coefficients for the linear model, impurity gain for the forest. Both
    are crude -- they are reported to sanity-check that the model leans on
    repayment history rather than on demographics, not as a causal claim.
    """
    prep = model.named_steps["prep"] if hasattr(model, "named_steps") else None
    if prep is None:
        return []
    names = list(prep.get_feature_names_out())
    clf = model.named_steps["clf"]
    if hasattr(clf, "coef_"):
        vals = np.abs(clf.coef_[0])
    elif hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
    else:
        return []
    order = np.argsort(-vals)
    return [{"feature": names[i].split("__", 1)[-1], "weight": float(vals[i])}
            for i in order]
