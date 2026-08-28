"""
Fit every model, tune on validation, freeze.

This script never reads the test split. It writes fitted estimators to
output/models/ and a validation report to output/reports/, and that report
is the last place any choice in this project gets made -- which
hyperparameters, which model, which threshold. src/evaluation/run_eval.py
then reads test exactly once.

Selection is by validation *regret*, not validation AUC. That is the whole
argument of the project, so it would be incoherent to select on AUC and
then complain that AUC is the wrong objective.

Usage:  python -m src.models.run_training
"""

import json
import time
from typing import Dict, List

import joblib
import numpy as np

from src.data.loader import ROOT, get_splits
from src.data.schema import FEATURES
from src.evaluation import decision as dec
from src.evaluation.metrics import calibration, discrimination
from src.models.baselines import BASELINES
from src.models.train import feature_importance, fit_predict

MODEL_DIR = ROOT / "output" / "models"
REPORT = ROOT / "output" / "reports" / "validation.json"

# Small, honest grids. Wide enough that no model is handicapped by a bad
# default, narrow enough that 4,500 validation rows are not being asked to
# resolve hundreds of near-identical configurations -- which is how a
# validation split stops measuring anything, as Project 4 found out.
GRIDS: Dict[str, List[dict]] = {
    "logistic": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "logistic_balanced": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "random_forest": [{"min_samples_leaf": m} for m in (5, 20, 50)],
    "hist_gbm": [{"learning_rate": lr, "max_leaf_nodes": n}
                 for lr in (0.03, 0.06, 0.1) for n in (15, 31, 63)],
}


def main() -> None:
    train, val, test = get_splits()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    y_val = val["default"].to_numpy()
    # Balance actually outstanding in the final billing month, floored at
    # zero: a credit balance is not negative exposure, it is no exposure.
    exposure_val = np.maximum(val["BILL_AMT1"].to_numpy(), 0.0)

    results: Dict[str, dict] = {}
    searches: Dict[str, list] = {}

    for name, cls in BASELINES.items():
        t0 = time.time()
        model = cls().fit(train[FEATURES], train["default"].to_numpy())
        p = model.predict_proba(val[FEATURES])
        results[name] = _score(y_val, p, exposure_val, time.time() - t0, {})
        _persist(model, name, p)
        _log(name, results[name])

    for name, grid in GRIDS.items():
        best, best_row, best_probs, best_model = None, None, None, None
        rows = []
        t0 = time.time()
        for params in grid:
            model, probs = fit_predict(name, FEATURES, train, {"val": val}, **params)
            row = _score(y_val, probs["val"], exposure_val, 0.0, params)
            rows.append({"params": params, "roc_auc": row["roc_auc"],
                         "regret": row["regret_at_selected"]})
            if best_row is None or row["regret_at_selected"] < best_row["regret_at_selected"]:
                best, best_row, best_probs, best_model = params, row, probs["val"], model
        best_row["seconds"] = round(time.time() - t0, 2)
        results[name] = best_row
        searches[name] = rows
        if name in ("logistic", "random_forest"):
            results[name]["top_features"] = feature_importance(best_model, FEATURES)[:12]
        _persist(best_model, name, best_probs)
        _log(name, best_row)

    # Isotonic calibration on top of the winning boosting configuration.
    t0 = time.time()
    cal_params = {k: v for k, v in results["hist_gbm"]["params"].items()}
    model, probs = fit_predict("hist_gbm_calibrated", FEATURES, train,
                               {"val": val}, **cal_params)
    results["hist_gbm_calibrated"] = _score(
        y_val, probs["val"], exposure_val, time.time() - t0, cal_params)
    _persist(model, "hist_gbm_calibrated", probs["val"])
    _log("hist_gbm_calibrated", results["hist_gbm_calibrated"])

    n_good = int((y_val == 0).sum())
    ranked = sorted(results.items(), key=lambda kv: kv[1]["regret_at_selected"])
    selected = ranked[0][0]
    by_auc = max(results.items(), key=lambda kv: kv[1]["roc_auc"])[0]

    payload = {
        "cost_ratio": dec.DEFAULT_RATIO,
        "theoretical_threshold": dec.optimal_threshold(),
        "selected_model": selected,
        "best_auc_model": by_auc,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test_held_out": int(len(test)),
        "val_regret_decline_all": float(n_good),
        "val_regret_approve_all": float(int((y_val == 1).sum()) * dec.DEFAULT_RATIO),
        "models": results,
        "search": searches,
    }
    REPORT.write_text(json.dumps(payload, indent=2))
    print(f"\nlowest validation regret : {selected}")
    print(f"highest validation AUC   : {by_auc}")
    print(f"wrote {REPORT.relative_to(ROOT)}")


def _persist(model, name: str, probs: np.ndarray) -> None:
    joblib.dump(model, MODEL_DIR / f"{name}.joblib")
    np.save(MODEL_DIR / f"val_probs_{name}.npy", probs)


def _log(name: str, row: dict) -> None:
    print(f"{name:22s} val AUC {row['roc_auc']:.4f}  ECE {row['ece']:.4f}  "
          f"t* {row['selected_threshold']:.3f}  regret {row['regret_at_selected']:8.1f}  "
          f"({row['seconds']:.1f}s)  {row['params']}")


def _score(y, p, exposure, seconds: float, params: dict) -> dict:
    best = dec.best_threshold(y, p)
    n_good = int((y == 0).sum())
    return {
        **discrimination(y, p),
        **calibration(y, p),
        "params": params,
        "seconds": round(seconds, 2),
        "selected_threshold": best["threshold"],
        "theoretical_threshold": best["theoretical"],
        "regret_at_selected": best["regret"],
        "regret_at_half": dec.regret(y, p, 0.5),
        # A degenerate winner -- flag everyone -- means the scores carry no
        # decision value at this cost ratio, however good the AUC looks.
        "degenerate_policy": bool(best["flag_rate"] > 0.999 or best["flag_rate"] < 0.001),
        "flag_rate_at_selected": best["flag_rate"],
        "beats_decline_all": bool(best["regret"] < n_good),
        "regret_at_selected_exposure_weighted": dec.regret(
            y, p, best["threshold"], exposure=exposure),
    }


if __name__ == "__main__":
    main()
