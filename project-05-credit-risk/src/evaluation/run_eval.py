"""
The single pass over the test split.

Everything read here was frozen by src/models/run_training.py: which model,
which hyperparameters, which threshold. Nothing below tunes anything. The
one model fitted in this script -- the demographically blinded variant --
reuses the already-selected hyperparameters and is trained on train only;
it exists to answer an audit question, not to compete for selection.

Usage:  python -m src.evaluation.run_eval
"""

import json
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from src.data.loader import ROOT, get_splits
from src.data.schema import FEATURES, PROTECTED, group_labels
from src.evaluation import decision as dec
from src.evaluation import fairness as fair
from src.evaluation import significance as sig
from src.evaluation.metrics import calibration, discrimination, rates_at, reliability_bins
from src.models.train import fit_predict

MODEL_DIR = ROOT / "output" / "models"
REPORTS = ROOT / "output" / "reports"

# The demographics a lender is not supposed to lend on. AGE is dropped
# alongside SEX/EDUCATION/MARRIAGE for the blinded variant -- it is
# protected under ECOA and it is the one of the four that a tree model can
# use most freely.
BLINDED_OUT = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]


def main() -> None:
    train, val, test = get_splits()
    val_report = json.loads((REPORTS / "validation.json").read_text())
    selected = val_report["selected_model"]
    ratio = val_report["cost_ratio"]

    y = test["default"].to_numpy()
    exposure = np.maximum(test["BILL_AMT1"].to_numpy(), 0.0)
    n_good = int((y == 0).sum())

    print(f"test: {len(test):,} accounts, {int(y.sum()):,} defaults "
          f"({y.mean():.4f}), exposure NT${exposure.sum():,.0f}")
    print(f"selected on validation: {selected}  "
          f"threshold {val_report['models'][selected]['selected_threshold']:.3f}\n")

    # ---- every model, scored once ------------------------------------
    probs: Dict[str, np.ndarray] = {}
    models: Dict[str, dict] = {}
    for name, row in val_report["models"].items():
        model = joblib.load(MODEL_DIR / f"{name}.joblib")
        p = model.predict_proba(test[FEATURES])
        p = p[:, 1] if p.ndim == 2 else p
        probs[name] = p
        t = row["selected_threshold"]
        models[name] = {
            **discrimination(y, p),
            **calibration(y, p),
            "threshold_from_validation": t,
            "rates": rates_at(y, p, t),
            "regret": dec.regret(y, p, t, ratio),
            "regret_at_half": dec.regret(y, p, 0.5, ratio),
            "regret_at_theoretical": dec.regret(y, p, dec.optimal_threshold(ratio), ratio),
            "regret_exposure_weighted": dec.regret(y, p, t, ratio, exposure),
            "value": dec.value_of_thresholding(y, p, t, ratio),
            "val_regret": row["regret_at_selected"],
            "val_roc_auc": row["roc_auc"],
        }
        np.save(MODEL_DIR / f"test_probs_{name}.npy", p)
        m = models[name]
        print(f"{name:22s} AUC {m['roc_auc']:.4f}  ECE {m['ece']:.4f}  "
              f"regret {m['regret']:8.1f}  @0.5 {m['regret_at_half']:8.1f}  "
              f"saved {m['value']['saved_vs_best_naive']:+8.1f}")

    p_sel = probs[selected]
    t_sel = val_report["models"][selected]["selected_threshold"]

    # ---- does the threshold matter more than the model? --------------
    threshold_study = {
        "selected_threshold": t_sel,
        "theoretical_threshold": dec.optimal_threshold(ratio),
        "curve": dec.sweep(y, p_sel, ratio, grid=np.linspace(0.01, 0.99, 99)),
        "regret_decline_all": float(n_good),
        "regret_approve_all": float(int(y.sum()) * ratio),
        "ratio_sensitivity": dec.ratio_sensitivity(y, p_sel),
        "ratio_sensitivity_weak_model": dec.ratio_sensitivity(y, probs["logistic"]),
    }

    # ---- capacity-constrained review ---------------------------------
    topk = dec.topk_review(y, p_sel, exposure)

    # ---- who pays for the errors -------------------------------------
    audit: Dict[str, dict] = {}
    for attr in PROTECTED:
        groups = test[attr].to_numpy()
        labels = group_labels(attr)
        report = fair.group_report(y, p_sel, groups, t_sel, ratio, exposure)
        for row in report:
            key = row["group"]
            row["label"] = str(labels.get(int(key) if key.isdigit() else key, key))
        audit[attr] = {
            "threshold": t_sel,
            "groups": report,
            "gaps": fair.gaps(report),
            "cost_of_equal_opportunity": fair.cost_of_equalising(
                y, p_sel, groups, t_sel, "recall", ratio),
            "cost_of_demographic_parity": fair.cost_of_equalising(
                y, p_sel, groups, t_sel, "flag_rate", ratio),
        }
        g = audit[attr]["gaps"]
        print(f"\n{attr:9s} gaps  decline {g['demographic_parity_gap']:.4f}  "
              f"recall {g['equal_opportunity_gap']:.4f}  "
              f"FPR {g['false_positive_rate_gap']:.4f}  "
              f"AUC {g['auc_gap']:.4f}  (base rate {g['base_rate_gap']:.4f})")

    # ---- does dropping demographics help? ----------------------------
    blinded_features = [f for f in FEATURES if f not in BLINDED_OUT]
    params = val_report["models"][selected].get("params", {})
    _, blind_probs = fit_predict(selected, blinded_features, train,
                                 {"val": val, "test": test}, **params)
    p_blind = blind_probs["test"]
    b_thresh = dec.best_threshold(val["default"].to_numpy(), blind_probs["val"], ratio)["threshold"]
    np.save(MODEL_DIR / "test_probs_blinded.npy", p_blind)

    blinded = {
        "features_removed": BLINDED_OUT,
        "n_features": len(blinded_features),
        "threshold_from_validation": b_thresh,
        **discrimination(y, p_blind),
        **calibration(y, p_blind),
        "regret": dec.regret(y, p_blind, b_thresh, ratio),
        "vs_full_auc": sig.auc_difference(y, p_blind, p_sel),
        "vs_full_regret": sig.regret_difference(y, p_blind, b_thresh, p_sel, t_sel, ratio),
        "gaps": {},
    }
    for attr in PROTECTED:
        groups = test[attr].to_numpy()
        blinded["gaps"][attr] = fair.gaps(
            fair.group_report(y, p_blind, groups, b_thresh, ratio, exposure))
    print(f"\nblinded ({len(blinded_features)} features)  AUC {blinded['roc_auc']:.4f}  "
          f"regret {blinded['regret']:.1f}")
    for attr in PROTECTED:
        print(f"  {attr:9s} decline gap {audit[attr]['gaps']['demographic_parity_gap']:.4f} "
              f"-> {blinded['gaps'][attr]['demographic_parity_gap']:.4f}")

    # ---- is any of this significant? ---------------------------------
    def compare(a: str, b: str) -> dict:
        ta = val_report["models"][a]["selected_threshold"]
        tb = val_report["models"][b]["selected_threshold"]
        return {
            "auc": sig.auc_difference(y, probs[a], probs[b]),
            "regret_per_account": sig.regret_difference(
                y, probs[a], ta, probs[b], tb, ratio),
            "mcnemar_on_decisions": sig.mcnemar(y, probs[a], ta, probs[b], tb),
        }

    tests = {
        f"{selected}_vs_delinquency_rule": compare(selected, "delinquency_rule"),
        f"{selected}_vs_logistic": compare(selected, "logistic"),
        f"{selected}_vs_random_forest": compare(selected, "random_forest"),
        "logistic_balanced_vs_logistic": compare("logistic_balanced", "logistic"),
    }
    for k, v in tests.items():
        r = v["regret_per_account"]
        print(f"\n{k}\n  AUC diff {v['auc']['point_estimate']:+.4f} "
              f"[{v['auc']['ci_low']:+.4f}, {v['auc']['ci_high']:+.4f}]"
              f"\n  regret/account {r['point_estimate']:+.4f} "
              f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
              f"   McNemar p={v['mcnemar_on_decisions']['p_value']:.2e}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "test_results.json").write_text(json.dumps({
        "n_test": int(len(test)),
        "n_defaults": int(y.sum()),
        "base_rate": float(y.mean()),
        "total_exposure": float(exposure.sum()),
        "cost_ratio": ratio,
        "selected_model": selected,
        "models": models,
        "threshold_study": threshold_study,
        "topk_review": topk,
        "significance": tests,
        "reliability": {k: reliability_bins(y, v) for k, v in probs.items()},
    }, indent=2))
    (REPORTS / "fairness.json").write_text(json.dumps({
        "selected_model": selected,
        "threshold": t_sel,
        "audit": audit,
        "blinded": blinded,
    }, indent=2))
    print(f"\nwrote {(REPORTS / 'test_results.json').relative_to(ROOT)} "
          f"and {(REPORTS / 'fairness.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
