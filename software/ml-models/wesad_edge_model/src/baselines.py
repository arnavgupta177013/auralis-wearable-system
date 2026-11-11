"""
Model specifications for the stress detection LOSO harness (Step B).

Every model is described by a `Spec`: which input columns it uses, whether those columns are
personal-z normalised, a small hyperparameter grid, and fit / score functions. The harness in
src/train.py treats them all the same way (inner GroupKFold for tuning + threshold, then one
score on the held-out subject).

    (a) majority class            always predicts the training majority (non-stress)
    (b) universal HR threshold    score = hr_mean, threshold chosen on training subjects
    (c) personal HR z threshold   score = personal-z hr_mean (SD floor from training subjects)
    (d) logistic regression       one per feature configuration (see src/schemes.py)
    time-only reference           shallow tree on minutes since baseline start. NOT A CANDIDATE:
                                  it only shows how strong the protocol-order shortcut is.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.schemes import Config, all_configs, feature_list

SEED = 0
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
TREE_DEPTHS = [1, 2, 3, 4]
TIME_COL = "time_since_baseline_start_min"


@dataclass
class Spec:
    key: str                         # unique id, used in file names / tables
    display: str                     # label used in tables and plots
    model: str                       # majority | hr_threshold | logreg | time_only
    features: list
    fit: Callable                    # fit(X DataFrame, y, params) -> fitted object
    score: Callable                  # score(fitted, X DataFrame) -> continuous score (higher = stress)
    grid: list = field(default_factory=lambda: [None])
    personal: bool = False           # z-score features against own calibration (fold-aware floor)
    candidate: bool = True
    majority: bool = False           # prediction = training majority class, no threshold
    cfg: Optional[Config] = None
    order: int = 0
    role: str = "baseline"           # baseline | main | reference (with HRV) | shortcut (time-only)


# ----------------------------------------------------------------------------- (a) majority class
def _fit_majority(X, y, params):
    return {"prevalence": float(np.mean(y)), "majority": int(np.mean(y) > 0.5)}


def _score_majority(fitted, X):
    return np.full(len(X), fitted["prevalence"])          # constant score -> ROC-AUC 0.5, PR-AUC = prevalence


# ----------------------------------------------------------------------------- (b), (c) HR thresholds
def _fit_hr(X, y, params):
    return {"fill": float(np.nanmedian(X["hr_mean"]))}    # NaN HR -> training median (no test info)


def _score_hr(fitted, X):
    return X["hr_mean"].fillna(fitted["fill"]).values


# ----------------------------------------------------------------------------- (d) logistic regression
def make_logreg(features: list) -> tuple:
    def fit(X, y, params):
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),     # fitted on training rows only
            ("scale", StandardScaler()),                      # fitted on training rows only
            ("lr", LogisticRegression(C=params["C"], class_weight="balanced", max_iter=5000,
                                      random_state=SEED)),
        ])
        return pipe.fit(X[features].values, y)

    def score(fitted, X):
        return fitted.predict_proba(X[features].values)[:, 1]
    return fit, score


# ----------------------------------------------------------------------------- time-only reference
def _fit_time(X, y, params):
    tree = DecisionTreeClassifier(max_depth=params["max_depth"], class_weight="balanced", random_state=SEED)
    return tree.fit(X[[TIME_COL]].values, y)


def _score_time(fitted, X):
    return fitted.predict_proba(X[[TIME_COL]].values)[:, 1]


# ----------------------------------------------------------------------------- all specs
def logreg_display(cfg: Config) -> str:
    s = {"universal": "universal", "personal_z": "personal z", "relative_only": "relative-only"}[cfg.scheme]
    s += " +motion" if cfg.motion else " -motion"
    if cfg.scheme != "relative_only":
        s += " +drift" if cfg.drift else " -drift"
    s += " +hr_std" if cfg.hr_std else " -hr_std"
    s += " +HRV (HRV reference)" if cfg.hrv else " -HRV"
    return "LR " + s


def logreg_spec(cfg: Config, order: int = 0) -> Spec:
    feats = feature_list(cfg)
    fit, score = make_logreg(feats)
    return Spec(key="logreg__" + cfg.name, display=logreg_display(cfg), model="logreg", features=feats,
                fit=fit, score=score, grid=[{"C": c} for c in C_GRID],
                personal=cfg.scheme == "personal_z", cfg=cfg, order=order,
                candidate=not cfg.is_reference, role="reference" if cfg.is_reference else "main")


def all_specs() -> list:
    specs = [
        Spec("majority", "majority class", "majority", [], _fit_majority, _score_majority, majority=True, order=0),
        Spec("hr_universal", "HR threshold (universal)", "hr_threshold", ["hr_mean"], _fit_hr, _score_hr, order=1),
        Spec("hr_personal_z", "HR z threshold (personal)", "hr_threshold", ["hr_mean"], _fit_hr, _score_hr,
             personal=True, order=2),
    ]
    specs += [logreg_spec(c, order=10 + i) for i, c in enumerate(all_configs())]
    specs.append(time_only_spec())
    return specs


def time_only_spec() -> Spec:
    # Deliberately uses the time column. It bypasses the time-feature assertion ONLY because
    # candidate=False; the harness refuses a time column for any candidate model.
    return Spec("time_only", "time only (NOT a candidate)", "time_only", [TIME_COL], _fit_time, _score_time,
                grid=[{"max_depth": d} for d in TREE_DEPTHS], candidate=False, order=99, role="shortcut")
