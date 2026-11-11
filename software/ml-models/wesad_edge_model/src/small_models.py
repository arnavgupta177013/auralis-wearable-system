"""
Step C model specifications: small deployable models on the pre-registered feature pool D, the labelled
references, the paper's four detectors (Part A), and model-size accounting.

Every spec is a `baselines.Spec` (so the Step B harness functions work unchanged) extended with:
    floor  : "train" (harness training-only SD floor) | "paper" (10 % of the subject's own SD; TRANSDUCTIVE)
    meta   : dict of table columns (scheme, factors, part, ...)
Feature lists always follow src/feature_spec.py order (asserted).
"""

from dataclasses import dataclass, field
from itertools import product

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src import feature_spec as fs
from src.baselines import C_GRID, Spec

SEED = 0
D = list(fs.DEPLOYABLE)            # hr_mean, hr_std, eda_slope, scr_count, temp_slope, acc_mag_mean, motion_energy
ALWAYS = ["hr_mean", "eda_slope", "scr_count"]
MOTION = ["acc_mag_mean", "motion_energy"]
HRV = ["rmssd", "sdnn", "pnn50"]
ALL_NON_HRV = [f for f in fs.SPEC_ORDER if f not in HRV and fs.FEATURES[fs.SPEC_ORDER.index(f)]["role"]
               in ("deployable", "reference_absolute")]           # the 11 features of the drift references
PAPER_FEATURES = ["hr_mean", "hr_std", "rmssd", "sdnn", "pnn50", "scl_mean", "eda_std", "eda_slope",
                  "scr_count", "scr_amp_mean", "temp_mean", "temp_slope", "acc_mag_mean", "motion_energy"]

GRIDS = {     # simplest first: harness tie-break picks the first of equal inner PR-AUC
    "logreg": [{"C": c} for c in C_GRID],
    "tree": [{"max_depth": d, "min_samples_leaf": l} for d in (2, 3, 4, 5) for l in (100, 20)],
    "rf": [{"n_estimators": n, "max_depth": d} for n in (10, 30) for d in (3, 5)],
    "xgb": [{"n_estimators": n, "max_depth": d, "learning_rate": lr}
            for n in (20, 50) for d in (2, 3) for lr in (0.1, 0.3)],
}
MODEL_NAMES = {"logreg": "LR", "tree": "DT", "rf": "RF", "xgb": "XGB", "rf_paper": "RF-200 (paper)",
               "hr_threshold": "HR rule", "majority": "majority", "time_only": "time only"}


@dataclass
class SpecX(Spec):
    floor: str = "train"
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- estimators
def _estimator(model: str, params: dict, y):
    if model == "logreg":
        return LogisticRegression(C=params["C"], class_weight="balanced", max_iter=5000, random_state=SEED)
    if model == "tree":
        return DecisionTreeClassifier(class_weight="balanced", random_state=SEED, **params)
    if model == "rf":
        return RandomForestClassifier(min_samples_leaf=20, max_features="sqrt", class_weight="balanced",
                                      random_state=SEED, n_jobs=1, **params)
    if model == "rf_paper":   # the baseline's RF (wesad_baseline.make_model), single-threaded, same seed
        return RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=3, class_weight="balanced",
                                      random_state=42, n_jobs=1)
    if model == "xgb":
        y = np.asarray(y)
        spw = float((y == 0).sum() / max((y == 1).sum(), 1))
        return XGBClassifier(objective="binary:logistic", scale_pos_weight=spw, random_state=SEED, n_jobs=1,
                             tree_method="hist", eval_metric="logloss", **params)
    raise ValueError(model)


def make_fit_score(model: str, features: list):
    def fit(X, y, params):
        steps = [("impute", SimpleImputer(strategy="median"))]          # fitted on training rows only
        if model == "logreg":
            steps.append(("scale", StandardScaler()))
        steps.append(("model", _estimator(model, params, y)))
        return Pipeline(steps).fit(X[features].values, y)

    def score(fitted, X):
        return fitted.predict_proba(X[features].values)[:, 1]
    return fit, score


def _fit_hr_col(col):
    def fit(X, y, params):
        return {"fill": float(np.nanmedian(X[col]))}

    def score(fitted, X):
        return X[col].fillna(fitted["fill"]).values
    return fit, score


# ----------------------------------------------------------------------------- configurations
def d_subsets() -> list:
    """The 8 pre-registered factor subsets of D: (hr_std, motion, temp_slope) on/off."""
    out = []
    for hr_std, motion, temp in product((True, False), repeat=3):
        feats = list(ALWAYS) + (["hr_std"] if hr_std else []) + (MOTION if motion else []) + \
            (["temp_slope"] if temp else [])
        out.append({"hr_std": hr_std, "motion": motion, "temp_slope": temp, "features": fs.ordered(feats)})
    return out


def subset_label(sub: dict) -> str:
    return " ".join([("+" if sub[k] else "-") + k for k in ("hr_std", "motion", "temp_slope")])


SCHEME_NAMES = {"relative_only": "relative-only", "personal_z": "personal z", "universal_all": "universal (all)",
                "personal_z_all": "personal z (all)"}


def model_spec(model: str, scheme: str, features: list, role: str, order: int = 0, extra: dict = None,
               grid: list = None, floor: str = "train", key_suffix: str = "") -> SpecX:
    fs.assert_spec_order(features)
    personal = scheme.startswith("personal_z")
    fit, score = make_fit_score(model, features)
    meta = {"scheme": scheme, "model_family": model, **(extra or {})}
    name = "+".join(features)
    key = f"{model}__{scheme}__{name}{key_suffix}"
    disp = f"{MODEL_NAMES[model]} {SCHEME_NAMES.get(scheme, scheme)}"
    if extra and "subset" in extra:
        disp += f" [{extra['subset']}]"
    return SpecX(key=key, display=disp + key_suffix, model=model, features=list(features), fit=fit, score=score,
                 grid=grid or GRIDS[model], personal=personal, candidate=(role == "primary"), order=order,
                 role=role, floor=floor, meta=meta)


def primary_specs() -> list:
    specs, o = [], 100
    for model in ("logreg", "tree", "rf", "xgb"):
        for scheme in ("relative_only", "personal_z"):
            for sub in d_subsets():
                specs.append(model_spec(model, scheme, sub["features"], "primary", o,
                                        {"subset": subset_label(sub), **{k: sub[k] for k in
                                                                         ("hr_std", "motion", "temp_slope")}}))
                o += 1
    return specs


def reference_specs() -> list:
    """Drift/absolute references: all 11 non-HRV features, all factors on. Never selected."""
    specs, o = [], 500
    for model in ("logreg", "tree", "rf", "xgb"):
        for scheme in ("universal_all", "personal_z_all"):
            specs.append(model_spec(model, scheme, ALL_NON_HRV, "reference_drift", o,
                                    {"subset": "all non-HRV (incl. scl_mean, eda_std, scr_amp_mean, temp_mean)",
                                     "hr_std": True, "motion": True, "temp_slope": True}))
            o += 1
    return specs


def hr_spec(personal: bool, floor: str = "train", order: int = 1) -> SpecX:
    fit, score = _fit_hr_col("hr_mean")
    name = "HR z threshold (personal)" if personal else "HR threshold (universal)"
    if personal and floor == "paper":
        name += " [paper floor, TRANSDUCTIVE]"
    return SpecX(key=("hr_personal_z" if personal else "hr_universal") + ("__paperfloor" if floor == "paper" else ""),
                 display=name, model="hr_threshold", features=["hr_mean"], fit=fit, score=score,
                 personal=personal, order=order, role="baseline", floor=floor,
                 meta={"scheme": "personal_z" if personal else "universal", "model_family": "hr_threshold"})


def paper_rf_spec(scheme: str, features: list, floor: str = "train", suffix: str = "") -> SpecX:
    s = model_spec("rf_paper", scheme, features, "paper_reference", 900, grid=[{}], floor=floor,
                   key_suffix=suffix)
    s.meta.update(subset=f"{len(features)} features")
    return s


# ----------------------------------------------------------------------------- size accounting
NODE_BYTES = 12           # float32 threshold-or-value + int16 feature + int16 left + int16 right + 2 B padding
FLOAT_BYTES = 4


def model_size(spec: Spec, fitted) -> dict:
    """Parameter count, trees, nodes and ESTIMATED flat-array C size of one fitted model."""
    n_feat = len(spec.features)
    pre = n_feat * (2 if spec.personal else 1)          # imputer medians (+ personal-z floor values)
    if spec.model in ("hr_threshold",):
        return dict(n_params=2 + (1 if spec.personal else 0), n_trees=0, n_nodes=0, n_leaves=0,
                    est_bytes=FLOAT_BYTES * (2 + (1 if spec.personal else 0)))
    if spec.model == "majority":
        return dict(n_params=1, n_trees=0, n_nodes=0, n_leaves=0, est_bytes=FLOAT_BYTES)
    est = fitted.steps[-1][1] if hasattr(fitted, "steps") else fitted
    if spec.model == "logreg":
        n = n_feat + 1
        return dict(n_params=n + pre, n_trees=0, n_nodes=0, n_leaves=0, est_bytes=FLOAT_BYTES * (n + pre))
    if spec.model in ("tree", "time_only"):
        trees = [est]
    elif spec.model in ("rf", "rf_paper"):
        trees = est.estimators_
    elif spec.model == "xgb":
        df = est.get_booster().trees_to_dataframe()
        n_nodes, n_leaves = len(df), int((df.Feature == "Leaf").sum())
        n_trees = int(df.Tree.nunique())
        n_params = 2 * (n_nodes - n_leaves) + n_leaves + 1 + pre
        return dict(n_params=n_params, n_trees=n_trees, n_nodes=n_nodes, n_leaves=n_leaves,
                    est_bytes=NODE_BYTES * n_nodes + FLOAT_BYTES * (1 + pre))
    else:
        raise ValueError(spec.model)
    n_nodes = int(sum(t.tree_.node_count for t in trees))
    n_leaves = int(sum(t.tree_.n_leaves for t in trees))
    if spec.model == "time_only":
        pre = 0
    return dict(n_params=2 * (n_nodes - n_leaves) + n_leaves + pre, n_trees=len(trees), n_nodes=n_nodes,
                n_leaves=n_leaves, est_bytes=NODE_BYTES * n_nodes + FLOAT_BYTES * pre)
