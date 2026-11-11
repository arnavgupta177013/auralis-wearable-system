"""
Step D: final stress detection model (decisions in NOTES.md, "Step D - final model: DECISIONS", written after the
Step C results and before this run).

    R  reference (pre-registered Step C selection): LR relative-only [hr_mean, eda_slope, scr_count], 30 s, 5 s step
    D  deployment candidate: same, with the gain/offset-invariant eda_slope_rel, scr_count_rel (5-min calibration)

Run from software/ml-models/wesad_edge_model (repo venv; data read-only):
    python -m src.final_model --data "C:\\datasets\\WESAD"
Inputs : results/metrics/features_w30.csv, features_eda_w30.csv (src.hw_parity --eda), stepc_* files (reproduction
         checks), stepc_condition_order.csv (stress onset, src.protocol)
Outputs: results/metrics/final_*.csv|json, results/graphs/final_*.png, models/* (see NOTES.md, Step D)

Every LOSO uses the Step B/C harness unchanged (src.runner.run_loso_ext: leakage assertions on every outer and inner
fold). The held-out subject is touched once. The deployment model is fitted on all 15 subjects with the same inner
GroupKFold logic (src.runner.tune_ext) and therefore has NO held-out estimate.
"""

import argparse
import ast
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import checks
from src import evaluate as ev
from src import feature_spec as fs
from src import preprocessing as pp
from src.evaluate import SERIES, save_tight
from src.explore import CLASS_COLORS, INK, INK2, MUTED
from src.hw_parity import EDA_LSB_US, SCR_REL_K
from src.predict import MODEL_STEM
from src.runner import evaluate_ops, prepare_ext, smooth, tune_ext
from src.small_models import model_spec
from src.baselines import C_GRID
from src.stepc import onset_map, run_parallel, tabulate
from src.stepc_followup import eda_variant
from src.train import INNER_FOLDS, f1_threshold, fit_checked, load_windows, metric_checklist

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
GRAPHS = ROOT / "results" / "graphs"
MODELS = ROOT / "models"
WIN = 30
FEATS_R = ["hr_mean", "eda_slope", "scr_count"]
FEATS_D = fs.ordered([fs.REDEFINED.get(f, f) for f in FEATS_R])
SMOOTH_MN = (4, 5)
C_EXT = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]          # POST-HOC sensitivity only
MAIN_OPS = {"OP1": "pred_op1", "OP2_spec90": "pred_op2", "OP1_smooth4of5": "pred_s45"}
REF_OPS = {"OP1_smooth_inner_mn_reference": "pred_sm"}
ALL_OPS = {**MAIN_OPS, **REF_OPS}
DEFAULT_OP = "OP1_smooth4of5"
WORST = ["S13", "S2", "S3"]
LC_SIZES = (2, 4, 6, 8, 10, 12)
LC_DRAWS = 5
NAMES = {"R": "R (reference: eda_slope, scr_count)", "D": "D (deployment: eda_slope_rel, scr_count_rel)"}
COL = {"R": SERIES[0], "D": SERIES[1]}
MARK = {"R": "o", "D": "s"}
SUBSET = {"subset": "-hr_std -motion -temp_slope", "hr_std": False, "motion": False, "temp_slope": False}


# ----------------------------------------------------------------------------- data and specs
def load_data() -> dict:
    r = load_windows(WIN)
    d = eda_variant(r, WIN, "clean", True)       # adds eda_slope_rel / scr_count_rel (clean E4 data), asserts rows
    return {"R": r, "D": d}


def specs() -> dict:
    s = {"R": model_spec("logreg", "relative_only", FEATS_R, "final", 0, SUBSET),
         "D": model_spec("logreg", "relative_only", FEATS_D, "final", 1, SUBSET)}
    for c, feats in (("R", FEATS_R), ("D", FEATS_D)):
        s[c + "_ext"] = model_spec("logreg", "relative_only", feats, "post-hoc sensitivity", 2, SUBSET,
                                   grid=[{"C": c_} for c_ in C_EXT], key_suffix=" [POST-HOC extended C grid]")
    for k, v in s.items():
        v.meta["config"] = k.split("_")[0]
    return s


# ----------------------------------------------------------------------------- 1. LOSO
def run_final_loso(dfs: dict, sp: dict, onset: dict, n_jobs: int) -> tuple:
    order = ["R", "D", "R_ext", "D_ext"]
    tasks = [(dfs[k.split("_")[0]], sp[k], WIN, "binary", onset, k, None) for k in order]
    jobs = run_parallel(tasks, n_jobs)
    for j in jobs:
        p = j["pred"]
        p["pred_s45"] = smooth(p.pred_op1, p.subject.values, p.t_start_s.values, *SMOOTH_MN)
        j["res"] = evaluate_ops(p, onset, ALL_OPS)
    spec_by = {(sp[k].key, k): sp[k] for k in order}
    res, per, folds = tabulate(jobs, spec_by, ops=ALL_OPS)
    for t in (res, per, folds):
        t.insert(0, "config", t.part.str.split("_").str[0])
        t["role"] = np.where(t.part.str.endswith("_ext"), "POST-HOC sensitivity (extended C grid; not used for selection)",
                             "final")
    for t in (res, per):
        t["operating_point_default"] = t.op == DEFAULT_OP
    return {j["tag"]: j for j in jobs}, res, per, folds


def reproduction_checks(jobs: dict) -> dict:
    """R must reproduce the Step C grid row, D the Step C G3 'redefined, clean' row (pre-stated in NOTES.md)."""
    out = {}
    pg = pd.read_csv(METRICS / "stepc_per_subject_grid.csv")
    key_r = jobs["R"]["key"]
    old_r = pg[(pg.key == key_r) & (pg.window_s == WIN) & (pg.task == "binary")]
    pf = pd.read_csv(METRICS / "stepc_per_subject_followup.csv")
    old_d = pf[(pf.part == "G3_clean_1") & (pf.scheme == "relative_only")]
    assert len(old_d.display.unique()) == 1 and "redef clean" in old_d.display.iloc[0], old_d.display.unique()
    for c, old in (("R", old_r), ("D", old_d)):
        for op_old, op_new in (("OP1", "OP1"), ("OP2_spec90", "OP2_spec90"), ("smoothed", "OP1_smooth_inner_mn_reference")):
            a = old[old.op == op_old].set_index("subject").f1.sort_index()
            b = jobs[c]["res"][op_new]["per"].set_index("subject").f1.sort_index()
            assert len(a) == 15 and (a.index == b.index).all(), f"{c} {op_old}: subject sets differ"
            d = float(np.max(np.abs(a.values - b.values)))
            assert d <= 1e-9, f"STOP: {c} {op_old} does not reproduce Step C (max per-subject F1 diff {d})"
            out[f"{c}_{op_old}_max_abs_f1_diff_vs_stepc"] = d
    ps = pd.read_csv(METRICS / "stepc_predictions_selected.csv")
    ps = ps[(ps.key == key_r) & (ps.window_s == WIN)].sort_values(["subject", "t_start_s"])
    new = jobs["R"]["pred"].sort_values(["subject", "t_start_s"])
    assert len(ps) == len(new)
    d = float(np.max(np.abs(ps.score.values - new.score.values)))
    assert d <= 1e-12 and (ps.pred_op1.values == new.pred_op1.values).all(), f"STOP: R scores differ from Step C ({d})"
    out["R_max_abs_score_diff_vs_stepc"] = d
    return out


# ----------------------------------------------------------------------------- 2. train vs validation (inner folds)
def _fold_job(df_all, spec, test_s: str, c_chosen: float, seed: int) -> tuple:
    sc = df_all[df_all.scored]
    train = sc[sc.subject != test_s]
    y = train.y.values
    checks.assert_no_subject_overlap(set(train.subject), {test_s})
    checks.assert_no_calibration_rows(train, "training set")
    rng = np.random.default_rng(seed)
    vc, lc = [], []
    for j, (itr, iva) in enumerate(GroupKFold(n_splits=INNER_FOLDS).split(train, groups=train.subject)):
        tr_s, va_s = set(train.subject.iloc[itr]), set(train.subject.iloc[iva])
        checks.assert_no_subject_overlap(tr_s, va_s)
        assert test_s not in tr_s | va_s
        X = prepare_ext(df_all, train, spec, tr_s)
        for params in spec.grid:
            m = fit_checked(spec, X.iloc[itr], y[itr], params, tr_s, va_s)
            s_tr, s_va = spec.score(m, X.iloc[itr]), spec.score(m, X.iloc[iva])
            thr = f1_threshold(y[itr], s_tr)
            vc.append({"outer_test_subject": test_s, "inner_fold": j, "C": params["C"],
                       "pr_auc_train": average_precision_score(y[itr], s_tr),
                       "pr_auc_val": average_precision_score(y[iva], s_va),
                       "roc_auc_train": roc_auc_score(y[itr], s_tr), "roc_auc_val": roc_auc_score(y[iva], s_va),
                       "f1_train": f1_score(y[itr], s_tr >= thr), "f1_val": f1_score(y[iva], s_va >= thr)})
        subs = sorted(tr_s, key=lambda s: int(s[1:]))
        sub_tr = train.subject.values[itr]
        for n in LC_SIZES:
            for draw in range(LC_DRAWS if n < len(subs) else 1):
                pick = set(rng.choice(subs, n, replace=False))
                idx = itr[np.isin(sub_tr, list(pick))]
                m = fit_checked(spec, X.iloc[idx], y[idx], {"C": c_chosen}, pick, va_s)
                s_tr, s_va = spec.score(m, X.iloc[idx]), spec.score(m, X.iloc[iva])
                lc.append({"outer_test_subject": test_s, "inner_fold": j, "n_train_subjects": n, "draw": draw,
                           "C": c_chosen, "pr_auc_train": average_precision_score(y[idx], s_tr),
                           "pr_auc_val": average_precision_score(y[iva], s_va),
                           "roc_auc_train": roc_auc_score(y[idx], s_tr), "roc_auc_val": roc_auc_score(y[iva], s_va)})
    return vc, lc


def train_val_curves(dfs, sp, folds, n_jobs) -> tuple:
    from joblib import Parallel, delayed
    tasks = []
    for c in ("R", "D"):
        f = folds[folds.part == c].set_index("test_subject")
        for i, s in enumerate(sorted(f.index, key=lambda x: int(x[1:]))):
            tasks.append((c, s, ast.literal_eval(f.loc[s, "params"])["C"], i))
    out = Parallel(n_jobs=n_jobs)(delayed(_fold_job)(dfs[c], sp[c], s, cc, seed) for c, s, cc, seed in tasks)
    vc = pd.concat([pd.DataFrame(v).assign(config=c) for (c, *_), (v, _) in zip(tasks, out)], ignore_index=True)
    lc = pd.concat([pd.DataFrame(l).assign(config=c) for (c, *_), (_, l) in zip(tasks, out)], ignore_index=True)
    return vc, lc


# ----------------------------------------------------------------------------- 3. deployment model (all 15)
def fit_deployment(df: pd.DataFrame, spec) -> dict:
    sc = df[df.scored]
    subj = set(sc.subject)
    assert len(subj) == 15
    checks.assert_no_calibration_rows(sc, "deployment training set")
    checks.assert_no_time_feature(spec.features)
    fs.assert_spec_order(spec.features)
    y = sc.y.values
    info = tune_ext(df, sc, y, spec)                                   # inner GroupKFold(7) over all 15 subjects
    X = prepare_ext(df, sc, spec, subj)
    model = fit_checked(spec, X, y, info["params"], subj, set())
    s = spec.score(model, X)
    resub = {"pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s)),
             "f1_op1": float(f1_score(y, s >= info["thr_op1"]))}
    return {"model": model, "info": info, "resubstitution_NOT_an_estimate": resub, "n_windows": int(len(sc))}


def folded_coefficients(model) -> dict:
    imp, scl, lr = (model.named_steps[k] for k in ("impute", "scale", "model"))
    med, mu, sd = imp.statistics_, scl.mean_, scl.scale_
    w, b = lr.coef_[0], float(lr.intercept_[0])
    return {"imputer_medians": med.tolist(), "scaler_means": mu.tolist(), "scaler_sds": sd.tolist(),
            "weights_standardised": w.tolist(), "bias_standardised": b,
            "weights_folded": (w / sd).tolist(), "bias_folded": float(b - np.sum(w * mu / sd))}


# ----------------------------------------------------------------------------- 4. save
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:                                        # read-only
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def calibration_info(df: pd.DataFrame) -> dict:
    g = pd.read_csv(METRICS / "stepc_eda_gain_offset.csv")
    warm = df.groupby("subject").calib_start_s.first()
    return {
        "duration_s": pp.CALIB_MIN * 60,
        "what_the_wearer_does": [
            "Put the device on (GSR electrodes, PPG sensor) and keep it on for the whole session.",
            "Let the filters run before calibrating: in WESAD the causal filters had run for "
            f"{warm.min():.0f}-{warm.max():.0f} s before the calibration period began; at least 60 s "
            "(> 5 x the 8 s EMA tau) is RECOMMENDED but this warm-up rule is untested.",
            "Sit still and at rest for 5 minutes: seated, relaxed, hand still, no talking, no walking, no "
            "eating/drinking, comfortable room (WESAD baseline: seated, reading neutral magazines; the "
            "calibration period is its first 5 min).",
            "Do not start calibration during or straight after exercise or a stressful event.",
        ],
        "statistics_computed": {
            "sigma_lp_cal": "population SD (ddof 0) of the causally low-passed EDA samples (1 Hz Butterworth "
                            "order 2 at 4 Hz, SOS state carried from power-on) over the 5-min calibration period "
                            "(1200 samples at 4 Hz). Device units (mV) are fine: only ratios are used.",
            "sigma_ph_cal": "population SD (ddof 0) of the phasic EDA (low-passed EDA minus causal EMA tonic, "
                            "tau 8 s) over the same 1200 samples.",
        },
        "used_by": {
            "eda_slope_rel": "least-squares slope (per s) of the low-passed EDA over the window / sigma_lp_cal",
            "scr_count_rel": f"number of scipy.signal.find_peaks(phasic window, prominence >= {SCR_REL_K} x "
                             "sigma_ph_cal, distance >= 4 samples = 1 s) in the window",
            "hr_mean": "no calibration",
        },
        "stored_on_device": ["sigma_lp_cal", "sigma_ph_cal"],
        "validity_rule_UNTESTED": "repeat calibration if either sigma is 0 or not finite; there is no validated "
                                  "lower bound on the prototype",
        "wesad_reference_values_uS": {
            "sigma_lp_cal_min": float(g.sigma_lp_cal_clean.min()), "sigma_lp_cal_median": float(g.sigma_lp_cal_clean.median()),
            "sigma_lp_cal_max": float(g.sigma_lp_cal_clean.max()), "sigma_ph_cal_min": float(g.sigma_ph_cal_clean.min()),
            "sigma_ph_cal_median": float(g.sigma_ph_cal_clean.median()), "sigma_ph_cal_max": float(g.sigma_ph_cal_clean.max()),
            "note": "E4 wrist EDA in uS; NOT comparable to finger mV on the prototype",
        },
        "pipeline_note": "In WESAD the calibration period is the first 5 min of the baseline condition; the same "
                         "stream is used afterwards, so the filters are NOT restarted between calibration and use.",
    }


def calibration_markdown(cal: dict) -> str:
    L = ["# Calibration procedure (deployment model D)", "",
         f"Duration: {cal['duration_s']} s (5 min), once at first use (and whenever the electrodes are re-placed).", "",
         "## What the wearer does"] + [f"- {x}" for x in cal["what_the_wearer_does"]] + \
        ["", "## Statistics computed from the 5-min period"] + \
        [f"- `{k}`: {v}" for k, v in cal["statistics_computed"].items()] + \
        ["", "## Where they are used"] + [f"- `{k}`: {v}" for k, v in cal["used_by"].items()] + \
        ["", f"Stored on the device: {', '.join(cal['stored_on_device'])}.", "",
         f"Validity (UNTESTED): {cal['validity_rule_UNTESTED']}.", "",
         "WESAD reference values (E4 wrist, uS; not comparable to the prototype's finger mV): "
         + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k}: {v}" for k, v in cal["wesad_reference_values_uS"].items()),
         "", cal["pipeline_note"], "",
         "Exact filter and feature definitions: `feature_spec.json`. Nothing here is validated on the prototype."]
    return "\n".join(L) + "\n"


def save_models(dep: dict, spec, df: pd.DataFrame, data_root: Path, final_res: pd.DataFrame, repro: dict) -> dict:
    MODELS.mkdir(exist_ok=True)
    info = dep["info"]
    coef = folded_coefficients(dep["model"])
    thr1, thr2 = float(info["thr_op1"]), float(info["thr_op2"])
    cal = calibration_info(df)
    export = {
        "model": "logistic regression (stress detection; WESAD lab stressor, NOT panic/anxiety)",
        "config": "D", "feature_spec_version": fs.VERSION, "feature_order": list(spec.features),
        "window_s": WIN, "step_s": pp.STEP_S,
        "score": "x_i <- imputer_medians[i] if x_i is NaN; z = bias_folded + sum_i weights_folded[i] * x_i; "
                 "p = 1 / (1 + exp(-z)); window alert if p >= threshold (equivalently z >= logit_threshold)",
        **coef,
        "C": float(info["params"]["C"]),
        "threshold_op1": thr1, "logit_threshold_op1": float(np.log(thr1 / (1 - thr1))),
        "threshold_op2": thr2, "logit_threshold_op2": float(np.log(thr2 / (1 - thr2))),
        "default_operating_point": "OP1 + 4-of-5 smoothing",
        "smoothing": {"m": SMOOTH_MN[0], "n": SMOOTH_MN[1],
                      "rule": "alert at window t if >= m of the OP1 window decisions at t, t-5 s, ..., t-5(n-1) s are "
                              "positive (causal; windows not yet available count as negative)"},
        "inner_cv_choice_of_m_of_n_not_used": list(info["mn"]),
        "calibration": cal,
        "input_rates_hz": {"BVP": pp.FS["BVP"], "EDA": pp.FS["EDA"]},
        "expected_performance": "the LOSO result of config D (results/metrics/final_results.csv); this fitted model "
                                "has no held-out estimate",
    }
    jpath = MODELS / f"{MODEL_STEM}.json"
    jpath.write_text(json.dumps(export, indent=2), encoding="utf-8")
    bundle = {"pipeline": dep["model"], "feature_order": list(spec.features), "feature_spec_version": fs.VERSION,
              "window_s": WIN, "step_s": pp.STEP_S, "threshold_op1": thr1, "threshold_op2": thr2,
              "smoothing_m_of_n": list(SMOOTH_MN)}
    kpath = MODELS / f"{MODEL_STEM}.joblib"
    joblib.dump(bundle, kpath)
    spath = fs.write(MODELS / "feature_spec.json")
    assert json.loads(spath.read_text()) == json.loads(fs.SPEC_JSON.read_text()), \
        "models/feature_spec.json differs from the module-root feature_spec.json"
    (MODELS / "calibration_procedure.md").write_text(calibration_markdown(cal), encoding="utf-8")
    (MODELS / "window.json").write_text(json.dumps({"window_s": WIN, "step_s": pp.STEP_S, **fs.WINDOW,
                                                    "smoothing_m_of_n": list(SMOOTH_MN)}, indent=2), encoding="utf-8")

    # metadata
    print("  hashing data files (read-only) ...", flush=True)
    data_files = [{"file": str(p), "sha256": sha256(p), "bytes": p.stat().st_size} for _, p in pp.find_subjects(data_root)]
    assert len(data_files) == 15, f"expected 15 WESAD .pkl files, found {len(data_files)}"
    derived = [METRICS / f"features_w{WIN}.csv", METRICS / f"features_eda_w{WIN}.csv",
               METRICS / "stepc_eda_gain_offset.csv", METRICS / "stepc_condition_order.csv"]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    keep = ["config", "op", "f1_mean", "f1_sd", "f1_ci_low", "f1_ci_high", "pr_auc_mean", "pr_auc_ci_low",
            "pr_auc_ci_high", "roc_auc_mean", "sensitivity_mean", "specificity_mean", "precision_mean",
            "balanced_accuracy_mean", "fa_windows_per_h_mean", "fa_events_per_h_mean", "worst_subject", "worst_f1",
            "latency_median_s"]
    meta = {
        "created": dt.datetime.now().isoformat(timespec="seconds"), "git_head_before_this_commit": head,
        "python": sys.version, "platform": platform.platform(),
        "libraries": {"numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
                      "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "joblib": joblib.__version__,
                      "matplotlib": matplotlib.__version__},
        "data_files": data_files,
        "derived_feature_files": [{"file": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in derived],
        "model_files": [{"file": p.name, "sha256": sha256(p)} for p in (jpath, kpath, spath)],
        "deployment_model": {
            "config": "D", "features": list(spec.features), "window_s": WIN, "step_s": pp.STEP_S,
            "trained_on": "all 15 WESAD subjects, scored windows only (calibration windows excluded)",
            "n_training_windows": dep["n_windows"], "inner_cv": f"GroupKFold({INNER_FOLDS}) by subject over all 15",
            "C_chosen": float(info["params"]["C"]), "C_grid": C_GRID, "C_at_grid_edge": info["params"]["C"] == min(C_GRID),
            "inner_mean_pr_auc": float(info["inner_pr_auc"]), "inner_oof_f1_op1": float(info["inner_f1_op1"]),
            "threshold_op1": thr1, "threshold_op2": thr2,
            "inner_cv_m_of_n_not_used": list(info["mn"]), "shipped_m_of_n": list(SMOOTH_MN),
            "resubstitution_metrics_NOT_an_estimate": dep["resubstitution_NOT_an_estimate"],
            "held_out_estimate": "NONE. Expected performance = LOSO result of config D below.",
        },
        "loso_metrics": final_res[keep].to_dict(orient="records"),
        "reproduction_of_step_c": repro,
        "not_validated": "no prototype data (finger EDA in mV, fingertip PPG); IEC approval pending",
    }
    (MODELS / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return export


# ----------------------------------------------------------------------------- 5. round trip
def write_roundtrip(dep: dict, spec, df: pd.DataFrame, export: dict) -> dict:
    sc = df[df.scored]
    rng = np.random.default_rng(0)
    nan_idx = sc.index[sc[spec.features].isna().any(axis=1)]
    pick = np.sort(np.r_[rng.choice(sc.index.difference(nan_idx), 300, replace=False), nan_idx[:30]])
    rows = sc.loc[pick, ["subject", "t_start_s"] + spec.features].copy()
    rows["synthetic_nan"] = 0
    # No scored 30 s window has a missing D feature, so SYNTHETIC rows (copies of real windows with features set to
    # NaN, flagged synthetic_nan = 1) exercise the imputer path a device needs when, e.g., too few beats are found.
    syn = []
    for k, f in enumerate(spec.features + ["__all__"]):
        r = sc.loc[rng.choice(sc.index, 10, replace=False), ["subject", "t_start_s"] + spec.features].copy()
        r[spec.features if f == "__all__" else [f]] = np.nan
        syn.append(r.assign(synthetic_nan=1))
    rows = pd.concat([rows] + syn, ignore_index=True)
    p = spec.score(dep["model"], rows)
    rows["p_ref"] = p
    rows["alert_op1_ref"] = (p >= export["threshold_op1"]).astype(int)
    rows["alert_op2_ref"] = (p >= export["threshold_op2"]).astype(int)
    path = MODELS / "roundtrip_windows.csv"
    rows.to_csv(path, index=False, float_format="%.17g")
    back = pd.read_csv(path)                    # exactness asserted below
    for c in spec.features + ["p_ref"]:
        assert np.array_equal(back[c].values, rows[c].values, equal_nan=True), f"CSV round trip of {c} not exact"
    # in-memory numpy-from-JSON agreement on ALL scored windows (float64 and float32)
    from src.predict import alert, proba_json
    X = sc[spec.features]
    pm = spec.score(dep["model"], X)
    pj = proba_json(export, X)
    p32 = proba_json(export, X, np.float32)
    out = {"n_roundtrip_windows": int(len(rows)), "n_real_windows_with_nan": int(len(nan_idx[:30])),
           "n_synthetic_nan_rows": int(rows.synthetic_nan.sum()),
           "all_windows_json_max_abs_diff": float(np.max(np.abs(pj - pm))),
           "all_windows_json_op1_disagreements": int((alert(pj, export["threshold_op1"]) != alert(pm, export["threshold_op1"])).sum()),
           "all_windows_float32_op1_disagreements": int((alert(p32, np.float32(export["threshold_op1"])) != alert(pm, export["threshold_op1"])).sum()),
           "all_windows_float32_max_abs_prob_diff": float(np.max(np.abs(p32.astype(float) - pm))),
           "n_all_windows": int(len(X))}
    assert out["all_windows_json_max_abs_diff"] <= 1e-12 and out["all_windows_json_op1_disagreements"] == 0
    r = subprocess.run([sys.executable, "-m", "src.predict", "--roundtrip"], cwd=ROOT, capture_output=True, text=True)
    print(r.stdout.strip()); print(r.stderr.strip()) if r.returncode else None
    assert r.returncode == 0 and "ROUNDTRIP PASS" in r.stdout, "STOP: fresh-process round-trip test failed"
    out["fresh_process"] = json.loads(r.stdout.strip().split("ROUNDTRIP PASS ", 1)[1])
    return out


# ----------------------------------------------------------------------------- plots
def _title(ax_or_fig, text, fig=False):
    (ax_or_fig.suptitle if fig else ax_or_fig.set_title)(text, fontweight="bold", fontsize=9.5)


def plot_roc_pr(jobs, res):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4))
    for c in ("R", "D"):
        p = jobs[c]["pred"]
        fpr, tpr, _ = roc_curve(p.y, p.score)
        a1.plot(fpr, tpr, color=COL[c], linewidth=2, label=f"{c}  pooled AUC {roc_auc_score(p.y, p.score):.3f}")
        pr, rc, _ = precision_recall_curve(p.y, p.score)
        a2.plot(rc, pr, color=COL[c], linewidth=2, label=f"{c}  pooled AP {average_precision_score(p.y, p.score):.3f}")
        for op, mk in (("OP1", "o"), ("OP2_spec90", "^"), ("OP1_smooth4of5", "D")):
            r = res[(res.part == c) & (res.op == op)].iloc[0]
            sens = r.pooled_sensitivity; spec_ = r.pooled_specificity; prec = r.pooled_precision
            a1.scatter(1 - spec_, sens, marker=mk, s=60, color=COL[c], edgecolors="#fcfcfb", linewidths=2, zorder=4)
            a2.scatter(sens, prec, marker=mk, s=60, color=COL[c], edgecolors="#fcfcfb", linewidths=2, zorder=4)
    for mk, lab in (("o", "OP1"), ("^", "OP2 (spec ≥ 0.90)"), ("D", "OP1 + 4-of-5 (default)")):
        a1.scatter([], [], marker=mk, color=MUTED, label=f"pooled point: {lab}")
    a1.plot([0, 1], [0, 1], color=MUTED, linestyle=":", linewidth=1)
    prev = jobs["R"]["pred"].y.mean()
    a2.axhline(prev, color=MUTED, linestyle=":", linewidth=1, label=f"chance (prevalence {prev:.2f})")
    a1.set(xlabel="false positive rate (1 − specificity)", ylabel="sensitivity", title="ROC")
    a2.set(xlabel="recall (sensitivity)", ylabel="precision", ylim=(0, 1.02), title="Precision-recall")
    a1.legend(loc="lower right", fontsize=7.5); a2.legend(loc="lower left", fontsize=7.5)
    _title(fig, "Stress detection: pooled LOSO ROC and PR curves, LR 30 s, R vs D "
                "(15 held-out subjects, each fold its own model)", fig=True)
    save_tight(fig, "final_roc_pr.png")


def plot_calibration(jobs):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle=":", linewidth=1, label="perfect calibration")
    rows = []
    for c in ("R", "D"):
        p = jobs[c]["pred"]
        fr, mp = calibration_curve(p.y, p.score, n_bins=10, strategy="quantile")
        ax.plot(mp, fr, color=COL[c], marker=MARK[c], linewidth=2, markersize=6, label=c)
        rows += [{"config": c, "bin": i, "mean_predicted": a, "observed_stress_fraction": b} for i, (a, b) in enumerate(zip(mp, fr))]
    ax.set(xlabel="mean predicted probability (quantile bins)", ylabel="observed stress fraction",
           xlim=(0, 1), ylim=(0, 1), title="Reliability (pooled held-out windows)")
    ax.legend(fontsize=7.5, loc="upper left")
    bins = np.linspace(0, 1, 41)
    for ax, c in zip(axes[1:], ("R", "D")):
        p = jobs[c]["pred"]
        ax.hist(p.score[p.y == 0], bins=bins, color=MUTED, alpha=0.6, label="non-stress windows")
        ax.hist(p.score[p.y == 1], bins=bins, histtype="step", color=COL[c], linewidth=2, label="stress windows")
        ax.set(xlabel="predicted probability", ylabel="windows", title=f"Score distribution, {c}")
        ax.legend(fontsize=7.5)
    _title(fig, "Stress detection: calibration of LOSO probabilities (class-balanced LR: probabilities are NOT "
                "calibrated to the 35 % stress prevalence)", fig=True)
    save_tight(fig, "final_calibration.png")
    pd.DataFrame(rows).to_csv(METRICS / "final_calibration_curve.csv", index=False)


def _cm_axes(ax, tp, fn, fp, tn, title, fs_=9):
    m = np.array([[tn, fp], [fn, tp]], float)
    rate = m / m.sum(axis=1, keepdims=True)
    ax.imshow(rate, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(m[i, j])}\n{rate[i, j]:.0%}", ha="center", va="center", fontsize=fs_,
                    color="#fcfcfb" if rate[i, j] > 0.6 else INK)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.grid(False)
    ax.set_xticklabels(["pred non-stress", "pred stress"], fontsize=fs_ - 2)
    ax.set_yticklabels(["non-stress", "stress"], fontsize=fs_ - 2)
    ax.set_title(title, fontsize=fs_)


def plot_confusion(res, per):
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6))
    for i, c in enumerate(("R", "D")):
        for j, op in enumerate(MAIN_OPS):
            r = res[(res.part == c) & (res.op == op)].iloc[0]
            _cm_axes(axes[i, j], r.pooled_tp, r.pooled_fn, r.pooled_fp, r.pooled_tn, f"{c}, {op}")
    _title(fig, "Stress detection: pooled confusion matrices, LOSO 30 s (rows = true class; % of the row)", fig=True)
    save_tight(fig, "final_confusion_pooled.png")
    for c in ("R", "D"):
        fig, axes = plt.subplots(3, 5, figsize=(14, 8.4))
        g = per[(per.part == c) & (per.op == DEFAULT_OP)].set_index("subject")
        for ax, s in zip(axes.ravel(), sorted(g.index, key=lambda x: int(x[1:]))):
            r = g.loc[s]
            _cm_axes(ax, r.tp, r.fn, r.fp, r.tn, f"{s}  F1 {r.f1:.2f}", fs_=8)
        _title(fig, f"Stress detection: per-subject confusion matrices, {NAMES[c]}, default operating point "
                    f"(OP1 + 4-of-5), LOSO 30 s", fig=True)
        save_tight(fig, f"final_confusion_per_subject_{c}.png")


def plot_spread(per):
    metrics = ["f1", "sensitivity", "specificity", "precision", "pr_auc", "roc_auc", "fa_events_per_h"]
    ops = ["OP1", DEFAULT_OP]
    fig, axes = plt.subplots(len(ops), len(metrics), figsize=(16, 6.8))
    for i, op in enumerate(ops):
        for j, m in enumerate(metrics):
            ax = axes[i, j]
            a = per[(per.part == "R") & (per.op == op)].set_index("subject")[m]
            b = per[(per.part == "D") & (per.op == op)].set_index("subject")[m].reindex(a.index)
            for s in a.index:
                ax.plot([0, 1], [a[s], b[s]], color="#c3c2b7", linewidth=0.8, zorder=1)
            for k, (c, v) in enumerate((("R", a), ("D", b))):
                ax.scatter(np.full(len(v), k), v, color=COL[c], marker=MARK[c], s=22, zorder=3,
                           edgecolors="#fcfcfb", linewidths=0.8)
                lo, hi = ev.bootstrap_ci(v.values)
                ax.errorbar(k + 0.18, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], fmt="_", color=INK,
                            markersize=10, capsize=3, linewidth=1.2, zorder=4)
                worst = v.idxmax() if m == "fa_events_per_h" else v.idxmin()
                ax.annotate(worst, (k, v[worst]), xytext=(-14, 0), textcoords="offset points", fontsize=6.5,
                            color=INK2, ha="right", va="center")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["R", "D"]); ax.set_xlim(-0.6, 1.5); ax.grid(axis="x", visible=False)
            ax.set_title(m if i == 0 else "", fontsize=9)
            if j == 0:
                ax.set_ylabel(op, fontsize=9)
    _title(fig, "Stress detection: per-subject spread, LOSO 30 s (dots = held-out subjects, grey lines pair the same "
                "subject, bar = mean with subject-bootstrap 95 % CI, label = worst subject); "
                "threshold-free AUCs equal in both rows", fig=True)
    fig.tight_layout()
    save_tight(fig, "final_per_subject_spread.png")


def plot_worst(jobs, folds):
    fig, axes = plt.subplots(len(WORST), 2, figsize=(13, 2.6 * len(WORST)), sharey=True)
    names = {1: "baseline", 2: "stress", 3: "amusement"}
    marks = {1: "o", 2: "^", 3: "s"}
    for i, s in enumerate(WORST):
        for j, c in enumerate(("R", "D")):
            ax = axes[i, j]
            p = jobs[c]["pred"]; p = p[p.subject == s]
            fr = folds[(folds.part == c) & (folds.test_subject == s)].iloc[0]
            for lab in (1, 2, 3):
                g = p[p.label == lab]
                ax.scatter(g.t_start_s / 60, g.score, s=7, color=CLASS_COLORS[lab], marker=marks[lab],
                           label=names[lab] if (i, j) == (0, 0) else None, zorder=3)
            ax.axhline(fr.thr_op1, color=INK, linewidth=1, linestyle="--", label="OP1 threshold" if (i, j) == (0, 0) else None)
            ax.axhline(fr.calib_score_mean, color=INK2, linewidth=1, linestyle=":",
                       label="mean score, own calibration rest" if (i, j) == (0, 0) else None)
            al = p[p.pred_s45 == 1]
            ax.scatter(al.t_start_s / 60, np.full(len(al), -0.04), marker="|", s=30, color=INK,
                       label="alert (OP1 + 4-of-5)" if (i, j) == (0, 0) else None)
            ax.set_ylim(-0.08, 1.02)
            ax.set_title(f"{s}, {c}: OP1 sens {(p[p.y == 1].pred_op1).mean():.2f}, spec {1 - (p[p.y == 0].pred_op1).mean():.2f}",
                         fontsize=9)
            if j == 0:
                ax.set_ylabel("P(stress)")
            if i == len(WORST) - 1:
                ax.set_xlabel("minutes from recording start (scored windows only)")
    fig.legend(loc="lower center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    _title(fig, "Stress detection: held-out score time course for the worst subjects (S13, S2, S3), LOSO 30 s", fig=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_tight(fig, "final_worst_subjects.png")


def plot_train_val(vc, lc):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    for c in ("R", "D"):
        g = vc[vc.config == c].groupby("C")
        for k, (m, ax) in enumerate((("pr_auc", axes[0]), ("f1", axes[1]))):
            for part, ls in (("train", "-"), ("val", "--")):
                mu, sd = g[f"{m}_{part}"].mean(), g[f"{m}_{part}"].std()
                xs = mu.index * (0.85 if c == "R" else 1.15) * (0.93 if part == "train" else 1.07)   # offset only
                ax.errorbar(xs, mu.values, yerr=sd.values, color=COL[c], linestyle=ls, marker=MARK[c],
                            markersize=5, capsize=2, linewidth=1.6, elinewidth=0.8, alpha=0.9,
                            label=f"{c} {'inner-train' if part == 'train' else 'inner-validation'}")
        h = lc[lc.config == c].groupby("n_train_subjects")
        for part, ls in (("train", "-"), ("val", "--")):
            mu, sd = h[f"pr_auc_{part}"].mean(), h[f"pr_auc_{part}"].std()
            xs = mu.index + (-0.3 if c == "R" else 0.3) + (-0.1 if part == "train" else 0.1)      # offset only
            axes[2].errorbar(xs, mu.values, yerr=sd.values, color=COL[c], linestyle=ls, marker=MARK[c],
                             markersize=5, capsize=2, linewidth=1.6, elinewidth=0.8, alpha=0.9,
                             label=f"{c} {'train' if part == 'train' else 'validation'}")
    axes[0].set(xscale="log", xlabel="C (pre-registered grid; smaller = stronger L2)", ylabel="PR-AUC (pooled per inner fold)",
                title="Validation curve: PR-AUC")
    axes[1].set(xscale="log", xlabel="C", ylabel="F1 (threshold = max F1 on inner-train)",
                title="Validation curve: F1 at the transferred threshold")
    axes[2].set(xlabel="number of training subjects (inner-train subset)", ylabel="PR-AUC",
                title="Learning curve at the fold's chosen C")
    axes[2].set_xticks(LC_SIZES)
    for ax in axes:
        ax.legend(fontsize=7)
    _title(fig, "Stress detection: training vs validation inside the inner folds only (15 outer x 7 inner folds; "
                "mean ± SD), LR 30 s, R and D\nTrain = pooled over the inner-training subjects, validation = pooled "
                "over the 2 inner-validation subjects, so pooled PR-AUC levels are not strictly comparable;\n"
                "points offset horizontally for legibility", fig=True)
    fig.tight_layout()
    save_tight(fig, "final_train_vs_validation.png")


# ----------------------------------------------------------------------------- worst-subject table
def worst_table(per, folds, dfs, preds: dict) -> pd.DataFrame:
    g = pd.read_csv(METRICS / "stepc_eda_gain_offset.csv").set_index("subject")
    rows = []
    for c in ("R", "D"):
        for s in WORST:
            fr = folds[(folds.part == c) & (folds.test_subject == s)].iloc[0]
            df = dfs[c]
            cal = df[df.calib & (df.subject == s)]
            ps = preds[c][preds[c].subject == s]
            for op in MAIN_OPS:
                r = per[(per.part == c) & (per.op == op) & (per.subject == s)].iloc[0]
                rows.append({"config": c, "subject": s, "op": op, **{k: r[k] for k in (
                    "tp", "fn", "fp", "tn", "sensitivity", "specificity", "precision", "f1", "roc_auc", "pr_auc",
                    "fa_events_per_h", "latency_s")},
                    "thr_op1": fr.thr_op1, "thr_op2": fr.thr_op2, "params": fr.params,
                    "calib_score_mean": fr.calib_score_mean,
                    **{f"median_score_{n}": float(ps[ps.label == lab].score.median())
                       for lab, n in ((1, "baseline"), (2, "stress"), (3, "amusement"))},
                    "calib_hr_mean_median_bpm": float(cal.hr_mean.median()),
                    "sigma_lp_cal_uS": g.loc[s, "sigma_lp_cal_clean"], "sigma_ph_cal_uS": g.loc[s, "sigma_ph_cal_clean"]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="WESAD root (read-only; used for the .pkl checksums)")
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    np.random.seed(0)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 200)
    dfs, sp, onset = load_data(), specs(), onset_map()
    assert sp["D"].features == FEATS_D == ["hr_mean", "eda_slope_rel", "scr_count_rel"]

    print("1. final LOSO (R, D, post-hoc extended C grid) ...", flush=True)
    jobs, res, per, folds = run_final_loso(dfs, sp, onset, a.jobs)
    repro = reproduction_checks(jobs)
    print("   reproduction of Step C: PASS", json.dumps(repro))
    main_rows = res[(res.role == "final") & res.op.isin(MAIN_OPS)].copy()
    supp = res[~res.index.isin(main_rows.index)].copy()
    main_rows.to_csv(METRICS / "final_results.csv", index=False)
    supp.to_csv(METRICS / "final_results_supplement.csv", index=False)
    per.to_csv(METRICS / "final_per_subject.csv", index=False)
    folds.to_csv(METRICS / "final_folds.csv", index=False)
    pd.concat([jobs[c]["pred"].assign(config=c) for c in ("R", "D")]).to_csv(METRICS / "final_predictions.csv", index=False)
    wt = worst_table(per, folds, dfs, {c: jobs[c]["pred"] for c in ("R", "D")})
    wt.to_csv(METRICS / "final_worst_subjects.csv", index=False)

    print("2. train vs validation (inner folds only) ...", flush=True)
    vc, lc = train_val_curves(dfs, sp, folds, a.jobs)
    vc.to_csv(METRICS / "final_train_val_curve.csv", index=False)
    lc.to_csv(METRICS / "final_learning_curve.csv", index=False)

    print("3. plots ...", flush=True)
    plot_roc_pr(jobs, res); plot_calibration(jobs); plot_confusion(res, per); plot_spread(per)
    plot_worst(jobs, folds); plot_train_val(vc, lc)

    print("4. deployment model D on all 15 subjects ...", flush=True)
    dep = fit_deployment(dfs["D"], sp["D"])
    export = save_models(dep, sp["D"], dfs["D"], Path(a.data), main_rows, repro)
    print("5. round trip ...", flush=True)
    rt = write_roundtrip(dep, sp["D"], dfs["D"], export)
    summary = {"reproduction": repro, "deployment": {k: v for k, v in export.items() if k in (
        "feature_order", "C", "threshold_op1", "threshold_op2", "weights_folded", "bias_folded", "imputer_medians",
        "scaler_means", "scaler_sds", "weights_standardised", "bias_standardised", "inner_cv_choice_of_m_of_n_not_used")},
        "deployment_inner": {"inner_mean_pr_auc": dep["info"]["inner_pr_auc"], "inner_oof_f1_op1": dep["info"]["inner_f1_op1"],
                             "resubstitution_NOT_an_estimate": dep["resubstitution_NOT_an_estimate"]},
        "roundtrip": rt}
    (METRICS / "final_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # outputs exist
    outs = [METRICS / n for n in ("final_results.csv", "final_results_supplement.csv", "final_per_subject.csv",
                                  "final_folds.csv", "final_predictions.csv", "final_worst_subjects.csv",
                                  "final_train_val_curve.csv", "final_learning_curve.csv", "final_calibration_curve.csv",
                                  "final_summary.json")]
    outs += [GRAPHS / n for n in ("final_roc_pr.png", "final_calibration.png", "final_confusion_pooled.png",
                                  "final_confusion_per_subject_R.png", "final_confusion_per_subject_D.png",
                                  "final_per_subject_spread.png", "final_worst_subjects.png", "final_train_vs_validation.png")]
    outs += [MODELS / n for n in (f"{MODEL_STEM}.joblib", f"{MODEL_STEM}.json", "feature_spec.json",
                                  "calibration_procedure.md", "window.json", "metadata.json", "roundtrip_windows.csv")]
    for p in outs:
        assert p.exists() and p.stat().st_size > 0, f"missing or empty output: {p}"
    print(f"All {len(outs)} output files exist and are non-empty.")

    show(res, folds, vc, lc, wt, summary)
    print("\n(checklist 1: results/metrics/final_results.csv; header text comes from train.metric_checklist)")
    metric_checklist(main_rows)
    print("\n(checklist 2: results/metrics/final_results_supplement.csv: post-hoc C grid + inner-m-of-n reference rows)")
    metric_checklist(supp)


def show(res, folds, vc, lc, wt, summary):
    cols = ["config", "role", "op", "f1_mean", "f1_sd", "f1_ci_low", "f1_ci_high", "pr_auc_mean", "pr_auc_ci_low",
            "pr_auc_ci_high", "roc_auc_mean", "accuracy_mean", "balanced_accuracy_mean", "precision_mean",
            "sensitivity_mean", "specificity_mean", "fa_windows_per_h_mean", "fa_events_per_h_mean",
            "pooled_tp", "pooled_fp", "pooled_tn", "pooled_fn", "worst_subject", "worst_f1", "latency_median_s",
            "n_missed_episodes", "params_mode"]
    print("\n=== Step D LOSO results (mean across 15 held-out subjects; SD; subject-bootstrap 95 % CI) ===")
    print(res[cols].round(3).to_string(index=False))
    print("\n=== CI per metric (final rows) ===")
    for _, r in res[res.role == "final"].iterrows():
        print(f"  {r.config} {r.op:32s} " + "  ".join(
            f"{m} {r[f'{m}_mean']:.3f} [{r[f'{m}_ci_low']:.3f}, {r[f'{m}_ci_high']:.3f}]" for m in ev.METRICS))
    print("\n=== chosen C per outer fold ===")
    print(folds.groupby(["part"]).params.value_counts().to_string())
    print("\n=== validation curve (inner folds; mean over 105) ===")
    print(vc.groupby(["config", "C"])[["pr_auc_train", "pr_auc_val", "roc_auc_train", "roc_auc_val", "f1_train", "f1_val"]]
          .mean().round(4).to_string())
    print("\n=== learning curve (inner folds) ===")
    print(lc.groupby(["config", "n_train_subjects"])[["pr_auc_train", "pr_auc_val", "roc_auc_train", "roc_auc_val"]]
          .agg(["mean", "std"]).round(4).to_string())
    print("\n=== worst subjects ===")
    print(wt.round(3).to_string(index=False))
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
