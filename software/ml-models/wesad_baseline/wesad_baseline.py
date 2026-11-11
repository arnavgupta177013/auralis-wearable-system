"""
Auralis — WESAD wrist baseline: universal vs personal-baseline comparison
==========================================================================

What this does
--------------
1. Loads the WESAD *wrist* (Empatica E4) signals for every subject:
   BVP 64 Hz, EDA 4 Hz, TEMP 4 Hz, ACC 32 Hz. These are the closest match
   to Auralis's sensors (PPG, GSR, skin temp, IMU).
2. Cuts them into sliding windows (default 60 s long, 15 s step) and keeps
   windows whose WESAD label is >= 90 % one condition.
3. Extracts features per window: HR, HRV (RMSSD, SDNN, pNN50), EDA level,
   slope and SCR count, temperature mean and slope, motion energy.
4. Compares four detectors under leave-one-subject-out (LOSO) evaluation:
     rule_universal   one fixed HR threshold for everybody (raw bpm)
     rule_personal    one threshold on the HR z-score vs the person's own rest
     rf_universal     Random Forest on raw features
     rf_personal      Random Forest on features z-scored vs the person's rest
   (+ xgb_* versions if xgboost is installed)
5. Saves results, per-subject scores, confusion matrices and plots.

Personal baseline = the first CALIB_MIN minutes of each subject's WESAD
"baseline" condition (seated rest). Those calibration windows are excluded
from scoring for EVERY method, so all four are scored on the same windows.

Labels
------
WESAD labels: 1 baseline, 2 stress (TSST), 3 amusement, 4 meditation.
Task "binary": stress (2) vs non-stress (1, 3)   [as in Schmidt et al. 2018]
Task "three":  baseline (1) vs stress (2) vs amusement (3)
WESAD stress is a lab stressor (public speaking + mental arithmetic).
It is NOT panic. Do not report these results as panic detection.

Usage (PowerShell, from the repo root)
--------------------------------------
    python software/ml-models/wesad_baseline/wesad_baseline.py --data "D:\\datasets\\WESAD"
    python software/ml-models/wesad_baseline/wesad_baseline.py --data "D:\\datasets\\WESAD" --task three

--data must point at the folder that contains S2, S3, ... S17.
Outputs go to software/ml-models/wesad_baseline/out/ unless --out is given.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------- constants
FS = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32}
FS_LABEL = 700                      # WESAD labels are at the chest (RespiBAN) rate
WIN_S = 60                          # window length, seconds
STEP_S = 15                         # window step, seconds
PURITY = 0.90                       # fraction of window that must share one label
CALIB_MIN = 5                       # minutes of baseline used as personal calibration
SEED = 42
FEATURES = [
    "hr_mean", "hr_std", "rmssd", "sdnn", "pnn50",
    "scl_mean", "scl_std", "scl_slope", "scr_count", "scr_amp_mean",
    "temp_mean", "temp_slope",
    "acc_mag_mean", "motion_energy",
]


# ----------------------------------------------------------------------------- loading
def load_subject(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    w = d["signal"]["wrist"]
    return {
        "BVP": np.asarray(w["BVP"], dtype=float).ravel(),
        "EDA": np.asarray(w["EDA"], dtype=float).ravel(),
        "TEMP": np.asarray(w["TEMP"], dtype=float).ravel(),
        "ACC": np.asarray(w["ACC"], dtype=float) / 64.0,    # E4 ACC unit is 1/64 g
        "label": np.asarray(d["label"]).ravel().astype(int),
    }


def find_subjects(root: Path) -> list:
    out = []
    for p in sorted(root.glob("S*"), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 999):
        pkl = p / f"{p.name}.pkl"
        if pkl.exists():
            out.append((p.name, pkl))
    return out


# ----------------------------------------------------------------------------- features
def bvp_features(bvp: np.ndarray) -> dict:
    fs = FS["BVP"]
    nan = dict(hr_mean=np.nan, hr_std=np.nan, rmssd=np.nan, sdnn=np.nan, pnn50=np.nan, ppg_quality=0.0)
    if len(bvp) < fs * 10:
        return nan
    b, a = sps.butter(3, [0.7, 3.5], btype="band", fs=fs)
    x = sps.filtfilt(b, a, bvp)
    # peaks at least 0.33 s apart (max 180 bpm), prominence relative to signal spread
    peaks, _ = sps.find_peaks(x, distance=int(0.33 * fs), prominence=0.3 * np.std(x))
    if len(peaks) < 5:
        return nan
    ibi = np.diff(peaks) / fs                                # seconds
    ibi = ibi[(ibi > 0.33) & (ibi < 1.5)]                    # 40-180 bpm
    if len(ibi) < 4:
        return nan
    med = np.median(ibi)
    good = np.abs(ibi - med) < 0.3 * med                     # reject artefact beats
    quality = float(good.mean())
    ibi = ibi[good]
    if len(ibi) < 4:
        return nan
    hr = 60.0 / ibi
    d = np.diff(ibi) * 1000.0                                # ms
    return dict(
        hr_mean=float(hr.mean()),
        hr_std=float(hr.std()),
        rmssd=float(np.sqrt(np.mean(d ** 2))) if len(d) else np.nan,
        sdnn=float(np.std(ibi * 1000.0)),
        pnn50=float(np.mean(np.abs(d) > 50)) if len(d) else np.nan,
        ppg_quality=quality,
    )


def eda_features(eda: np.ndarray) -> dict:
    fs = FS["EDA"]
    t = np.arange(len(eda)) / fs
    b, a = sps.butter(2, 1.0, btype="low", fs=fs)
    x = sps.filtfilt(b, a, eda) if len(eda) > 15 else eda
    tonic = uniform_filter1d(x, size=max(3, int(8 * fs)), mode="nearest")   # ~8 s moving average
    phasic = x - tonic
    pk, props = sps.find_peaks(phasic, prominence=0.01, distance=int(1 * fs))
    slope = np.polyfit(t, x, 1)[0] if len(x) > 2 else np.nan
    return dict(
        scl_mean=float(tonic.mean()),
        scl_std=float(x.std()),
        scl_slope=float(slope),                               # µS per second
        scr_count=float(len(pk)),
        scr_amp_mean=float(props["prominences"].mean()) if len(pk) else 0.0,
    )


def temp_features(temp: np.ndarray) -> dict:
    t = np.arange(len(temp)) / FS["TEMP"]
    slope = np.polyfit(t, temp, 1)[0] if len(temp) > 2 else np.nan
    return dict(temp_mean=float(temp.mean()), temp_slope=float(slope))      # °C, °C/s


def acc_features(acc: np.ndarray) -> dict:
    mag = np.linalg.norm(acc, axis=1)
    return dict(acc_mag_mean=float(mag.mean()), motion_energy=float(mag.std()))


def window_label(lab: np.ndarray):
    vals, counts = np.unique(lab, return_counts=True)
    i = counts.argmax()
    return int(vals[i]), counts[i] / len(lab)


def extract_subject(name: str, s: dict) -> pd.DataFrame:
    dur = len(s["label"]) / FS_LABEL
    rows = []
    t0 = 0.0
    while t0 + WIN_S <= dur:
        t1 = t0 + WIN_S
        lab, purity = window_label(s["label"][int(t0 * FS_LABEL):int(t1 * FS_LABEL)])
        if purity >= PURITY and lab in (1, 2, 3):
            seg = {k: s[k][int(t0 * FS[k]):int(t1 * FS[k])] for k in FS}
            r = {"subject": name, "t_start_s": t0, "label": lab}
            r.update(bvp_features(seg["BVP"]))
            r.update(eda_features(seg["EDA"]))
            r.update(temp_features(seg["TEMP"]))
            r.update(acc_features(seg["ACC"]))
            rows.append(r)
        t0 += STEP_S
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # calibration = first CALIB_MIN minutes of the baseline condition
    base = df[df.label == 1].sort_values("t_start_s")
    if len(base):
        cutoff = base.t_start_s.iloc[0] + CALIB_MIN * 60 - WIN_S
        df["calib"] = (df.label == 1) & (df.t_start_s <= cutoff)
    else:
        df["calib"] = False
    return df


# ----------------------------------------------------------------------------- normalisation
def personalise(df: pd.DataFrame) -> pd.DataFrame:
    """z-score each feature per subject against that subject's calibration windows."""
    out = df.copy()
    for subj, g in df.groupby("subject"):
        cal = g[g.calib]
        if len(cal) < 2:
            cal = g[g.label == 1]
        mu = cal[FEATURES].mean()
        sd = cal[FEATURES].std().replace(0, np.nan)
        # floor tiny SDs so a very stable rest period doesn't explode z-scores
        sd = np.maximum(sd.fillna(1.0), 0.1 * g[FEATURES].std().replace(0, 1.0).fillna(1.0))
        out.loc[g.index, FEATURES] = ((g[FEATURES] - mu) / sd).values
    return out


# ----------------------------------------------------------------------------- models
def target(df: pd.DataFrame, task: str) -> np.ndarray:
    if task == "binary":
        return (df.label == 2).astype(int).values
    return df.label.map({1: 0, 2: 1, 3: 2}).values


def class_names(task: str):
    return ["non-stress", "stress"] if task == "binary" else ["baseline", "stress", "amusement"]


def fit_threshold(x: np.ndarray, y: np.ndarray) -> float:
    """Single HR threshold that maximises F1 for 'stress' on the training subjects."""
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    cands = np.unique(np.quantile(x, np.linspace(0.05, 0.95, 91)))
    best_t, best_f = cands[0], -1
    for t in cands:
        f = f1_score(y, (x >= t).astype(int), zero_division=0)
        if f > best_f:
            best_t, best_f = t, f
    return float(best_t)


def make_model(kind: str, n_classes: int):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=3,
                                      class_weight="balanced", random_state=SEED, n_jobs=-1)
    return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.9,
                         colsample_bytree=0.9, random_state=SEED, n_jobs=-1,
                         eval_metric="logloss" if n_classes == 2 else "mlogloss")


def loso(raw: pd.DataFrame, pers: pd.DataFrame, task: str):
    methods = ["rule_universal", "rule_personal", "rf_universal", "rf_personal"]
    if HAVE_XGB:
        methods += ["xgb_universal", "xgb_personal"]
    n_classes = 2 if task == "binary" else 3
    preds = {m: np.full(len(raw), -1) for m in methods}
    thresholds = []
    importances = []

    subjects = sorted(raw.subject.unique(), key=lambda s: int(s[1:]))
    for test_s in subjects:
        tr = (raw.subject != test_s).values
        te = (raw.subject == test_s).values
        y_tr = target(raw[tr], task)

        # --- single-feature HR threshold (binary meaning: stress vs everything else)
        y_tr_bin = (raw[tr].label == 2).astype(int).values
        for m, df in (("rule_universal", raw), ("rule_personal", pers)):
            t = fit_threshold(df[tr].hr_mean.values, y_tr_bin)
            x_te = df[te].hr_mean.values
            p = (np.nan_to_num(x_te, nan=-np.inf) >= t).astype(int)     # missing HR -> no alert
            if task == "three":
                p = np.where(p == 1, 1, 0)        # rule can only say stress / baseline
            preds[m][te] = p
            thresholds.append({"test_subject": test_s, "method": m, "threshold": t})

        # --- ML models
        for kind in (["rf", "xgb"] if HAVE_XGB else ["rf"]):
            for variant, df in (("universal", raw), ("personal", pers)):
                X_tr = df[tr][FEATURES].fillna(df[tr][FEATURES].median())
                X_te = df[te][FEATURES].fillna(df[tr][FEATURES].median())
                mdl = make_model(kind, n_classes)
                mdl.fit(X_tr.values, y_tr)
                preds[f"{kind}_{variant}"][te] = mdl.predict(X_te.values)
                if kind == "rf":
                    importances.append(pd.Series(mdl.feature_importances_, index=FEATURES, name=f"{test_s}_{variant}"))
        print(f"  fold {test_s} done", flush=True)
    return methods, preds, pd.DataFrame(thresholds), pd.concat(importances, axis=1)


# ----------------------------------------------------------------------------- reporting
def score(raw, preds, methods, task):
    y = target(raw, task)
    avg = "binary" if task == "binary" else "macro"
    per_subj = []
    for m in methods:
        for s, g in raw.groupby("subject"):
            idx = g.index.values
            per_subj.append({
                "method": m, "subject": s, "n_windows": len(idx),
                "f1": f1_score(y[idx], preds[m][idx], average=avg, zero_division=0),
                "accuracy": accuracy_score(y[idx], preds[m][idx]),
            })
    per_subj = pd.DataFrame(per_subj)
    summary = (per_subj.groupby("method")[["f1", "accuracy"]]
               .agg(["mean", "std"]).round(3).reindex(methods))
    pooled = {m: {
        "f1_pooled": round(f1_score(y, preds[m], average=avg, zero_division=0), 3),
        "confusion_matrix": confusion_matrix(y, preds[m], labels=list(range(len(class_names(task))))).tolist(),
    } for m in methods}
    return per_subj, summary, pooled


def wilcoxon_tests(per_subj: pd.DataFrame, methods) -> pd.DataFrame:
    """Paired two-sided Wilcoxon signed-rank on per-subject F1: personal vs universal.

    Same subjects, same order (sorted by subject number). Holm correction across
    the comparisons. scipy method='auto': exact distribution when n <= 50 and there
    are no zero differences or tied |differences|; otherwise normal approximation.
    """
    from scipy.stats import wilcoxon
    pivot = per_subj.pivot(index="subject", columns="method", values="f1")
    pivot = pivot.loc[sorted(pivot.index, key=lambda s: int(s[1:]))]
    rows = []
    for kind in ("rule", "rf", "xgb"):
        a, b = f"{kind}_personal", f"{kind}_universal"
        if a not in methods or b not in methods:
            continue
        d = (pivot[a] - pivot[b]).values
        nz = np.abs(d[d != 0])
        exact = (len(d) <= 50) and (np.count_nonzero(d == 0) == 0) and (len(np.unique(nz)) == len(nz))
        res = wilcoxon(pivot[a].values, pivot[b].values, alternative="two-sided", method="auto")
        rows.append({"comparison": f"{a} vs {b}", "n_subjects": len(d),
                     "median_diff_f1": float(np.median(d)), "W": float(res.statistic),
                     "p_raw": float(res.pvalue), "n_zero_diffs": int(np.count_nonzero(d == 0)),
                     "n_tied_abs_diffs": int(len(nz) - len(np.unique(nz))),
                     "scipy_method": "exact" if exact else "normal approximation"})
    out = pd.DataFrame(rows)
    # Holm step-down correction
    order = np.argsort(out.p_raw.values)
    m, running, holm = len(out), 0.0, np.empty(len(out))
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * out.p_raw.values[i]))
        holm[i] = running
    out["p_holm"] = holm
    return out


def plot_all(raw, per_subj, pooled, importances, methods, task, out: Path):
    # 1. resting HR spread across subjects — the argument for personal baselines
    rest = raw[raw.label == 1]
    subs = sorted(rest.subject.unique(), key=lambda s: int(s[1:]))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.boxplot([rest[rest.subject == s].hr_mean.dropna() for s in subs], tick_labels=subs, showfliers=False)
    ax.set_ylabel("Heart rate at seated rest (bpm)")
    ax.set_title("Resting heart rate differs between people (WESAD baseline condition, wrist PPG)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "resting_hr_by_subject.png", dpi=150); plt.close(fig)

    # 2. per-subject F1, universal vs personal
    pivot = per_subj.pivot(index="subject", columns="method", values="f1")
    pivot = pivot.loc[sorted(pivot.index, key=lambda s: int(s[1:])), methods]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        ax.bar(np.arange(len(pivot)) + i * w - 0.4 + w / 2, pivot[m], w, label=m)
    ax.set_xticks(np.arange(len(pivot))); ax.set_xticklabels(pivot.index)
    ax.set_ylabel("F1 (stress)" if task == "binary" else "Macro F1")
    ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=len(methods), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    ax.set_title("Leave-one-subject-out F1 for each held-out subject")
    fig.tight_layout(); fig.savefig(out / "f1_per_subject.png", dpi=150); plt.close(fig)

    # 3. confusion matrices
    names = class_names(task)
    fig, axes = plt.subplots(1, len(methods), figsize=(3.2 * len(methods), 3.2))
    for ax, m in zip(np.atleast_1d(axes), methods):
        cm = np.array(pooled[m]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center", fontsize=8,
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, fontsize=7)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
        ax.set_title(m, fontsize=9); ax.set_xlabel("predicted", fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("true", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "confusion_matrices.png", dpi=150); plt.close(fig)

    # 4. feature importance (RF, mean across folds)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, variant in zip(axes, ["universal", "personal"]):
        cols = [c for c in importances.columns if c.endswith(variant)]
        imp = importances[cols].mean(axis=1).sort_values()
        ax.barh(imp.index, imp.values); ax.set_title(f"RF feature importance ({variant})")
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "feature_importance.png", dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder containing WESAD S2 ... S17")
    ap.add_argument("--out", default=str(Path(__file__).parent / "out"))
    ap.add_argument("--task", choices=["binary", "three"], default="binary")
    ap.add_argument("--reuse-features", action="store_true", help="skip extraction if features.csv exists")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    feat_csv = out / "features.csv"
    t_start = time.time()

    if args.reuse_features and feat_csv.exists():
        raw = pd.read_csv(feat_csv)
        print(f"Loaded {len(raw)} windows from {feat_csv}")
    else:
        subjects = find_subjects(Path(args.data))
        if not subjects:
            sys.exit(f"No S*/S*.pkl files found under {args.data}")
        print(f"Found {len(subjects)} subjects: {', '.join(s for s, _ in subjects)}")
        frames = []
        for name, pkl in subjects:
            print(f"  extracting {name} ...", flush=True)
            frames.append(extract_subject(name, load_subject(pkl)))
        raw = pd.concat(frames, ignore_index=True)
        raw.to_csv(feat_csv, index=False)
        print(f"Saved {len(raw)} windows to {feat_csv}")

    raw = raw.reset_index(drop=True)
    pers = personalise(raw)

    print(f"Running LOSO ({raw.subject.nunique()} folds, task={args.task}, xgboost={'yes' if HAVE_XGB else 'no'})")
    methods, preds, thr, imp = loso(raw, pers, args.task)

    scored = ~raw.calib.values                       # calibration windows never scored
    raw_s = raw[scored].reset_index(drop=True)
    preds_s = {m: p[scored] for m, p in preds.items()}
    per_subj, summary, pooled = score(raw_s, preds_s, methods, args.task)

    per_subj.to_csv(out / f"per_subject_{args.task}.csv", index=False)
    thr.to_csv(out / f"thresholds_{args.task}.csv", index=False)
    summary.to_csv(out / f"summary_{args.task}.csv")
    rest = raw[raw.label == 1].groupby("subject").hr_mean.median()
    meta = {
        "task": args.task, "window_s": WIN_S, "step_s": STEP_S, "purity": PURITY,
        "calibration_min": CALIB_MIN, "n_subjects": int(raw.subject.nunique()),
        "n_windows_scored": int(scored.sum()), "n_calibration_windows": int((~scored).sum()),
        "class_counts_scored": raw_s.label.value_counts().sort_index().to_dict(),
        "resting_hr_median_by_subject_bpm": rest.round(1).to_dict(),
        "resting_hr_range_across_subjects_bpm": [round(rest.min(), 1), round(rest.max(), 1)],
        "pooled": pooled,
        "note": "WESAD stress = lab stressor (TSST). Not panic. Wrist signals only.",
    }
    (out / f"results_{args.task}.json").write_text(json.dumps(meta, indent=2, default=float))
    plot_all(raw_s, per_subj, pooled, imp, methods, args.task, out)

    print("\nLOSO results (mean ± SD across held-out subjects)")
    print(summary.to_string())
    wil = wilcoxon_tests(per_subj, methods)
    wil.to_csv(out / f"wilcoxon_{args.task}.csv", index=False)
    print("\nWilcoxon signed-rank (paired, two-sided) on per-subject F1, personal vs universal")
    print(wil.to_string(index=False))
    print(f"\nResting HR across subjects: {rest.min():.0f}–{rest.max():.0f} bpm (median per subject)")
    print(f"Outputs in {out}  ({time.time() - t_start:.0f} s)")


if __name__ == "__main__":
    main()
