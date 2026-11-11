"""
Exploratory data analysis (Stage 2) for the Auralis stress detection model.

Run after src.features, from software/ml-models/wesad_edge_model:
    python -m src.explore

Reads results/metrics/features_w{W}.csv and writes tables to results/metrics/ and
figures to results/graphs/. Only SCORED windows (calibration period excluded) are used
for class comparisons; calibration windows appear only where stated (resting HR, z-score floors).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from src import preprocessing as pp
from src.features import FEATURES, RELATIVE_FEATURES, normalise

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
GRAPHS = ROOT / "results" / "graphs"

# ----------------------------------------------------------------------------- plot style
CLASS_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}      # baseline, stress, amusement
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#2a78d6", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
})


def subj_order(subjects):
    return sorted(set(subjects), key=lambda s: int(s[1:]))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(GRAPHS / name, dpi=150)
    plt.close(fig)


def scored(df):
    sc = df[df.scored]
    assert not sc.calib.any() and not sc.in_calib_period.any(), "calibration window in scored set"
    return sc


# ----------------------------------------------------------------------------- 1. class balance
def class_balance(df, w):
    sc = scored(df)
    t = sc.groupby("subject").label.value_counts().unstack(fill_value=0).rename(columns=pp.LABEL_NAMES)
    t = t.loc[subj_order(t.index)]
    t["calib_windows"] = df[df.calib].groupby("subject").size()
    t["excluded_overlap_calib"] = df[df.in_calib_period].groupby("subject").size()
    t["stress_fraction"] = (t["stress"] / t[["baseline", "stress", "amusement"]].sum(axis=1)).round(3)
    t.to_csv(METRICS / f"class_balance_w{w}.csv")

    fig, ax = plt.subplots(figsize=(9, 3.6))
    bottom = np.zeros(len(t))
    for lab in (1, 2, 3):
        v = t[pp.LABEL_NAMES[lab]].values
        ax.bar(t.index, v, bottom=bottom, color=CLASS_COLORS[lab], label=pp.LABEL_NAMES[lab],
               edgecolor="#fcfcfb", linewidth=1)
        bottom += v
    ax.set_ylabel("scored windows"); ax.grid(axis="x", visible=False)
    ax.set_title(f"Stress detection: scored windows per subject and class ({w} s windows, 5 s step)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    save(fig, f"class_balance_w{w}.png")
    return t


# ----------------------------------------------------------------------------- 2. resting HR
def resting_hr(df, w):
    rest = df[df.label == 1]
    subs = subj_order(rest.subject)
    t = rest.groupby("subject").hr_mean.describe()[["count", "25%", "50%", "75%"]].loc[subs]
    t.to_csv(METRICS / f"resting_hr_w{w}.csv")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.boxplot([rest[rest.subject == s].hr_mean.dropna() for s in subs], tick_labels=subs, showfliers=False,
               medianprops=dict(color="#2a78d6", linewidth=2), boxprops=dict(color=INK2),
               whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    ax.set_ylabel("heart rate at seated rest (bpm)"); ax.grid(axis="x", visible=False)
    ax.set_title(f"Resting heart rate differs between people (WESAD baseline, wrist PPG, {w} s windows)")
    save(fig, f"resting_hr_by_subject_w{w}.png")
    return t


# ----------------------------------------------------------------------------- 3. distributions
def feature_distributions(df, w, scheme):
    norm, cols = normalise(df, scheme)
    sc = scored(norm)
    n = len(cols); ncol = 5; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.5 * nrow))
    for ax, f in zip(axes.ravel(), cols):
        data = [sc.loc[sc.label == lab, f].dropna() for lab in (1, 2, 3)]
        bp = ax.boxplot(data, tick_labels=["base", "stress", "amus"], showfliers=False, patch_artist=True,
                        medianprops=dict(color=INK, linewidth=1.5), widths=0.6)
        for patch, lab in zip(bp["boxes"], (1, 2, 3)):
            patch.set_facecolor(CLASS_COLORS[lab]); patch.set_edgecolor("#fcfcfb")
        ax.set_title(f, fontsize=9); ax.grid(axis="x", visible=False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    unit = "raw units" if scheme == "universal" else "z vs own first 5 min of baseline"
    fig.suptitle(f"Stress detection features by class ({w} s windows, {scheme}, {unit}; outliers not drawn)",
                 fontweight="bold")
    save(fig, f"feature_distributions_{scheme}_w{w}.png")
    # numeric version
    rows = []
    for f in cols:
        for lab in (1, 2, 3):
            x = sc.loc[sc.label == lab, f]
            rows.append({"feature": f, "class": pp.LABEL_NAMES[lab], "n": int(x.notna().sum()),
                         "median": x.median(), "q25": x.quantile(.25), "q75": x.quantile(.75)})
    pd.DataFrame(rows).to_csv(METRICS / f"feature_distributions_{scheme}_w{w}.csv", index=False)


# ----------------------------------------------------------------------------- 4. correlation
def correlation(df, w):
    c = scored(df)[FEATURES].corr(method="spearman")
    c.to_csv(METRICS / f"feature_correlation_w{w}.csv")
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(c.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES))); ax.set_xticklabels(FEATURES, rotation=60, ha="right")
    ax.set_yticks(range(len(FEATURES))); ax.set_yticklabels(FEATURES); ax.grid(False)
    for (i, j), v in np.ndenumerate(c.values):
        if i != j and abs(v) >= 0.5:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Spearman ρ")
    ax.set_title(f"Feature correlation, scored windows ({w} s); |ρ| ≥ 0.5 labelled")
    save(fig, f"feature_correlation_w{w}.png")
    return c


# ----------------------------------------------------------------------------- 5. subject shifts
def subject_shifts(df, w):
    """Median(stress) - median(scored baseline), per subject, in personal-z units and raw units."""
    norm, _ = normalise(df, "personal_z")
    rows = []
    for name, d in (("z", scored(norm)), ("raw", scored(df))):
        for s, g in d.groupby("subject"):
            shift = g[g.label == 2][FEATURES].median() - g[g.label == 1][FEATURES].median()
            rows.append({"units": name, "subject": s, **shift.to_dict()})
    t = pd.DataFrame(rows)
    t.to_csv(METRICS / f"subject_shifts_stress_minus_baseline_w{w}.csv", index=False)
    z = t[t.units == "z"].set_index("subject")[FEATURES]
    z = z.loc[subj_order(z.index)]
    lim = 5
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(z.clip(-lim, lim).values, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(FEATURES))); ax.set_xticklabels(FEATURES, rotation=60, ha="right")
    ax.set_yticks(range(len(z))); ax.set_yticklabels(z.index); ax.grid(False)
    for (i, j), v in np.ndenumerate(z.values):
        if np.isfinite(v):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.03, label=f"z shift (colour clipped at ±{lim})")
    ax.set_title(f"Per-subject shift, stress minus baseline (median, personal z, {w} s windows)")
    save(fig, f"subject_shifts_w{w}.png")
    # sign consistency: how many subjects move in the majority direction
    cons = pd.DataFrame({"n_up": (z > 0).sum(), "n_down": (z < 0).sum(), "median_shift_z": z.median()})
    cons.to_csv(METRICS / f"subject_shift_consistency_w{w}.csv")
    return cons


# ----------------------------------------------------------------------------- 6. quality flags
FLAGS = ["flag_ppg", "flag_motion", "flag_eda_flat", "flag_eda_low", "flag_any"]


def quality_summary(dfs: dict):
    rows = []
    for w, df in dfs.items():
        for subset, d in (("all", df), ("scored", scored(df)), ("calib", df[df.calib])):
            r = {"window_s": w, "subset": subset, "n_windows": len(d)}
            for f in FLAGS:
                r[f] = int(d[f].sum()); r[f + "_pct"] = round(100 * d[f].mean(), 2)
            rows.append(r)
        for lab in (1, 2, 3):
            d = scored(df)[scored(df).label == lab]
            r = {"window_s": w, "subset": f"scored_{pp.LABEL_NAMES[lab]}", "n_windows": len(d)}
            for f in FLAGS:
                r[f] = int(d[f].sum()); r[f + "_pct"] = round(100 * d[f].mean(), 2)
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(METRICS / "quality_flags_summary.csv", index=False)
    per_subj = pd.concat({w: df.groupby("subject")[FLAGS].sum() for w, df in dfs.items()}, names=["window_s"])
    per_subj.to_csv(METRICS / "quality_flags_per_subject.csv")

    fig, ax = plt.subplots(figsize=(8, 3.4))
    shades = {20: "#86b6ef", 30: "#2a78d6", 60: "#0d366b"}
    a = t[t.subset == "all"].set_index("window_s")
    x = np.arange(len(FLAGS)); bw = 0.26
    for i, w in enumerate(a.index):
        vals = [a.loc[w, f + "_pct"] for f in FLAGS]
        ax.bar(x + (i - 1) * bw, vals, bw, color=shades.get(w, "#2a78d6"), label=f"{w} s", edgecolor="#fcfcfb")
        for xi, v in zip(x, vals):
            ax.text(xi + (i - 1) * bw, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(FLAGS); ax.set_ylabel("% of all windows flagged")
    ax.grid(axis="x", visible=False); ax.legend(title="window", ncol=3)
    ax.set_title("Quality flags (windows are flagged, never removed)")
    save(fig, "quality_flags.png")
    return t


# ----------------------------------------------------------------------------- 7. time shortcut
def time_shortcut(dfs: dict):
    """LOSO, binary stress vs non-stress, depth-3 tree on ONE input at a time.
    Fixed hyperparameters (no tuning), so no inner CV is needed. If 'time since baseline start'
    alone detects stress well, the protocol order is a shortcut a model could exploit."""
    inputs = {"time_since_baseline_start_min": None, "temp_mean": "temp_mean",
              "scl_mean": "scl_mean", "hr_mean": "hr_mean"}
    rows = []
    for w, df in dfs.items():
        sc = scored(df).copy()
        sc["time_since_baseline_start_min"] = (sc.t_start_s - sc.calib_start_s) / 60
        y = (sc.label == 2).astype(int).values
        for name in inputs:
            x = sc[[name]].values
            for s in subj_order(sc.subject):
                tr = (sc.subject != s).values; te = ~tr
                xtr = np.where(np.isnan(x[tr]), np.nanmedian(x[tr]), x[tr])
                xte = np.where(np.isnan(x[te]), np.nanmedian(x[tr]), x[te])
                m = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42).fit(xtr, y[tr])
                p = m.predict_proba(xte)[:, 1]
                rows.append({"window_s": w, "input": name, "subject": s,
                             "roc_auc": roc_auc_score(y[te], p), "pr_auc": average_precision_score(y[te], p),
                             "f1": f1_score(y[te], (p >= 0.5).astype(int), zero_division=0)})
    per = pd.DataFrame(rows)
    per.to_csv(METRICS / "time_shortcut_per_subject.csv", index=False)
    summ = per.groupby(["window_s", "input"])[["roc_auc", "pr_auc", "f1"]].agg(["mean", "std", "min"]).round(3)
    summ.to_csv(METRICS / "time_shortcut_summary.csv")

    fig, axes = plt.subplots(1, len(dfs), figsize=(4 * len(dfs), 3.4), sharey=True)
    for ax, w in zip(np.atleast_1d(axes), dfs):
        d = per[per.window_s == w]
        names = list(inputs)
        ax.boxplot([d[d.input == n].pr_auc for n in names], tick_labels=["time", "temp", "SCL", "HR"],
                   medianprops=dict(color="#eb6834", linewidth=2), boxprops=dict(color=INK2),
                   whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
        ax.axhline(0.35, color=MUTED, linestyle="--", linewidth=1)
        ax.text(4.45, 0.36, "chance ≈ stress share", ha="right", va="bottom", fontsize=7, color=MUTED)
        ax.set_title(f"{w} s windows"); ax.set_ylim(0, 1.05); ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel("per-subject PR-AUC (LOSO)")
    fig.suptitle("Time-in-session shortcut check: single-input stress detection", fontweight="bold")
    save(fig, "time_shortcut.png")
    return summ


# ----------------------------------------------------------------------------- 8. causal vs zero-phase
def causal_vs_zerophase(window_lengths):
    rows = []
    for w in window_lengths:
        c = pd.read_csv(METRICS / f"hrv_causal_vs_zerophase_w{w}.csv")
        for f in ("hr_mean", "rmssd", "sdnn", "pnn50", "n_beats"):
            a, b = c[f + "_causal"], c[f + "_zerophase"]
            ok = a.notna() & b.notna()
            d = (a - b)[ok]
            rows.append({"window_s": w, "feature": f, "n_both": int(ok.sum()),
                         "n_nan_causal": int(a.isna().sum()), "n_nan_zerophase": int(b.isna().sum()),
                         "bias_causal_minus_zp": d.mean(), "sd_diff": d.std(),
                         "mean_abs_diff": d.abs().mean(), "p95_abs_diff": d.abs().quantile(.95),
                         "pearson_r": np.corrcoef(a[ok], b[ok])[0, 1], "mean_zerophase": b[ok].mean()})
    t = pd.DataFrame(rows)
    t.to_csv(METRICS / "hrv_causal_vs_zerophase_summary.csv", index=False)

    w = window_lengths[0]
    c = pd.read_csv(METRICS / f"hrv_causal_vs_zerophase_w{w}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, f, unit in zip(axes, ("hr_mean", "rmssd"), ("bpm", "ms")):
        a, b = c[f + "_causal"], c[f + "_zerophase"]; ok = a.notna() & b.notna()
        mean, diff = (a[ok] + b[ok]) / 2, (a[ok] - b[ok])
        ax.scatter(mean, diff, s=8, alpha=0.25, color="#2a78d6", edgecolors="none")
        bias, sd = diff.mean(), diff.std()
        for v, ls in ((bias, "-"), (bias + 1.96 * sd, "--"), (bias - 1.96 * sd, "--")):
            ax.axhline(v, color=INK2, linestyle=ls, linewidth=1)
        ax.text(0.99, 0.97, f"bias {bias:+.2f} {unit}\nLoA ±{1.96 * sd:.2f} {unit}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=INK2)
        ax.set_xlabel(f"mean of both ({unit})"); ax.set_ylabel(f"causal − zero-phase ({unit})")
        ax.set_title(f"{f}: causal vs zero-phase band-pass ({w} s windows)")
    save(fig, f"hrv_causal_vs_zerophase_w{w}.png")
    return t


# ----------------------------------------------------------------------------- 9. z-score investigation
def zscore_summary(window_lengths):
    rows = []
    for w in window_lengths:
        z = pd.read_csv(METRICS / f"zscore_investigation_w{w}.csv")
        for f, g in z.groupby("feature"):
            rows.append({"window_s": w, "feature": f,
                         "n_subjects_floor_applied": int(g.floor_applied.sum()),
                         "max_abs_z_nofloor": g.z_stress_nofloor.abs().max(),
                         "max_abs_z_baseline_floor": g.z_stress_baseline_floor.abs().max(),
                         "max_abs_z_lso_floor": g.z_stress_lso_floor.abs().max(),
                         "median_abs_z_lso_floor": g.z_stress_lso_floor.abs().median(),
                         "worst_subject_lso": g.loc[g.z_stress_lso_floor.abs().idxmax(), "subject"]})
    t = pd.DataFrame(rows)
    t.to_csv(METRICS / "zscore_investigation_summary.csv", index=False)
    return t


# ----------------------------------------------------------------------------- main
def main():
    GRAPHS.mkdir(parents=True, exist_ok=True); METRICS.mkdir(parents=True, exist_ok=True)
    build = json.loads((METRICS / "feature_build_summary.json").read_text())
    wins = [int(w) for w in build["windows"]]
    dfs = {w: pd.read_csv(METRICS / f"features_w{w}.csv") for w in wins}
    for w, df in dfs.items():
        pp.assert_calibration_never_scored(df)
        class_balance(df, w); resting_hr(df, w); correlation(df, w); subject_shifts(df, w)
        for scheme in ("universal", "personal_z"):          # relative_only = subset of universal
            feature_distributions(df, w, scheme)
        print(f"  {w} s done", flush=True)
    quality_summary(dfs)
    print(time_shortcut(dfs).to_string())
    print(causal_vs_zerophase(wins).round(3).to_string(index=False))
    print(zscore_summary(wins).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
