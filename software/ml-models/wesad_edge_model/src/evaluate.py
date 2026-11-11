"""
Metrics, aggregation, statistics and plots for the stress detection LOSO harness (Step B).

Per held-out subject and pooled:
    accuracy, balanced accuracy, precision, sensitivity, specificity, F1, ROC-AUC, PR-AUC,
    confusion matrix (tp, fp, tn, fn), false alarms per hour of non-stress time.
Aggregation: mean +/- SD across subjects, subject-level bootstrap 95 % CI (resample subjects),
worst subject. Statistics: paired Wilcoxon signed-rank on per-subject F1 with Holm correction.

False alarms per hour
---------------------
Windows overlap (5 s step), so non-stress time = (number of scored non-stress windows) x 5 s,
NOT x window length. Two counts:
    fa_windows_per_h : every false-positive window counts once
    fa_events_per_h  : consecutive false-positive windows (adjacent on the 5 s grid) = one alarm event
Precision and F1 are set to 0 when no window is predicted as stress (sklearn zero_division=0).
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_score,
                             roc_auc_score, roc_curve, precision_recall_curve)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import preprocessing as pp
from src.explore import GRAPHS, INK, INK2, MUTED, save   # shared plot style (rcParams set on import)

N_BOOT = 2000
SEED = 0
METRICS = ["accuracy", "balanced_accuracy", "precision", "sensitivity", "specificity", "f1",
           "roc_auc", "pr_auc", "fa_windows_per_h", "fa_events_per_h"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # validated categorical order
MARKERS = ["o", "s", "^", "D", "v"]


# ----------------------------------------------------------------------------- per-subject metrics
def count_alarm_events(t_start: np.ndarray, fp: np.ndarray, step_s: float = pp.STEP_S) -> int:
    """Number of false-alarm events: runs of FP windows that are adjacent on the step grid."""
    t = np.sort(np.asarray(t_start)[np.asarray(fp, dtype=bool)])
    if len(t) == 0:
        return 0
    return int(1 + np.sum(np.diff(t) > step_s + 1e-6))


def binary_metrics(y, score, pred, t_start=None) -> dict:
    """All metrics for one set of windows (one subject, or pooled when t_start is grouped by caller)."""
    y, score, pred = np.asarray(y, int), np.asarray(score, float), np.asarray(pred, int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    both = len(np.unique(y)) == 2
    hours = (tn + fp) * pp.STEP_S / 3600.0
    out = dict(
        n=int(len(y)), n_stress=int(tp + fn), n_nonstress=int(tn + fp), prevalence=float(y.mean()),
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
        accuracy=float((tp + tn) / len(y)), balanced_accuracy=float(np.nanmean([sens, spec])),
        precision=float(precision_score(y, pred, zero_division=0)), sensitivity=float(sens),
        specificity=float(spec), f1=float(f1_score(y, pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y, score)) if both else np.nan,
        pr_auc=float(average_precision_score(y, score)) if both else np.nan,
        nonstress_hours=float(hours),
        fa_windows_per_h=float(fp / hours) if hours > 0 else np.nan,
    )
    if t_start is not None:
        ev = count_alarm_events(t_start, (pred == 1) & (y == 0))
        out.update(fa_events=ev, fa_events_per_h=float(ev / hours) if hours > 0 else np.nan)
    return out


def per_subject_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    """pred_df columns: subject, t_start_s, y, score, pred. One row per held-out subject."""
    rows = []
    for s, g in pred_df.groupby("subject"):
        rows.append({"subject": s, **binary_metrics(g.y, g.score, g.pred, g.t_start_s)})
    return pd.DataFrame(rows)


def pooled_metrics(pred_df: pd.DataFrame, per_subj: pd.DataFrame) -> dict:
    """Metrics on all held-out windows concatenated (each fold has its own model and threshold).
    Alarm events are counted per subject and summed."""
    m = binary_metrics(pred_df.y, pred_df.score, pred_df.pred)
    m["fa_events"] = int(per_subj.fa_events.sum())
    m["fa_events_per_h"] = float(per_subj.fa_events.sum() / per_subj.nonstress_hours.sum())
    return m


# ----------------------------------------------------------------------------- aggregation
def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple:
    """95 % CI of the mean across subjects, resampling SUBJECTS with replacement."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(per_subj: pd.DataFrame, pooled: dict) -> dict:
    """mean, SD, bootstrap CI for every metric + pooled values + worst subject (lowest F1)."""
    out = {}
    for m in METRICS:
        lo, hi = bootstrap_ci(per_subj[m].values)
        out.update({f"{m}_mean": per_subj[m].mean(), f"{m}_sd": per_subj[m].std(ddof=1),
                    f"{m}_ci_low": lo, f"{m}_ci_high": hi})
    for m in METRICS + ["tp", "fp", "tn", "fn"]:
        out[f"pooled_{m}"] = pooled[m]
    worst = per_subj.sort_values(["f1", "pr_auc"]).iloc[0]
    out.update(worst_subject=worst.subject, worst_f1=worst.f1, worst_pr_auc=worst.pr_auc,
               worst_pr_auc_subject=per_subj.sort_values("pr_auc").iloc[0].subject,
               worst_subject_pr_auc_value=per_subj.pr_auc.min())
    return out


# ----------------------------------------------------------------------------- statistics
def holm(p: list) -> list:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def wilcoxon_paired(a: pd.Series, b: pd.Series) -> dict:
    """Two-sided paired Wilcoxon signed-rank on per-subject values (aligned by subject).
    Exact null distribution if there are no zero differences and no tied |differences|,
    otherwise the normal approximation (zeros dropped, 'wilcox' method)."""
    d = (a - b.reindex(a.index)).dropna()
    nz = d[d != 0]
    ties = len(np.unique(np.abs(nz.round(12)))) < len(nz)
    method = "exact" if (len(nz) == len(d) and not ties) else "asymptotic"
    if len(nz) == 0:
        return dict(n=len(d), n_zero=len(d), median_diff=0.0, W=np.nan, p=1.0, method="all zero")
    res = stats.wilcoxon(nz.values, method=method, zero_method="wilcox")
    return dict(n=int(len(d)), n_zero=int(len(d) - len(nz)), median_diff=float(d.median()),
                mean_diff=float(d.mean()), W=float(res.statistic), p=float(res.pvalue),
                method="exact" if method == "exact" else "normal approximation")


# ----------------------------------------------------------------------------- plots
def plot_per_subject_f1(per_subj: pd.DataFrame, labels: list, win: int, fname: str) -> None:
    """Dot plot: per-subject F1 for a few configurations (labels = list of (key, display name))."""
    subs = sorted(per_subj.subject.unique(), key=lambda s: int(s[1:]))
    x = np.arange(len(subs))
    fig, ax = plt.subplots(figsize=(10, 3.9))
    off = np.linspace(-0.28, 0.28, len(labels))
    for i, (key, disp) in enumerate(labels):
        g = per_subj[per_subj.key == key].set_index("subject").reindex(subs)
        ax.scatter(x + off[i], g.f1, s=34, color=SERIES[i], marker=MARKERS[i], label=disp,
                   edgecolors="#fcfcfb", linewidths=1, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(subs); ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("F1 (held-out subject)"); ax.grid(axis="x", visible=False)
    ax.set_title(f"Stress detection: per-subject F1, LOSO ({win} s windows, main configurations)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8)
    save_tight(fig, fname)


def save_tight(fig, name: str) -> None:
    """Save with the bounding box grown to fit legends/labels outside the axes."""
    fig.savefig(GRAPHS / name, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def plot_config_comparison(results: pd.DataFrame, metric: str, fname: str, chance: float = None) -> None:
    """Mean across subjects with subject-level bootstrap 95 % CI, every configuration, one panel per window."""
    wins = sorted(results.window_s.unique())
    keys = results[results.window_s == wins[0]].sort_values(["candidate", "order"], ascending=[True, True]).display
    fig, axes = plt.subplots(1, len(wins), figsize=(4.2 * len(wins) + 2.5, 0.27 * len(keys) + 1.6), sharey=True)
    for ax, w in zip(np.atleast_1d(axes), wins):
        r = results[results.window_s == w].set_index("display").reindex(keys)
        y = np.arange(len(r))
        colors = [MUTED if not c else (SERIES[0] if m == "logreg" else SERIES[1])
                  for c, m in zip(r.candidate, r.model)]
        ax.hlines(y, r[f"{metric}_ci_low"], r[f"{metric}_ci_high"], color=colors, linewidth=2)
        ax.scatter(r[f"{metric}_mean"], y, color=colors, s=26, zorder=3, edgecolors="#fcfcfb", linewidths=1)
        if chance is not None:
            ax.axvline(chance, color=INK2, linestyle=":", linewidth=1)
        ax.set_yticks(y); ax.set_yticklabels(r.index, fontsize=7.5); ax.invert_yaxis()
        ax.set_title(f"{w} s windows"); ax.set_xlabel(f"{metric} (mean across 15 subjects)")
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"Stress detection: {metric} per configuration, LOSO, mean with subject-level bootstrap 95% CI\n"
                 "blue = logistic regression (no HRV); orange = baselines; grey = NOT candidates "
                 "(with-HRV reference, time-only shortcut)" + ("; dotted = stress prevalence" if chance else ""),
                 fontweight="bold", fontsize=9.5)
    fig.tight_layout()
    save_tight(fig, fname)


def plot_roc_pr(pred: pd.DataFrame, title_cfg: str, fname: str) -> None:
    """Pooled ROC and PR curves (all held-out windows; each fold has its own model)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    fpr, tpr, _ = roc_curve(pred.y, pred.score)
    a1.plot(fpr, tpr, color=SERIES[0], linewidth=2, label=f"pooled AUC {roc_auc_score(pred.y, pred.score):.3f}")
    a1.plot([0, 1], [0, 1], color=MUTED, linestyle=":", linewidth=1, label="chance")
    a1.set_xlabel("false positive rate (1 − specificity)"); a1.set_ylabel("sensitivity"); a1.legend(loc="lower right")
    a1.set_title("ROC")
    prec, rec, _ = precision_recall_curve(pred.y, pred.score)
    prev = pred.y.mean()
    a2.plot(rec, prec, color=SERIES[0], linewidth=2,
            label=f"pooled AP {average_precision_score(pred.y, pred.score):.3f}")
    a2.axhline(prev, color=MUTED, linestyle=":", linewidth=1, label=f"chance (prevalence {prev:.2f})")
    a2.set_xlabel("recall (sensitivity)"); a2.set_ylabel("precision"); a2.set_ylim(0, 1.02); a2.legend(loc="lower left")
    a2.set_title("Precision-recall")
    fig.suptitle(f"Stress detection: pooled LOSO ROC and PR curves, best logistic regression ({title_cfg})",
                 fontweight="bold", fontsize=9.5)
    save(fig, fname)
