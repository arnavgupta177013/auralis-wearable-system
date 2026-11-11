"""
Step C, Part H: paper-ready numbers and figures. Does NOT edit the paper.

Run from software/ml-models/wesad_edge_model (after src.paper_reconcile):
    python -m src.paper_outputs
Writes results/paper_numbers.md, results/metrics/paper_numbers.csv, results/metrics/paper_resting_hr_by_subject.csv
and results/graphs/{resting_hr_by_subject, f1_per_subject, confusion_matrices, feature_importance}.png.

Every value is read from a file (path given next to it). "published" = the frozen baseline pipeline
(wesad_baseline/out, produced by wesad_baseline.py); "corrected" = the same four detectors in the Step B harness
(Part A step N5_ii: causal features, 60 s windows, 5 s step, training-only floor, inner-fold threshold).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.evaluate import MARKERS, SERIES, save_tight
from src.explore import INK, INK2, MUTED          # noqa: F401 (import sets the shared rcParams)
from src.train import load_windows

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
GRAPHS = ROOT / "results" / "graphs"
BOUT = ROOT.parent / "wesad_baseline" / "out"
DET = ["rule_universal", "rule_personal", "rf_universal", "rf_personal"]
DET_NAME = {"rule_universal": "fixed HR rule", "rule_personal": "personal HR z rule",
            "rf_universal": "RF universal", "rf_personal": "RF personal"}
CORR_STEP, CORR_TH = "N5_ii", "inner F1 (OP1)"
REL = lambda p: str(Path(p).resolve().relative_to(ROOT.parent)).replace("\\", "/")
S_BASE, S_PR, S_PO, S_PS = "wesad_baseline/wesad_baseline.py", "wesad_edge_model/src/paper_reconcile.py", \
    "wesad_edge_model/src/paper_outputs.py", "wesad_edge_model/src/features.py"


def err_type(fp, fn) -> str:
    if abs(fp - fn) <= 0.1 * max(fp, fn):
        return "balanced (FP and FN within 10 %)"
    return "false alarms (FP)" if fp > fn else "missed stress (FN)"


def subj_sort(s):
    return sorted(s, key=lambda x: int(x[1:]))


def corrected_per() -> pd.DataFrame:
    p = pd.read_csv(METRICS / "stepc_paper_per_subject.csv")
    return p[(p.step == CORR_STEP) & (p.threshold == CORR_TH)]


def ladder() -> pd.DataFrame:
    return pd.read_csv(METRICS / "stepc_paper_ladder.csv")


# ----------------------------------------------------------------------------- numbers
def numbers() -> list:
    rows = []
    add = lambda item, value, source, script, note="": rows.append(
        {"placeholder": item, "value": value, "source_file": source, "producing_script": script, "note": note})
    rb = json.loads((BOUT / "results_binary.json").read_text())
    lo, hi = rb["resting_hr_range_across_subjects_bpm"]
    add("[X] resting HR, lowest subject median (bpm), published", lo, REL(BOUT / "results_binary.json"), S_BASE,
        "median hr_mean over ALL baseline-condition windows (incl. calibration), 60 s / 15 s step, filtfilt pipeline")
    add("[Y] resting HR, highest subject median (bpm), published", hi, REL(BOUT / "results_binary.json"), S_BASE, "")
    n60 = load_windows(60)
    rest = n60[n60.label == 1].groupby("subject").hr_mean.median()
    rest.rename("resting_hr_median_bpm").reset_index().to_csv(METRICS / "paper_resting_hr_by_subject.csv", index=False)
    add("[X]/[Y] alternative, causal harness features (bpm)", f"{rest.min():.1f} / {rest.max():.1f} "
        f"({rest.idxmin()} / {rest.idxmax()})", REL(METRICS / "paper_resting_hr_by_subject.csv"), S_PO,
        "same definition on features_w60.csv (causal, 5 s step); produced from " + REL(METRICS / "features_w60.csv"))

    sb = pd.read_csv(BOUT / "summary_binary.csv", header=[0, 1], index_col=0)
    L = ladder()
    for tag, d in zip("abcd", DET):
        f = sb.loc[d]
        add(f"[{tag}] F1 {DET_NAME[d]}, published (mean ± SD, 15 subjects)",
            f"{f[('f1', 'mean')]:.3f} ± {f[('f1', 'std')]:.3f}", REL(BOUT / "summary_binary.csv"), S_BASE,
            "mapping [a]-[d] -> the four detectors in this order is ASSUMED (draft not in docs/)")
        r = L[(L.step == CORR_STEP) & (L.detector == d) & (L.threshold == CORR_TH)].iloc[0]
        add(f"[{tag}] F1 {DET_NAME[d]}, corrected harness", f"{r.f1_mean:.3f} ± {r.f1_sd:.3f} "
            f"[95 % CI {r.f1_ci_low:.3f}, {r.f1_ci_high:.3f}]", REL(METRICS / "stepc_paper_ladder.csv"), S_PR,
            f"step {CORR_STEP}, threshold {CORR_TH}")
        r3 = L[(L.step == "N5_iii") & (L.detector == d) & (L.threshold == CORR_TH)].iloc[0]
        add(f"[{tag}] F1 {DET_NAME[d]}, corrected harness, deployable features D only",
            f"{r3.f1_mean:.3f} ± {r3.f1_sd:.3f}", REL(METRICS / "stepc_paper_ladder.csv"), S_PR, "step N5_iii")

    wb = pd.read_csv(BOUT / "wilcoxon_binary.csv")
    for _, r in wb.iterrows():
        add(f"p-value {r.comparison}, published", f"p = {r.p_raw:.3f}, Holm(3) = {r.p_holm:.3f}, W = {r.W:.0f}, "
            f"{r.scipy_method}", REL(BOUT / "wilcoxon_binary.csv"), S_BASE, "")
    wc = pd.read_csv(METRICS / "stepc_paper_wilcoxon.csv")
    for _, r in wc[(wc.step == CORR_STEP) & (wc.threshold == CORR_TH)].iterrows():
        add(f"p-value {r.pair}, corrected harness", f"p = {r.p:.3f}, Holm(2) = {r.p_holm_2:.3f}, W = {r.W:.0f}, "
            f"median diff {r.median_diff:+.3f}, {r.method}", REL(METRICS / "stepc_paper_wilcoxon.csv"), S_PR, "")

    cc = rb["class_counts_scored"]
    add("stress / non-stress windows, published (scored)", f"{cc['2']} / {cc['1'] + cc['3']} "
        f"(baseline {cc['1']}, amusement {cc['3']})", REL(BOUT / "results_binary.json"), S_BASE,
        "60 s windows, 15 s step; 50 of these scored windows overlap the first 5 min (Step 0, difference 2)")
    r = L[(L.step == CORR_STEP) & (L.detector == "rf_universal") & (L.threshold == CORR_TH)].iloc[0]
    add("stress / non-stress windows, corrected harness (scored)",
        f"{int(r.pooled_tp + r.pooled_fn)} / {int(r.pooled_tn + r.pooled_fp)}", REL(METRICS / "stepc_paper_ladder.csv"),
        S_PR, "60 s windows, 5 s step (overlapping, not independent)")

    pub = pd.read_csv(BOUT / "per_subject_binary.csv").pivot(index="subject", columns="method", values="f1")
    cor = corrected_per().pivot(index="subject", columns="detector", values="f1")
    for name, tab, src, scr in (("published", pub, REL(BOUT / "per_subject_binary.csv"), S_BASE),
                                ("corrected harness", cor, REL(METRICS / "stepc_paper_per_subject.csv"), S_PR)):
        for kind in ("rule", "rf"):
            d = tab[f"{kind}_personal"] - tab[f"{kind}_universal"]
            add(f"subjects where personal baseline improved F1 ({kind}, {name})",
                f"{int((d > 0).sum())} of 15 improved, {int((d == 0).sum())} equal, {int((d < 0).sum())} reduced; "
                f"reduced: {', '.join(subj_sort(d[d < 0].index))}", src, scr if name == "published" else S_PO,
                "per-subject F1 difference personal - universal")

    for d in DET:
        (tn, fp), (fn, tp) = rb["pooled"][d]["confusion_matrix"]
        add(f"main confusion-matrix error, {DET_NAME[d]}, published",
            f"{err_type(fp, fn)}: FP {fp}, FN {fn} (TP {tp}, TN {tn})",
            REL(BOUT / "results_binary.json"), S_BASE, "pooled over held-out subjects")
        r = L[(L.step == CORR_STEP) & (L.detector == d) & (L.threshold == CORR_TH)].iloc[0]
        add(f"main confusion-matrix error, {DET_NAME[d]}, corrected harness",
            f"{err_type(r.pooled_fp, r.pooled_fn)}: FP {int(r.pooled_fp)}, "
            f"FN {int(r.pooled_fn)} (TP {int(r.pooled_tp)}, TN {int(r.pooled_tn)})",
            REL(METRICS / "stepc_paper_ladder.csv"), S_PR, "")

    sz = pd.read_csv(METRICS / "stepc_paper_rf_size.csv")
    for st, lab in (("P1", "published pipeline features"), (CORR_STEP, "corrected harness features")):
        g = sz[sz.step == st]
        add(f"RF model size (kB), 200 trees depth 8, {lab}",
            f"median {g.est_kB.median():.1f} kB, max {g.est_kB.max():.1f} kB (median {g.n_nodes.median():.0f} nodes)",
            REL(METRICS / "stepc_paper_rf_size.csv"), S_PR,
            "ESTIMATED: 12 B per node flat float32 array (no compiled code measured); one model per LOSO fold; "
            "CLAUDE.md budget 100 kB int8")
    return rows


def write_md(rows: list) -> None:
    pd.DataFrame(rows).to_csv(METRICS / "paper_numbers.csv", index=False)
    md = ["# Paper numbers (stress detection, WESAD wrist)", "",
          "Generated by `src/paper_outputs.py`. The paper draft is not in `docs/`, so placeholders are matched by the "
          "names given in the Step C brief; the mapping of [a]-[d] to the four detectors is an assumption to check. "
          "\"published\" = frozen baseline pipeline (`wesad_baseline/out`); \"corrected\" = the same four detectors "
          "in the Step B harness (Part A step N5_ii). The corrected values are the ones that respect the paper's own "
          "stated rules (causal features, calibration excluded, no test-subject data in the floor). The paper "
          "itself was not edited.", "",
          "| placeholder | value | source file | producing script | note |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['placeholder']} | {r['value']} | `{r['source_file']}` | `{r['producing_script']}` | {r['note']} |")
    md += ["", "Figures (results/graphs): `resting_hr_by_subject.png`, `f1_per_subject.png`, `confusion_matrices.png`, "
               "`feature_importance.png` (each shows the published pipeline and, where applicable, the corrected harness)."]
    (ROOT / "results" / "paper_numbers.md").write_text("\n".join(md) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- figures
def fig_resting_hr() -> None:
    raw = pd.read_csv(BOUT / "features.csv")
    rest = raw[raw.label == 1]
    subs = subj_sort(rest.subject.unique())
    fig, ax = plt.subplots(figsize=(9.5, 4))
    ax.boxplot([rest[rest.subject == s].hr_mean.dropna() for s in subs], tick_labels=subs, showfliers=False,
               medianprops=dict(color=SERIES[0], linewidth=2), boxprops=dict(color=INK2),
               whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    med = rest.groupby("subject").hr_mean.median()
    ax.set_ylabel("heart rate at seated rest (bpm)")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Stress detection (WESAD): resting heart rate differs between people\n"
                 f"subject medians {med.min():.1f}-{med.max():.1f} bpm; baseline-condition windows incl. calibration, "
                 "wrist PPG, published pipeline", fontsize=9.5)
    save_tight(fig, "resting_hr_by_subject.png")


def fig_f1_per_subject() -> None:
    pub = pd.read_csv(BOUT / "per_subject_binary.csv").rename(columns={"method": "detector"})
    cor = corrected_per()
    subs = subj_sort(pub.subject.unique())
    x = np.arange(len(subs))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)
    off = np.linspace(-0.27, 0.27, 4)
    for ax, tab, title in ((axes[0], pub, "published pipeline (wesad_baseline)"),
                           (axes[1], cor, "corrected harness (causal, calibration excluded, training-only floor)")):
        for i, d in enumerate(DET):
            g = tab[tab.detector == d].set_index("subject").reindex(subs)
            ax.scatter(x + off[i], g.f1, s=34, color=SERIES[i], marker=MARKERS[i], label=DET_NAME[d],
                       edgecolors="#fcfcfb", linewidths=1, zorder=3)
        ax.set_ylim(-0.03, 1.03); ax.set_ylabel("F1 (held-out subject)"); ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=9, loc="left")
    axes[1].set_xticks(x); axes[1].set_xticklabels(subs)
    axes[1].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    fig.suptitle("Stress detection: per-subject F1, leave-one-subject-out, 60 s windows", fontweight="bold",
                 fontsize=10)
    fig.tight_layout()
    save_tight(fig, "f1_per_subject.png")


def fig_confusion() -> None:
    rb = json.loads((BOUT / "results_binary.json").read_text())
    L = ladder()
    cmap = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", SERIES[0]])
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))
    names = ["non-stress", "stress"]
    for row, lab in enumerate(("published pipeline", "corrected harness")):
        for col, d in enumerate(DET):
            if row == 0:
                cm = np.array(rb["pooled"][d]["confusion_matrix"])
            else:
                r = L[(L.step == CORR_STEP) & (L.detector == d) & (L.threshold == CORR_TH)].iloc[0]
                cm = np.array([[r.pooled_tn, r.pooled_fp], [r.pooled_fn, r.pooled_tp]], dtype=int)
            ax = axes[row, col]
            ax.imshow(cm, cmap=cmap, vmin=0)
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=10,
                        color="#ffffff" if v > cm.max() * 0.6 else INK)
            ax.set_xticks([0, 1]); ax.set_xticklabels(names, fontsize=8)
            ax.set_yticks([0, 1]); ax.set_yticklabels(names, fontsize=8)
            ax.grid(False)
            ax.set_title(f"{DET_NAME[d]}\n{lab}", fontsize=8.5)
            if col == 0:
                ax.set_ylabel("true")
            if row == 1:
                ax.set_xlabel("predicted")
    fig.suptitle("Stress detection: pooled confusion matrices over held-out subjects (60 s windows; published = 15 s "
                 "step, corrected = 5 s step)", fontweight="bold", fontsize=10)
    fig.tight_layout()
    save_tight(fig, "confusion_matrices.png")


def fig_importance() -> None:
    p1 = pd.read_csv(METRICS / "stepc_paper_rf_importance_P1.csv").rename(columns={"scl_std": "eda_std",
                                                                                    "scl_slope": "eda_slope"})
    n5 = pd.read_csv(METRICS / "stepc_paper_rf_importance.csv")
    n5 = n5[n5.step == CORR_STEP]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, var in zip(axes, ("universal", "personal")):
        a = p1[p1.fold.str.endswith(var)].drop(columns="fold").mean()
        b = n5[n5.detector == f"rf_{var}"].filter(like="imp__").rename(columns=lambda c: c[5:]).mean()
        order = a.sort_values().index
        y = np.arange(len(order))
        ax.barh(y + 0.2, a[order], height=0.38, color=SERIES[0], label="published pipeline")
        ax.barh(y - 0.2, b.reindex(order), height=0.38, color=SERIES[1], label="corrected harness")
        ax.set_yticks(y); ax.set_yticklabels(order); ax.grid(axis="y", visible=False)
        ax.set_title(f"RF {var}", fontsize=9.5); ax.set_xlabel("mean impurity importance over 15 LOSO folds")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Stress detection: random-forest feature importance (200 trees, depth 8). HRV rows measure PPG noise "
                 "(Step A), not HRV", fontweight="bold", fontsize=9.5)
    fig.tight_layout()
    save_tight(fig, "feature_importance.png")


def main():
    rows = numbers()
    write_md(rows)
    fig_resting_hr(); fig_f1_per_subject(); fig_confusion(); fig_importance()
    print(pd.DataFrame(rows)[["placeholder", "value"]].to_string(index=False))


if __name__ == "__main__":
    main()
