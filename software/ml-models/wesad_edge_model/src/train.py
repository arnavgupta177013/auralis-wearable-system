"""
Step B: subject-independent validation harness for binary stress detection (stress vs non-stress).

Run from software/ml-models/wesad_edge_model (either form works):
    python -m src.train --windows 20 30 60
    python src\\train.py --windows 20 30 60

Inputs : results/metrics/features_w{W}.csv (from src.features)
Outputs: results/metrics/stepb_*.csv|json and results/graphs/stepb_*.png

Protocol (see NOTES.md, Step B)
-------------------------------
Outer loop : leave-one-subject-out (LOSO) over the 15 subjects.
Inner loop : GroupKFold(7) by subject on the 14 training subjects only. Hyperparameters ranked by mean
             inner-fold PR-AUC; decision threshold = max F1 on the pooled inner out-of-fold predictions.
Held-out   : the test subject is used once, for scoring only.
Population : only `scored` windows (not overlapping the first 5 min of baseline) are trained on and
             scored. Calibration windows are used only for each subject's personal mean/SD.
Leakage assertions (src/checks.py) run on every outer and inner fold.

Order of work in main():
  1. shuffled-label sanity check, 20 seeds (exit code 2 if ROC-AUC is not centred on 0.5)
  2. all models x configurations x window lengths (parallel over jobs)
  3. tables, Wilcoxon tests, PPG-flag sensitivity analysis, plots, output verification
"""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):                       # allow `python src\train.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import GroupKFold

from src import checks
from src import evaluate as ev
from src import preprocessing as pp
from src.baselines import TIME_COL, Spec, all_specs, logreg_spec
from src.schemes import Config, personal_z

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
GRAPHS = ROOT / "results" / "graphs"
INNER_FOLDS = 7
SEED = 0
SHUFFLE_SEEDS = list(range(20))          # seed 0 = run 1
SHUFFLE_CFG = Config("universal", motion=True, drift=True, hrv=False)
MAIN_CFGS = [Config("universal", True, True, False), Config("personal_z", True, True, False),
             Config("relative_only", True, False, False)]


# ----------------------------------------------------------------------------- data
def load_windows(win: int) -> pd.DataFrame:
    df = pd.read_csv(METRICS / f"features_w{win}.csv")
    pp.assert_calibration_never_scored(df)
    df["y"] = (df.label == 2).astype(int)
    df[TIME_COL] = (df.t_start_s - df.calib_start_s) / 60.0     # ONLY used by the time-only reference
    return df


def prepare(df_all: pd.DataFrame, rows: pd.DataFrame, spec: Spec, floor_subjects: set) -> pd.DataFrame:
    """Apply the spec's normalisation to `rows`. For personal z the SD floor uses `floor_subjects`
    (always the training subjects of the current fold) and we assert that it did."""
    if not spec.personal:
        return rows
    out, _, used = personal_z(df_all, rows, spec.features, floor_subjects)
    assert used == set(floor_subjects), "personal-z floor used unexpected subjects"
    return out


def fit_checked(spec: Spec, X: pd.DataFrame, y, params, train_subj: set, test_subj: set):
    """Fit, after asserting that every row comes from a training subject (covers imputer, scaler, model)."""
    checks.assert_fitted_on_train_only(set(X.subject), train_subj, test_subj, f"{spec.key} fit")
    return spec.fit(X, y, params)


def f1_threshold(y, s) -> float:
    """Threshold on score s that maximises F1 (predict stress if s >= threshold)."""
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[int(np.argmax(f1))])


def shuffle_within_subject(y: np.ndarray, subjects: np.ndarray, rng) -> np.ndarray:
    y = y.copy()
    for s in np.unique(subjects):
        idx = np.flatnonzero(subjects == s)
        y[idx] = y[rng.permutation(idx)]
    return y


# ----------------------------------------------------------------------------- inner loop
def tune(df_all: pd.DataFrame, train: pd.DataFrame, y_tr: np.ndarray, spec: Spec) -> dict:
    """Inner GroupKFold on the training subjects: choose params (mean fold PR-AUC) and threshold (pooled OOF F1)."""
    oof = {i: np.full(len(train), np.nan) for i in range(len(spec.grid))}
    fold_ap = {i: [] for i in range(len(spec.grid))}
    for itr, iva in GroupKFold(n_splits=INNER_FOLDS).split(train, groups=train.subject):
        tr, va = train.iloc[itr], train.iloc[iva]
        tr_s, va_s = set(tr.subject), set(va.subject)
        checks.assert_no_subject_overlap(tr_s, va_s)
        X = prepare(df_all, train, spec, floor_subjects=tr_s)          # inner floor: inner-train subjects
        for i, params in enumerate(spec.grid):
            m = fit_checked(spec, X.iloc[itr], y_tr[itr], params, tr_s, va_s)
            s = spec.score(m, X.iloc[iva])
            oof[i][iva] = s
            fold_ap[i].append(average_precision_score(y_tr[iva], s))
    mean_ap = [float(np.mean(fold_ap[i])) for i in range(len(spec.grid))]
    best = int(np.argmax(mean_ap))                    # ties -> first (smallest C / shallowest tree)
    assert not np.isnan(oof[best]).any()
    return {"params": spec.grid[best], "threshold": f1_threshold(y_tr, oof[best]),
            "inner_pr_auc": mean_ap[best]}


# ----------------------------------------------------------------------------- outer loop
def run_loso(df_all: pd.DataFrame, spec: Spec, shuffle_seed=None) -> tuple:
    """Full LOSO for one spec. Returns (predictions DataFrame, per-fold DataFrame)."""
    if spec.role != "shortcut":                    # only the labelled time-only reference may use time
        checks.assert_no_time_feature(spec.features)
    sc = df_all[df_all.scored]
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    preds, folds = [], []
    for test_s in sorted(sc.subject.unique(), key=lambda s: int(s[1:])):
        train, test = sc[sc.subject != test_s], sc[sc.subject == test_s]
        train_subj, test_subj = set(train.subject), {test_s}
        checks.assert_no_subject_overlap(train_subj, test_subj)
        checks.assert_no_calibration_rows(train, "training set")
        checks.assert_no_calibration_rows(test, "test set")
        y_tr = train.y.values
        if rng is not None:
            y_tr = shuffle_within_subject(y_tr, train.subject.values, rng)

        if spec.majority:
            fitted = fit_checked(spec, train, y_tr, None, train_subj, test_subj)
            s = spec.score(fitted, test)
            pred = np.full(len(test), fitted["majority"])
            info = {"params": None, "threshold": np.nan, "inner_pr_auc": np.nan}
        else:
            info = tune(df_all, train, y_tr, spec)
            X = prepare(df_all, pd.concat([train, test]), spec, floor_subjects=train_subj)
            fitted = fit_checked(spec, X.loc[train.index], y_tr, info["params"], train_subj, test_subj)
            s = spec.score(fitted, X.loc[test.index])
            pred = (s >= info["threshold"]).astype(int)
        preds.append(pd.DataFrame({"subject": test_s, "t_start_s": test.t_start_s.values, "label": test.label.values,
                                   "y": test.y.values, "score": s, "pred": pred,
                                   "flag_ppg": test.flag_ppg.values, "flag_motion": test.flag_motion.values}))
        folds.append({"test_subject": test_s, "n_train": len(train), "n_test": len(test),
                      "params": json.dumps(info["params"]), "threshold": info["threshold"],
                      "inner_pr_auc": info["inner_pr_auc"]})
    return pd.concat(preds, ignore_index=True), pd.DataFrame(folds)


def run_job(df_all: pd.DataFrame, spec: Spec, win: int, shuffle_seed=None) -> dict:
    pred, folds = run_loso(df_all, spec, shuffle_seed)
    per = ev.per_subject_metrics(pred)
    # return the key, not the Spec (its fit/score closures need not travel back between processes)
    return {"win": win, "key": spec.key, "pred": pred, "folds": folds, "per": per,
            "pooled": ev.pooled_metrics(pred, per)}


# ----------------------------------------------------------------------------- reporting helpers
def spec_columns(spec: Spec, win: int) -> dict:
    c = spec.cfg
    return {"window_s": win, "key": spec.key, "display": spec.display, "model": spec.model,
            "role": spec.role, "candidate": spec.candidate, "order": spec.order,
            "scheme": c.scheme if c else "", "motion": c.motion if c else np.nan,
            "drift": (c.drift and c.scheme != "relative_only") if c else np.nan,
            "hr_std": c.hr_std if c else np.nan, "hrv": c.hrv if c else np.nan,
            "n_features": len(spec.features), "features": " ".join(spec.features)}


def fmt(r, m, d=3) -> str:
    return f"{r[f'{m}_mean']:.{d}f} ± {r[f'{m}_sd']:.{d}f} [{r[f'{m}_ci_low']:.{d}f}, {r[f'{m}_ci_high']:.{d}f}]"


def shuffled_label_check(df20: pd.DataFrame, n_jobs: int = -1) -> dict:
    """Shuffled-label sanity check, 20 seeds (criterion in NOTES.md, changed after run 1).

    Same configuration, inner folds, C grid and threshold selection as the real harness; only the
    training labels are permuted within each training subject. Test labels are never touched.
    PASS if the mean of the per-seed mean ROC-AUCs is in 0.5 +/- 0.05 AND the 2.5-97.5 percentile
    interval of the per-seed means contains 0.5. Per-seed mean PR-AUC = empirical null for PR-AUC.
    """
    spec = logreg_spec(SHUFFLE_CFG)
    jobs = Parallel(n_jobs=n_jobs)(delayed(run_job)(df20, spec, 20, seed) for seed in SHUFFLE_SEEDS)
    rows = []
    for seed, job in zip(SHUFFLE_SEEDS, jobs):
        per = job["per"].set_index("subject")
        # AP of the REVERSED score, to test the "average precision is asymmetric" explanation directly
        per["pr_auc_reversed_score"] = {s: average_precision_score(g.y, -g.score)
                                        for s, g in job["pred"].groupby("subject")}
        rows.append(per.reset_index().assign(seed=seed))
    per_subj = pd.concat(rows, ignore_index=True)
    per_seed = per_subj.groupby("seed").agg(
        mean_roc_auc=("roc_auc", "mean"), mean_pr_auc=("pr_auc", "mean"), mean_prevalence=("prevalence", "mean"),
        frac_folds_roc_below_05=("roc_auc", lambda x: float((x < 0.5).mean())), mean_f1=("f1", "mean")).reset_index()
    r, pr = per_seed.mean_roc_auc, per_seed.mean_pr_auc
    lo, hi = np.percentile(r, [2.5, 97.5])
    stats_of = lambda v: {"mean": float(v.mean()), "sd": float(v.std()), "min": float(v.min()), "max": float(v.max()),
                          "p2.5": float(np.percentile(v, 2.5)), "p97.5": float(np.percentile(v, 97.5))}
    roc = per_subj.roc_auc
    out = {
        "config": spec.display, "window_s": 20, "n_seeds": len(SHUFFLE_SEEDS),
        "shuffle": "labels permuted within each training subject; test labels untouched",
        "roc_auc_seed_means": stats_of(r),
        "pr_auc_seed_means_empirical_null": stats_of(pr),
        "mean_prevalence": float(per_subj.prevalence.mean()),
        "frac_folds_roc_below_0.5": float((roc < 0.5).mean()),
        "frac_folds_roc_below_0.2_or_above_0.8": float(((roc < 0.2) | (roc > 0.8)).mean()),
        "frac_folds_roc_0.4_to_0.6": float(roc.between(0.4, 0.6).mean()),
        "mean_pr_auc_when_roc_above_0.5": float(per_subj[roc > 0.5].pr_auc.mean()),
        "mean_pr_auc_when_roc_below_0.5": float(per_subj[roc < 0.5].pr_auc.mean()),
        "mean_of_pr_auc_and_reversed_score_pr_auc": float(((per_subj.pr_auc + per_subj.pr_auc_reversed_score) / 2).mean()),
        "per_subject_roc_auc_across_seeds": per_subj.groupby("subject").roc_auc.agg(
            ["mean", "std", "min", "max"]).round(3).to_dict(orient="index"),
        "criterion": "mean of seed means in [0.45, 0.55] AND 2.5-97.5 percentile of seed means contains 0.5",
    }
    out["passed"] = bool(abs(r.mean() - 0.5) <= 0.05 and lo <= 0.5 <= hi)
    per_subj.to_csv(METRICS / "stepb_shuffled_label_per_subject.csv", index=False)
    per_seed.to_csv(METRICS / "stepb_shuffled_label_seeds.csv", index=False)
    (METRICS / "stepb_shuffled_label.json").write_text(json.dumps(out, indent=2))
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", type=int, nargs="+", default=list(pp.WINDOW_LENGTHS_S))
    ap.add_argument("--jobs", type=int, default=-1, help="parallel jobs (joblib); -1 = all cores")
    ap.add_argument("--shuffle-only", action="store_true", help="run only the shuffled-label check")
    args = ap.parse_args()
    np.random.seed(SEED)
    dfs = {w: load_windows(w) for w in args.windows}
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)

    # 1. shuffled-label sanity check first: if it fails, stop before anything else
    if 20 in dfs:
        sh = shuffled_label_check(dfs[20], args.jobs)
        print("\n=== Shuffled-label sanity check (20 s, 20 seeds) ===")
        print(pd.read_csv(METRICS / "stepb_shuffled_label_seeds.csv").round(3).to_string(index=False))
        print(json.dumps(sh, indent=2))
        if not sh["passed"]:
            print("STOP: shuffled-label ROC-AUC is not centred on 0.5 -> leakage warning.")
            sys.exit(2)
        if args.shuffle_only:
            return

    # 2. every spec x window
    specs = all_specs()
    spec_by_key = {s.key: s for s in specs}
    jobs = [(w, s) for w in args.windows for s in specs]
    print(f"\nRunning {len(jobs)} LOSO jobs ({len(specs)} specs x {len(args.windows)} windows) ...", flush=True)
    res = Parallel(n_jobs=args.jobs, verbose=0)(delayed(run_job)(dfs[w], s, w) for w, s in jobs)

    # 3. tables
    per_rows, res_rows, fold_rows = [], [], []
    for r in res:
        cols = spec_columns(spec_by_key[r["key"]], r["win"])
        per_rows.append(r["per"].assign(**cols))
        fold_rows.append(r["folds"].assign(window_s=r["win"], key=r["key"]))
        res_rows.append({**cols, **ev.summarise(r["per"], r["pooled"])})
    per_subj = pd.concat(per_rows, ignore_index=True)
    results = pd.DataFrame(res_rows).sort_values(["window_s", "order"]).reset_index(drop=True)
    per_subj.to_csv(METRICS / "stepb_per_subject.csv", index=False)
    results.to_csv(METRICS / "stepb_results.csv", index=False)
    pd.concat(fold_rows, ignore_index=True).to_csv(METRICS / "stepb_folds.csv", index=False)

    # Wilcoxon on per-subject F1, main LR configurations, Holm within each window length
    main_keys = {logreg_spec(c).key: c.scheme for c in MAIN_CFGS}
    wil = []
    for w in args.windows:
        f1 = {sch: per_subj[(per_subj.window_s == w) & (per_subj.key == k)].set_index("subject").f1
              for k, sch in main_keys.items()}
        pairs = [("universal", "personal_z"), ("universal", "relative_only"), ("personal_z", "relative_only")]
        rows = [{"window_s": w, "a": a, "b": b, **ev.wilcoxon_paired(f1[a], f1[b])} for a, b in pairs]
        for row, padj in zip(rows, ev.holm([x["p"] for x in rows])):
            row["p_holm"] = padj
        wil += rows
    wil = pd.DataFrame(wil)
    wil.to_csv(METRICS / "stepb_wilcoxon.csv", index=False)

    # best LR configuration (by mean per-subject PR-AUC, MAIN no-HRV configs only) + PPG-flag sensitivity
    lr = results[(results.model == "logreg") & (results.role == "main")]
    best = lr.loc[lr.pr_auc_mean.idxmax()]
    best_job = next(r for r in res if r["key"] == best.key and r["win"] == best.window_s)
    best_pred = best_job["pred"]
    best_pred.to_csv(METRICS / "stepb_predictions_best.csv", index=False)
    clean = best_pred[~best_pred.flag_ppg.astype(bool)]
    per_clean = ev.per_subject_metrics(clean)
    sens = pd.DataFrame([
        {"scoring": "all scored windows", "n_windows": len(best_pred),
         **ev.summarise(best_job["per"], best_job["pooled"])},
        {"scoring": "PPG-flagged windows excluded (model unchanged)", "n_windows": len(clean),
         **ev.summarise(per_clean, ev.pooled_metrics(clean, per_clean))},
    ]).assign(window_s=best.window_s, config=best.display)
    sens.to_csv(METRICS / "stepb_sensitivity_ppg.csv", index=False)

    # plots
    for w in args.windows:
        labels = [("hr_universal", "HR threshold (universal)"), ("hr_personal_z", "HR z threshold (personal)")]
        labels += [(logreg_spec(c).key, logreg_spec(c).display) for c in MAIN_CFGS]
        ev.plot_per_subject_f1(per_subj[per_subj.window_s == w], labels, w, f"stepb_per_subject_f1_w{w}.png")
    prev = float(per_subj[per_subj.key == "majority"].prevalence.mean())
    ev.plot_config_comparison(results, "pr_auc", "stepb_config_comparison_pr_auc.png", chance=prev)
    ev.plot_config_comparison(results, "f1", "stepb_config_comparison_f1.png")
    ev.plot_roc_pr(best_pred, f"{best.display}, {best.window_s} s", "stepb_roc_pr_best.png")

    # real PR-AUC vs the shuffled-label empirical null (strictly valid for the shuffled config at 20 s)
    null = pd.read_csv(METRICS / "stepb_shuffled_label_seeds.csv").mean_pr_auc.values
    vs_null = results[results.window_s == 20][["display", "role", "pr_auc_mean"]].copy()
    vs_null["null_mean"], vs_null["null_p97.5"], vs_null["null_max"] = null.mean(), np.percentile(null, 97.5), null.max()
    vs_null["empirical_p"] = [(1 + (null >= v).sum()) / (1 + len(null)) for v in vs_null.pr_auc_mean]
    vs_null["same_config_as_null"] = vs_null.display == logreg_spec(SHUFFLE_CFG).display
    vs_null.to_csv(METRICS / "stepb_pr_auc_vs_null.csv", index=False)

    summary = {"windows": args.windows, "n_jobs_run": len(jobs), "inner_folds": INNER_FOLDS, "seed": SEED,
               "n_bootstrap": ev.N_BOOT,
               "best_logreg": {"window_s": int(best.window_s), "config": best.display,
                               "pr_auc_mean": float(best.pr_auc_mean), "f1_mean": float(best.f1_mean)},
               "main_configs": [logreg_spec(c).display for c in MAIN_CFGS],
               "mean_prevalence": prev,
               "shuffled_null_pr_auc": {"mean": float(null.mean()), "max": float(null.max())}}
    (METRICS / "stepb_summary.json").write_text(json.dumps(summary, indent=2))

    # verify every output file exists and is non-empty
    outs = [METRICS / n for n in ["stepb_per_subject.csv", "stepb_results.csv", "stepb_folds.csv",
                                  "stepb_wilcoxon.csv", "stepb_sensitivity_ppg.csv", "stepb_predictions_best.csv",
                                  "stepb_summary.json", "stepb_shuffled_label.json",
                                  "stepb_shuffled_label_per_subject.csv", "stepb_shuffled_label_seeds.csv",
                                  "stepb_pr_auc_vs_null.csv"]]
    outs += [GRAPHS / f"stepb_per_subject_f1_w{w}.png" for w in args.windows]
    outs += [GRAPHS / n for n in ["stepb_config_comparison_pr_auc.png", "stepb_config_comparison_f1.png",
                                  "stepb_roc_pr_best.png"]]
    for p in outs:
        assert p.exists() and p.stat().st_size > 0, f"missing or empty output: {p}"
    print(f"\nAll {len(outs)} output files exist and are non-empty.")

    # print the real tables
    show = results.copy()
    show["F1 mean±SD [CI]"] = show.apply(lambda r: fmt(r, "f1"), axis=1)
    show["PR-AUC mean±SD [CI]"] = show.apply(lambda r: fmt(r, "pr_auc"), axis=1)
    for m in ["roc_auc", "balanced_accuracy", "sensitivity", "specificity", "precision", "fa_events_per_h"]:
        show[m] = show[f"{m}_mean"].round(3)
    show["accuracy"] = show["accuracy_mean"].round(3)
    show["fa_win_per_h"] = show["fa_windows_per_h_mean"].round(2)
    show["fa_events_per_h"] = show["fa_events_per_h_mean"].round(2)
    show["pooled TP/FP/TN/FN"] = show.apply(lambda r: f"{r.pooled_tp}/{r.pooled_fp}/{r.pooled_tn}/{r.pooled_fn}", axis=1)
    print("\n=== Step B results (mean across 15 held-out subjects; ± SD; [subject bootstrap 95 % CI]) ===")
    print(show[["window_s", "role", "display", "F1 mean±SD [CI]", "PR-AUC mean±SD [CI]", "roc_auc", "accuracy",
                "balanced_accuracy", "sensitivity", "specificity", "precision", "worst_subject", "worst_f1"]]
          .to_string(index=False))
    print("\n=== False alarms per hour of non-stress time (mean across subjects) and pooled confusion matrix ===")
    print(show[["window_s", "display", "fa_win_per_h", "fa_events_per_h", "pooled TP/FP/TN/FN"]].to_string(index=False))
    print("\n=== Wilcoxon signed-rank, per-subject F1, main LR configurations (Holm within window) ===")
    print(wil.round(4).to_string(index=False))
    print(f"\n=== PPG-flag sensitivity, best LR: {best.display}, {best.window_s} s ===")
    print(sens[["scoring", "n_windows", "f1_mean", "pr_auc_mean", "roc_auc_mean", "sensitivity_mean",
                "specificity_mean", "fa_events_per_h_mean"]].round(3).to_string(index=False))
    print("\n=== 20 s PR-AUC vs shuffled-label empirical null (20 seeds) ===")
    print(vs_null.round(3).to_string(index=False))
    metric_checklist(results)


def metric_checklist(results: pd.DataFrame) -> None:
    """Print, and assert, that every required metric exists and is filled for EVERY configuration."""
    required = {
        "accuracy": ["accuracy_mean"], "balanced accuracy": ["balanced_accuracy_mean"],
        "precision": ["precision_mean"], "sensitivity": ["sensitivity_mean"], "specificity": ["specificity_mean"],
        "F1": ["f1_mean"], "ROC-AUC": ["roc_auc_mean"], "PR-AUC": ["pr_auc_mean"],
        "mean ± SD across subjects": [f"{m}_sd" for m in ev.METRICS],
        "confusion matrix (pooled TP/FP/TN/FN)": ["pooled_tp", "pooled_fp", "pooled_tn", "pooled_fn"],
        "false alarms/h per window": ["fa_windows_per_h_mean"],
        "false alarms/h merged events": ["fa_events_per_h_mean"],
        "bootstrap 95 % CI (every metric)": [f"{m}_ci_{b}" for m in ev.METRICS for b in ("low", "high")],
        "pooled metrics": [f"pooled_{m}" for m in ev.METRICS],
        "worst subject": ["worst_subject", "worst_f1"],
    }
    print(f"\n=== Metric checklist: stepb_results.csv ({len(results)} configurations x window rows; "
          f"bootstrap {ev.N_BOOT} subject resamples) ===")
    ok_all = True
    for name, cols in required.items():
        present = all(c in results.columns for c in cols)
        filled = present and results[cols].notna().all().all()
        ok_all &= filled
        print(f"  [{'x' if filled else ' '}] {name:40s} {len(cols)} column(s), present={present}, no missing={filled}")
    assert ok_all, "a required metric is missing for at least one configuration"
    print("  Per-subject confusion matrix (tp/fp/tn/fn) for every configuration: stepb_per_subject.csv")


if __name__ == "__main__":
    main()
