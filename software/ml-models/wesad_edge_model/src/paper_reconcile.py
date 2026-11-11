"""
Step C, Part A: reconcile the paper's four detectors (fixed HR rule, personal HR z rule, RF universal, RF personal;
RF 200 trees, depth 8, min_samples_leaf 3, balanced) with the Step B harness. REFERENCE ONLY.

Run from software/ml-models/wesad_edge_model:
    python -m src.paper_reconcile
Reads (never writes) ../wesad_baseline/ (frozen code + out/). The baseline module is imported, not edited; in this
process only its HAVE_XGB flag is switched off (xgb is not one of the four detectors) and, for one seed check,
its SEED constant is set to 0.

Attribution ladder (pre-registered; each step changes one thing, so deltas are order-dependent):
    P0   published numbers (wesad_baseline/out)
    P1   baseline code re-run in-process on its own features.csv              (must reproduce P0)
    P1s  as P1 with RF seed 0 instead of 42                                    (size of RF seed noise)
    P2   P1 predictions re-scored WITHOUT windows overlapping the first 5 min (known difference 2)
    P3   + training rows also exclude calibration/overlap windows             (known difference 2, training side)
    P4   + training-only SD floor                                             (known difference 1)
    P5   Step B harness on the baseline features (inner-fold threshold; RF also at p >= 0.5)
    N15  harness on the causal Step B features, restricted to the baseline's 15 s grid (known difference 3)
    N5   harness on the full 5 s grid = variant (ii) training-only floor
    (i)  N5 with the paper floor (TRANSDUCTIVE)          (iii) N5 with deployable features D only
Outputs: results/metrics/stepc_paper_*.csv
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import evaluate as ev
from src import preprocessing as pp
from src.runner import run_loso_ext
from src.schemes import personal_z
from src.small_models import D, PAPER_FEATURES, hr_spec, model_size, paper_rf_spec
from src.train import load_windows, metric_checklist

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
BASELINE_DIR = ROOT.parent / "wesad_baseline"
BASELINE_OUT = BASELINE_DIR / "out"
DETECTORS = ["rule_universal", "rule_personal", "rf_universal", "rf_personal"]
RENAME = {"scl_std": "eda_std", "scl_slope": "eda_slope"}


def import_baseline():
    sys.path.insert(0, str(BASELINE_DIR))
    import wesad_baseline as wb
    wb.HAVE_XGB = False
    return wb


# ----------------------------------------------------------------------------- metrics on any step grid
def per_subject(pred: pd.DataFrame, step: float) -> pd.DataFrame:
    rows = []
    for s, g in pred.groupby("subject"):
        m = ev.binary_metrics(g.y, g.score, g.pred)
        hours = m["n_nonstress"] * step / 3600.0
        n_ev = ev.count_alarm_events(g.t_start_s.values, ((g.pred == 1) & (g.y == 0)).values, step_s=step)
        m.update(nonstress_hours=hours, fa_windows_per_h=m["fp"] / hours, fa_events=n_ev, fa_events_per_h=n_ev / hours)
        rows.append({"subject": s, **m})
    return pd.DataFrame(rows)


def pooled(pred: pd.DataFrame, per: pd.DataFrame) -> dict:
    m = ev.binary_metrics(pred.y, pred.score, pred.pred)
    h = per.nonstress_hours.sum()
    m.update(fa_windows_per_h=float(per.fp.sum() / h), fa_events=int(per.fa_events.sum()),
             fa_events_per_h=float(per.fa_events.sum() / h))
    return m


# ----------------------------------------------------------------------------- data
def paper_features() -> pd.DataFrame:
    """Baseline features.csv with harness bookkeeping columns added (calibration period from the label vector)."""
    raw = pd.read_csv(BASELINE_OUT / "features.csv").reset_index(drop=True)
    cal = load_windows(60).groupby("subject")[["calib_start_s", "calib_end_s"]].first()
    df = raw.rename(columns=RENAME).join(cal, on="subject")
    df["t_end_s"] = df.t_start_s + 60.0
    df["paper_calib"] = df.calib
    df["in_calib_period"] = (df.t_start_s < df.calib_end_s) & (df.t_end_s > df.calib_start_s)
    df["calib"] = (df.label == 1) & (df.t_start_s >= df.calib_start_s) & (df.t_end_s <= df.calib_end_s)
    df["scored"] = ~df.in_calib_period
    df["y"] = (df.label == 2).astype(int)
    df["flag_ppg"] = False
    df["flag_motion"] = False
    pp.assert_calibration_never_scored(df)
    assert (df.paper_calib <= df.in_calib_period).all()        # every baseline calib window overlaps the period
    return df


# ----------------------------------------------------------------------------- paper-protocol LOSO (P1-P4)
def paper_loso(wb, raw: pd.DataFrame, pers_fn, train_mask: np.ndarray) -> dict:
    """Same steps as wesad_baseline.loso (rules + RF), but keeps scores/probabilities and allows restricting the
    training rows and computing the personal table per fold (pers_fn(train_subjects) -> DataFrame)."""
    out = {m: {"score": np.full(len(raw), np.nan), "pred": np.full(len(raw), -1)} for m in DETECTORS}
    sizes = []
    subjects = sorted(raw.subject.unique(), key=lambda s: int(s[1:]))
    for test_s in subjects:
        tr = ((raw.subject != test_s).values & train_mask)
        te = (raw.subject == test_s).values
        pers = pers_fn(set(raw.subject[tr]))
        y_tr_bin = (raw[tr].label == 2).astype(int).values
        for m, df in (("rule_universal", raw), ("rule_personal", pers)):
            t = wb.fit_threshold(df[tr].hr_mean.values, y_tr_bin)
            x = df[te].hr_mean.values
            out[m]["pred"][te] = (np.nan_to_num(x, nan=-np.inf) >= t).astype(int)
            out[m]["score"][te] = np.where(np.isnan(x), np.nanmin(df[tr].hr_mean.values) - 1, x)
        for variant, df in (("universal", raw), ("personal", pers)):
            X_tr = df[tr][wb.FEATURES].fillna(df[tr][wb.FEATURES].median())
            X_te = df[te][wb.FEATURES].fillna(df[tr][wb.FEATURES].median())
            mdl = wb.make_model("rf", 2).fit(X_tr.values, y_tr_bin)
            out[f"rf_{variant}"]["pred"][te] = mdl.predict(X_te.values)
            out[f"rf_{variant}"]["score"][te] = mdl.predict_proba(X_te.values)[:, 1]
            sizes.append({"test_subject": test_s, "variant": variant,
                          "n_nodes": int(sum(e.tree_.node_count for e in mdl.estimators_)),
                          "n_leaves": int(sum(e.tree_.n_leaves for e in mdl.estimators_))})
        print(f"    fold {test_s}", flush=True)
    out["sizes"] = pd.DataFrame(sizes)
    return out


def step_rows(step: str, desc: str, raw: pd.DataFrame, res: dict, score_mask, threshold: str, cause: str,
              grid_s: float) -> tuple:
    rows, pers_rows = [], []
    for m in DETECTORS:
        p = pd.DataFrame({"subject": raw.subject.values, "t_start_s": raw.t_start_s.values, "y": raw.y.values,
                          "score": res[m]["score"], "pred": res[m]["pred"]})[score_mask]
        per = per_subject(p, grid_s)
        rows.append({"step": step, "description": desc, "detector": m, "threshold": threshold, "cause_vs_previous": cause,
                     "grid_s": grid_s, "n_scored": int(len(p)), **ev.summarise(per, pooled(p, per))})
        pers_rows.append(per.assign(step=step, detector=m))
    return rows, pers_rows


# ----------------------------------------------------------------------------- harness steps (P5, N15, N5, i, iii)
def harness_specs(features: list, floor: str = "train", suffix: str = "") -> dict:
    return {"rule_universal": hr_spec(False), "rule_personal": hr_spec(True, floor),
            "rf_universal": paper_rf_spec("universal", features, suffix=suffix),
            "rf_personal": paper_rf_spec("personal_z", features, floor, suffix)}


def _run(df, spec):
    return run_loso_ext(df, spec, keep_importance=True)


def main():
    wb = import_baseline()
    pub = pd.read_csv(BASELINE_OUT / "per_subject_binary.csv")
    raw = paper_features()
    base_raw = pd.read_csv(BASELINE_OUT / "features.csv").reset_index(drop=True)
    rows, per_rows, size_rows, imp_rows = [], [], [], []

    # P0 published
    for m in DETECTORS:
        g = pub[pub.method == m]
        rows.append({"step": "P0", "description": "published (wesad_baseline/out)", "detector": m,
                     "threshold": "paper", "cause_vs_previous": "", "grid_s": 15, "n_scored": int(g.n_windows.sum()),
                     "f1_mean": g.f1.mean(), "f1_sd": g.f1.std(), "accuracy_mean": g.accuracy.mean()})

    # P1 baseline code, in-process, on its own features.csv -> must reproduce P0
    print("P1: wesad_baseline.loso on features.csv", flush=True)
    pers = wb.personalise(base_raw)
    methods, preds, thr, imp = wb.loso(base_raw, pers, "binary")
    sc = ~base_raw.calib.values
    p1_per, _, _ = wb.score(base_raw[sc].reset_index(drop=True), {k: v[sc] for k, v in preds.items()}, methods,
                            "binary")
    chk = p1_per.merge(pub, on=["method", "subject"], suffixes=("", "_pub"))
    max_diff = float((chk.f1 - chk.f1_pub).abs().max())
    print(f"  P1 vs published per-subject F1: max |diff| = {max_diff:.3g}")
    assert max_diff < 1e-12, "STOP: baseline code does not reproduce its published numbers"
    imp.T.rename_axis("fold").reset_index().to_csv(METRICS / "stepc_paper_rf_importance_P1.csv", index=False)

    # replicate loop (keeps probabilities); with all training rows it must equal wb.loso
    allrows = np.ones(len(raw), bool)
    rep = paper_loso(wb, base_raw.assign(y=raw.y), lambda tr: pers, allrows)
    for m in DETECTORS:
        assert np.array_equal(rep[m]["pred"], preds[m]), f"STOP: replicate loop differs from wesad_baseline.loso ({m})"
    r, pr = step_rows("P1", "baseline code re-run in-process", raw, rep, ~raw.paper_calib.values, "paper",
                      "reproduction (max |dF1| per subject = %.3g)" % max_diff, 15)
    rows += r; per_rows += pr
    size_rows.append(rep["sizes"].assign(step="P1"))

    print("P1s: RF seed 0", flush=True)
    wb.SEED = 0
    rep_s = paper_loso(wb, base_raw.assign(y=raw.y), lambda tr: pers, allrows)
    wb.SEED = 42
    r, pr = step_rows("P1s", "as P1, RF random_state 0", raw, rep_s, ~raw.paper_calib.values, "paper",
                      "RF seed only (noise scale)", 15)
    rows += [x for x in r if x["detector"].startswith("rf")]; per_rows += [x for x in pr if x.detector.iloc[0].startswith("rf")]

    r, pr = step_rows("P2", "P1 re-scored without windows overlapping the first 5 min", raw, rep, raw.scored.values,
                      "paper", "known diff 2: scoring set", 15)
    rows += r; per_rows += pr

    print("P3: no calibration/overlap windows in training", flush=True)
    rep3 = paper_loso(wb, base_raw.assign(y=raw.y), lambda tr: pers, raw.scored.values)
    r, pr = step_rows("P3", "P2 + training excludes calibration/overlap windows", raw, rep3, raw.scored.values, "paper",
                      "known diff 2: training set", 15)
    rows += r; per_rows += pr

    print("P4: training-only SD floor", flush=True)
    base_cal = base_raw.copy()                                  # baseline calib windows for mean/SD, as P1-P3

    def pers_train_floor(train_subj):
        out, _, used = personal_z(base_cal, base_cal, wb.FEATURES, train_subj)
        assert used == set(train_subj)
        return out
    rep4 = paper_loso(wb, base_raw.assign(y=raw.y), pers_train_floor, raw.scored.values)
    r, pr = step_rows("P4", "P3 + training-only SD floor", raw, rep4, raw.scored.values, "paper",
                      "known diff 1: SD floor", 15)
    rows += r; per_rows += pr

    # harness steps
    print("P5 / N15 / N5 / (i) / (iii): harness runs", flush=True)
    n60 = load_windows(60)
    n15 = n60[(n60.t_start_s % 15) == 0].copy()
    steps = {
        "P5": (raw, harness_specs(PAPER_FEATURES), "harness on baseline features", 15,
               "harness bookkeeping (calibration windows from first baseline sample, exact F1 threshold, NaN HR -> median)"),
        "N15": (n15, harness_specs(PAPER_FEATURES), "harness, causal features, 15 s grid", 15,
                "known diff 3: causal filters / parabolic peaks / TEMP 1 Hz"),
        "N5_ii": (n60, harness_specs(PAPER_FEATURES), "(ii) harness, causal features, 5 s grid, training-only floor", 5,
                  "window step 15 -> 5 s"),
        "N5_i": (n60, harness_specs(PAPER_FEATURES, "paper", " [paper floor]"),
                 "(i) as (ii) but paper floor (TRANSDUCTIVE)", 5, "known diff 1 in the new pipeline: paper floor"),
        "N5_iii": (n60, harness_specs(D, suffix=" [D]"), "(iii) as (ii), deployable features D only", 5,
                   "known diff 4 + deployability: drop HRV and absolute EDA/TEMP"),
    }
    jobs = [(st, m, df, spec) for st, (df, specs, _, _, _) in steps.items() for m, spec in specs.items()
            if not (st == "N5_i" and m.endswith("universal"))]           # universal detectors have no floor
    out = Parallel(n_jobs=-1, verbose=5)(delayed(_run)(df, spec) for _, _, df, spec in jobs)
    res = {(st, m): o for (st, m, _, _), o in zip(jobs, out)}
    for st, m in [("N5_i", "rule_universal"), ("N5_i", "rf_universal")]:
        res[(st, m)] = res[("N5_ii", m)]
    for st, (df, specs, desc, grid, cause) in steps.items():
        for m in DETECTORS:
            pred, folds = res[(st, m)]
            ths = [("inner F1 (OP1)", pred.pred_op1)]
            if m.startswith("rf"):
                ths.append(("p >= 0.5 (paper)", (pred.score > 0.5).astype(int)))
            for th, pcol in ths:
                p = pred.assign(pred=pcol.values)
                per = per_subject(p, grid)
                rows.append({"step": st, "description": desc, "detector": m, "threshold": th,
                             "cause_vs_previous": cause, "grid_s": grid, "n_scored": int(len(p)),
                             "features": " ".join(specs[m].features), **ev.summarise(per, pooled(p, per))})
                per_rows.append(per.assign(step=st, detector=m, threshold=th))
            if m.startswith("rf"):
                size_rows.append(folds[["test_subject", "n_nodes", "n_leaves", "n_trees", "n_params", "est_bytes"]]
                                 .assign(step=st, variant=m))
                imp_rows.append(folds.filter(like="imp__").assign(step=st, detector=m, test_subject=folds.test_subject))

    ladder = pd.DataFrame(rows)
    ladder["threshold"] = ladder.threshold.fillna("paper")
    ladder.to_csv(METRICS / "stepc_paper_ladder.csv", index=False)
    pers_all = pd.concat(per_rows, ignore_index=True)
    pers_all["threshold"] = pers_all.threshold.fillna("paper")
    pers_all.to_csv(METRICS / "stepc_paper_per_subject.csv", index=False)
    sizes = pd.concat(size_rows, ignore_index=True)
    sizes["est_kB"] = (sizes.n_nodes * 12) / 1024
    sizes.to_csv(METRICS / "stepc_paper_rf_size.csv", index=False)
    pd.concat(imp_rows, ignore_index=True).to_csv(METRICS / "stepc_paper_rf_importance.csv", index=False)

    # Wilcoxon personal vs universal (rule, rf), Holm across the 2 pairs, per step x threshold
    wil = []
    for (st, th), g in pers_all.groupby(["step", "threshold"]):
        fam = []
        for kind in ("rule", "rf"):
            a = g[g.detector == f"{kind}_personal"].set_index("subject").f1
            b = g[g.detector == f"{kind}_universal"].set_index("subject").f1
            if len(a) and len(b):
                fam.append({"step": st, "threshold": th, "pair": f"{kind}_personal vs {kind}_universal",
                            **ev.wilcoxon_paired(a, b)})
        for x, p in zip(fam, ev.holm([f["p"] for f in fam])):
            x["p_holm_2"] = p
        wil += fam
    pub_w = pd.read_csv(BASELINE_OUT / "wilcoxon_binary.csv")
    for _, r in pub_w.iterrows():
        wil.append({"step": "P0", "threshold": "paper", "pair": r.comparison, "median_diff": r.median_diff_f1,
                    "W": r.W, "p": r.p_raw, "p_holm_3_published": r.p_holm, "method": r.scipy_method})
    wil = pd.DataFrame(wil)
    wil.to_csv(METRICS / "stepc_paper_wilcoxon.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
    show = ladder[["step", "detector", "threshold", "n_scored", "f1_mean", "f1_sd", "f1_ci_low", "f1_ci_high",
                   "pr_auc_mean", "sensitivity_mean", "specificity_mean", "fa_events_per_h_mean", "worst_subject",
                   "worst_f1", "cause_vs_previous"]]
    print("\n=== Part A ladder ===")
    print(show.round(3).to_string(index=False))
    print("\n=== Part A Wilcoxon personal vs universal ===")
    print(wil.round(4).to_string(index=False))
    print("\n(checklist below is for stepc_paper_ladder.csv rows P1-N5; P0 has only F1/accuracy as published)")
    metric_checklist(ladder[ladder.step != "P0"].reset_index(drop=True))


if __name__ == "__main__":
    main()
