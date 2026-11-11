"""
Step C follow-up and combine phases (called from src/stepc.py; see its docstring for the run order).

followup: C(i) stress vs amusement, C(ii) condition order x time-only, three-class macro F1 (selected config only),
          G3 EDA gain/offset re-evaluation (original vs redefined features, clean vs perturbed), G4 device-rate
          re-evaluation (held-out subject's features recomputed at device rates), descriptive smoothing grid,
          worst-subject table for the per-wearer threshold.
combine : one combined results table (grid + follow-up + Part A ladder), combined per-subject table, checklist.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src import checks
from src import evaluate as ev
from src import feature_spec as fs
from src import protocol
from src.hw_parity import DEVICE_VARIANTS, EDA_FEATS, REL_FEATS
from src.runner import SMOOTH_GRID, evaluate_ops, latency, latency_summary, prepare_ext, smooth
from src.small_models import ALL_NON_HRV, GRIDS, MOTION, SEED, model_spec
from src.stepc import (FAMILIES, GRAPHS, METRICS, grid_specs, onset_map, run_parallel, tabulate)
from src.train import INNER_FOLDS, load_windows, metric_checklist

WORST = ["S2", "S3", "S13"]


def _sel() -> dict:
    return json.loads((METRICS / "stepc_selection.json").read_text())


def _spec_by_key() -> dict:
    return {s.key: s for s in grid_specs()}


def counterpart(spec, scheme: str = None, features: list = None, role: str = None, suffix: str = ""):
    m = spec.meta
    scheme = scheme or m["scheme"]
    feats = features or spec.features
    extra = {k: m.get(k) for k in ("subset", "hr_std", "motion", "temp_slope")}
    return model_spec(spec.model, scheme, feats, role or spec.role, spec.order, extra, key_suffix=suffix)


# ----------------------------------------------------------------------------- C(i) configurations
def c_configs(results: pd.DataFrame, sel: dict) -> list:
    """(spec, window, label) list for stress vs amusement, as pre-registered (+ labelled exploratory extras)."""
    by_key = _spec_by_key()
    prim = results[(results.role == "primary") & (results.op == "OP1") & (results.task == "binary")]
    picks = []
    for fam in FAMILIES:
        r = prim[prim.model == fam].sort_values(["f1_mean", "pr_auc_mean"], ascending=False).iloc[0]
        picks.append((r.key, int(r.window_s), f"best {fam}"))
    for k in ("all_models", "trees_only"):
        picks.append((sel[k]["selected"]["key"], int(sel[k]["selected"]["window_s"]), f"selected ({k})"))
    out, seen = [], set()
    for key, w, why in picks:
        s = by_key[key]
        for spec, lab in [(s, why)] + [(by_key[model_spec(s.model, sch, ALL_NON_HRV, "x").key], f"drift reference for {why}")
                                        for sch in ("universal_all", "personal_z_all")]:
            if (spec.key, w) not in seen:
                seen.add((spec.key, w)); out.append((spec, w, lab))
        if s.meta.get("motion"):                                   # exploratory, NOT pre-registered
            feats = [f for f in s.features if f not in MOTION]
            e = counterpart(s, features=feats, role="exploratory", suffix=" [exploratory: motion off]")
            if (e.key, w) not in seen:
                seen.add((e.key, w)); out.append((e, w, f"exploratory: {why} without motion"))
    for w in sorted({w for _, w, _ in out}):
        t = by_key["time_only"]
        out.append((t, w, "time-only reference"))
    return out


# ----------------------------------------------------------------------------- three-class
def _est3(model, params, y):
    if model == "logreg":
        return LogisticRegression(C=params["C"], class_weight="balanced", max_iter=5000, random_state=SEED)
    if model == "tree":
        return DecisionTreeClassifier(class_weight="balanced", random_state=SEED, **params)
    if model == "rf":
        return RandomForestClassifier(min_samples_leaf=20, max_features="sqrt", class_weight="balanced",
                                      random_state=SEED, n_jobs=1, **params)
    return XGBClassifier(objective="multi:softprob", num_class=3, random_state=SEED, n_jobs=1, tree_method="hist",
                         eval_metric="mlogloss", **params)


def _fit3(spec, X, y, params, train_subj, test_subj):
    checks.assert_fitted_on_train_only(set(X.subject), train_subj, test_subj, f"{spec.key} 3-class fit")
    steps = [("impute", SimpleImputer(strategy="median"))] + ([("scale", StandardScaler())] if spec.model == "logreg" else [])
    pipe = Pipeline(steps + [("model", _est3(spec.model, params, y))])
    kw = {"model__sample_weight": compute_sample_weight("balanced", y)} if spec.model == "xgb" else {}
    return pipe.fit(X[spec.features].values, y, **kw)


def three_class(df_all: pd.DataFrame, spec) -> tuple:
    sc = df_all[df_all.scored]
    preds, folds = [], []
    for test_s in sorted(sc.subject.unique(), key=lambda s: int(s[1:])):
        train, test = sc[sc.subject != test_s], sc[sc.subject == test_s]
        tr_subj = set(train.subject)
        checks.assert_no_subject_overlap(tr_subj, {test_s})
        checks.assert_no_calibration_rows(train, "training set"); checks.assert_no_calibration_rows(test, "test set")
        y = train.label.map({1: 0, 2: 1, 3: 2}).values
        fold_f1 = np.zeros((len(spec.grid), INNER_FOLDS))
        for j, (itr, iva) in enumerate(GroupKFold(n_splits=INNER_FOLDS).split(train, groups=train.subject)):
            tr_s, va_s = set(train.subject.iloc[itr]), set(train.subject.iloc[iva])
            checks.assert_no_subject_overlap(tr_s, va_s)
            X = prepare_ext(df_all, train, spec, tr_s)
            for i, p in enumerate(spec.grid):
                m = _fit3(spec, X.iloc[itr], y[itr], p, tr_s, va_s)
                fold_f1[i, j] = f1_score(y[iva], m.predict(X.iloc[iva][spec.features].values), average="macro")
        best = int(np.argmax(fold_f1.mean(axis=1)))
        X = prepare_ext(df_all, pd.concat([train, test]), spec, tr_subj)
        m = _fit3(spec, X.loc[train.index], y, spec.grid[best], tr_subj, {test_s})
        p = m.predict(X.loc[test.index][spec.features].values)
        preds.append(pd.DataFrame({"subject": test_s, "true": test.label.map({1: 0, 2: 1, 3: 2}).values, "pred": p}))
        folds.append({"test_subject": test_s, "params": str(spec.grid[best]), "inner_macro_f1": fold_f1[best].mean()})
    pred = pd.concat(preds, ignore_index=True)
    per = pred.groupby("subject").apply(lambda g: pd.Series({
        "macro_f1": f1_score(g.true, g.pred, average="macro", labels=[0, 1, 2], zero_division=0),
        "accuracy": accuracy_score(g.true, g.pred), "balanced_accuracy": balanced_accuracy_score(g.true, g.pred)}),
        include_groups=False).reset_index()
    return pred, per, pd.DataFrame(folds)


# ----------------------------------------------------------------------------- G3 / G4 data
def eda_variant(df: pd.DataFrame, win: int, version: str, redefined: bool) -> pd.DataFrame:
    e = pd.read_csv(METRICS / f"features_eda_w{win}.csv", index_col="row")
    assert (e.index == df.index).all() and (e.t_start_s.values == df.t_start_s.values).all()
    out = df.copy()
    for k in EDA_FEATS + REL_FEATS:
        out[k] = e[f"{k}__{version}"].values
    if version == "clean":
        for k in EDA_FEATS:
            assert np.allclose(out[k], df[k], equal_nan=True), f"clean {k} differs"
    return out


def redefine(features: list) -> list:
    return fs.ordered([fs.REDEFINED.get(f, f) for f in features])


def device_frame(df: pd.DataFrame, win: int, variant: str) -> pd.DataFrame:
    d = pd.read_csv(METRICS / f"features_hw_w{win}__{variant}.csv", index_col="row")
    assert (d.index == df.index).all() and (d.t_start_s.values == df.t_start_s.values).all()
    out = df.copy()
    for c in d.columns:
        if c in out.columns and c not in ("subject", "t_start_s", "t_end_s", "label"):
            out[c] = d[c].values
    return out


# ----------------------------------------------------------------------------- follow-up
def phase_followup(n_jobs: int) -> None:
    sel = _sel()
    results = pd.read_csv(METRICS / "stepc_results_grid.csv")
    onset = onset_map()
    by_key = _spec_by_key()
    sel_spec = by_key[sel["all_models"]["selected"]["key"]]
    sel_w = int(sel["all_models"]["selected"]["window_s"])
    other = counterpart(sel_spec, "personal_z" if sel_spec.meta["scheme"] == "relative_only" else "relative_only")
    dfs = {w: load_windows(w) for w in (20, 30, 60)}

    tasks, spec_by = [], {}

    def add(df, spec, w, task, tag, test_src=None):
        tasks.append((df, spec, w, task, onset, tag, test_src))
        spec_by[(spec.key, tag)] = spec

    # C(i) stress vs amusement
    cfgs = c_configs(results, sel)
    for spec, w, lab in cfgs:
        d = dfs[w]
        add(d[(~d.scored) | d.label.isin([2, 3])].copy(), spec, w, "stress_vs_amusement", "C")
    # G3: original vs redefined EDA features, clean vs perturbed (all subjects perturbed)
    for spec in (sel_spec, other):
        for version in ("clean", "pert"):
            for redef in (False, True):
                feats = redefine(spec.features) if redef else spec.features
                sx = counterpart(spec, features=feats, role=f"G3 {'redefined' if redef else 'original'} {version}",
                                 suffix=f" [G3 {'redef' if redef else 'orig'} {version}]")
                add(eda_variant(dfs[sel_w], sel_w, version, redef), sx, sel_w, "binary", f"G3_{version}_{int(redef)}")
    # G4: held-out subject's features recomputed at device rates (model trained on E4-rate features)
    have_g4 = all((METRICS / f"features_hw_w{sel_w}__{v}.csv").exists() for v in DEVICE_VARIANTS)
    if have_g4:
        for v in DEVICE_VARIANTS:
            sx = counterpart(sel_spec, role=f"G4 {v}", suffix=f" [G4 test={v}]")
            add(dfs[sel_w], sx, sel_w, "binary", f"G4_{v}", device_frame(dfs[sel_w], sel_w, v))
    print(f"follow-up: {len(tasks)} LOSO jobs", flush=True)
    jobs = run_parallel(tasks, n_jobs)
    res, per, folds = tabulate(jobs, spec_by)
    lab = {(s.key, w): l for s, w, l in cfgs}
    res["c_label"] = [lab.get((k, w), "") if t == "C" else "" for k, w, t in zip(res.key, res.window_s, res.part)]
    res.to_csv(METRICS / "stepc_results_followup.csv", index=False)
    per.to_csv(METRICS / "stepc_per_subject_followup.csv", index=False)
    folds.to_csv(METRICS / "stepc_folds_followup.csv", index=False)

    # C(ii) order x time-only (binary from the grid, stress vs amusement from C)
    order = protocol.load_order()[["subject", "order", "sequence_labels_1_to_4"]].rename(columns={"order": "condition_order"})
    pg = pd.read_csv(METRICS / "stepc_per_subject_grid.csv")
    t_bin = pg[(pg.key == "time_only") & (pg.op == "OP1")].assign(task="binary")
    t_sa = per[(per.key == "time_only") & (per.op == "OP1")].assign(task="stress_vs_amusement")
    tt = pd.concat([t_bin, t_sa]).drop(columns="order").merge(order, on="subject")
    tt[["task", "window_s", "subject", "condition_order", "sequence_labels_1_to_4", "f1", "pr_auc", "roc_auc", "sensitivity",
        "specificity", "tp", "fp", "tn", "fn"]].to_csv(METRICS / "stepc_order_time_only_per_subject.csv", index=False)
    og = tt.groupby(["task", "window_s", "condition_order"]).agg(n=("subject", "size"), f1_mean=("f1", "mean"),
                                                       f1_sd=("f1", "std"), roc_auc_mean=("roc_auc", "mean"),
                                                       pr_auc_mean=("pr_auc", "mean"),
                                                       sens_mean=("sensitivity", "mean"),
                                                       spec_mean=("specificity", "mean")).reset_index()
    og.to_csv(METRICS / "stepc_order_time_only.csv", index=False)
    # same split for the physiological stress-vs-amusement configurations
    pc = per[(per.part == "C") & (per.op == "OP1") & (per.key != "time_only")].drop(columns="order").merge(order, on="subject")
    pc.groupby(["key", "display", "window_s", "condition_order"]).agg(n=("subject", "size"), f1_mean=("f1", "mean"),
                                                             roc_auc_mean=("roc_auc", "mean")).reset_index() \
        .to_csv(METRICS / "stepc_order_physiology_sa.csv", index=False)

    # three-class, selected configuration only
    p3, per3, f3 = three_class(dfs[sel_w], sel_spec)
    cm = confusion_matrix(p3.true, p3.pred, labels=[0, 1, 2])
    lo, hi = ev.bootstrap_ci(per3.macro_f1.values)
    three = {"config": sel_spec.display, "window_s": sel_w, "features": sel_spec.features,
             "macro_f1_mean": float(per3.macro_f1.mean()), "macro_f1_sd": float(per3.macro_f1.std()),
             "macro_f1_ci": [lo, hi], "accuracy_mean": float(per3.accuracy.mean()),
             "balanced_accuracy_mean": float(per3.balanced_accuracy.mean()),
             "worst_subject": per3.sort_values("macro_f1").iloc[0].subject, "worst_macro_f1": float(per3.macro_f1.min()),
             "pooled_confusion_rows_true_cols_pred_[baseline,stress,amusement]": cm.tolist(),
             "pooled_macro_f1": float(f1_score(p3.true, p3.pred, average="macro"))}
    (METRICS / "stepc_three_class.json").write_text(json.dumps(three, indent=2))
    per3.merge(f3, left_on="subject", right_on="test_subject").to_csv(METRICS / "stepc_three_class_per_subject.csv",
                                                                       index=False)

    # descriptive smoothing grid on the selected predictions (NOT selected on held-out; for transparency only)
    ps = pd.read_csv(METRICS / "stepc_predictions_selected.csv")
    ps = ps[(ps.key == sel_spec.key) & (ps.window_s == sel_w)]
    rows = []
    for m, n in SMOOTH_GRID:
        p = ps.assign(pred=smooth(ps.pred_op1, ps.subject.values, ps.t_start_s.values, m, n))
        pr = ev.per_subject_metrics(p).merge(latency(p, "pred", onset), on="subject")
        rows.append({"m": m, "n": n, **ev.summarise(pr, ev.pooled_metrics(p, pr)), **latency_summary(pr)})
    pd.DataFrame(rows).to_csv(METRICS / "stepc_smoothing_grid_selected_descriptive.csv", index=False)

    # worst subjects: global vs per-wearer threshold (and the other operating points)
    keys = {(sel_spec.key, sel_w)}
    prim = results[(results.role == "primary") & (results.op == "OP1")]
    for fam in FAMILIES:
        r = prim[prim.model == fam].sort_values(["f1_mean", "pr_auc_mean"], ascending=False).iloc[0]
        keys.add((r.key, int(r.window_s)))
    w_rows = pg[pg.subject.isin(WORST) & pg.apply(lambda r: (r.key, r.window_s) in keys, axis=1)]
    w_rows[["window_s", "display", "op", "subject", "tp", "fn", "fp", "tn", "sensitivity", "specificity", "f1",
            "fa_events_per_h", "roc_auc", "latency_s"]].to_csv(METRICS / "stepc_worst_subjects.csv", index=False)

    print_followup(res, og, three)


def print_followup(res, og, three) -> None:
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)
    cols = ["part", "window_s", "display", "op", "f1_mean", "f1_ci_low", "f1_ci_high", "pr_auc_mean", "roc_auc_mean",
            "sensitivity_mean", "specificity_mean", "fa_events_per_h_mean", "worst_subject", "worst_f1"]
    print("\n=== follow-up results (OP1) ===")
    print(res[res.op == "OP1"][cols + ["c_label"]].round(3).to_string(index=False))
    print("\n=== time-only by condition order ===")
    print(og.round(3).to_string(index=False))
    print("\n=== three-class ===")
    print(json.dumps(three, indent=2))


# ----------------------------------------------------------------------------- combine
def phase_combine() -> None:
    g = pd.read_csv(METRICS / "stepc_results_grid.csv")
    f = pd.read_csv(METRICS / "stepc_results_followup.csv")
    a = pd.read_csv(METRICS / "stepc_paper_ladder.csv")
    a = a[a.step != "P0"].assign(part="A", task="binary", window_s=60, op=a.threshold, key=a.step + "__" + a.detector,
                                 display="paper reference " + a.step + " " + a.detector, role="paper_reference")
    comb = pd.concat([g, f, a], ignore_index=True, sort=False)
    first = ["part", "task", "window_s", "role", "display", "op", "step", "detector"]
    comb = comb[[c for c in first if c in comb] + [c for c in comb.columns if c not in first]]
    comb.to_csv(METRICS / "stepc_combined_results.csv", index=False)
    pa = pd.read_csv(METRICS / "stepc_paper_per_subject.csv").assign(part="A", window_s=60)
    per = pd.concat([pd.read_csv(METRICS / "stepc_per_subject_grid.csv"),
                     pd.read_csv(METRICS / "stepc_per_subject_followup.csv"), pa], ignore_index=True, sort=False)
    per.to_csv(METRICS / "stepc_combined_per_subject.csv", index=False)
    print(f"combined results: {len(comb)} rows; per-subject: {len(per)} rows")
    print("\n(checklist below is for stepc_combined_results.csv; header text comes from train.metric_checklist)")
    metric_checklist(comb)
