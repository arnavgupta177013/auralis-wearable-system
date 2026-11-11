"""
Step C extension of the Step B LOSO harness (src/train.py). Nothing in train.py is changed; this module calls its
`prepare`, `fit_checked`, `f1_threshold` and the src/checks.py assertions, and adds what Step C needs:

    OP1         threshold = max F1 on pooled inner out-of-fold (OOF) scores        (identical to train.tune)
    OP2         lowest threshold with pooled inner OOF specificity >= 0.90
    per_wearer  threshold = mean + k * max(SD, 0.01) of the model's scores on the wearer's own calibration
                windows; k chosen by pooled inner OOF F1 (inner validation subjects' calibration windows)
    smoothed    OP1 predictions, alert if >= m of the last n windows (5 s grid) are positive; (m, n) chosen by
                pooled inner OOF F1
    paper floor optional TRANSDUCTIVE personal-z floor used by the baseline (Part A reference only)

Parameter choice (mean inner-fold PR-AUC, ties -> first) is the same code path as train.tune; the unit test
tests/test_stepc.py asserts that run_loso_ext picks the same parameters and OP1 threshold as train.run_loso.
Held-out subject: its scored windows are scored once; its calibration windows (unlabelled rest, as on a device)
are scored only to set the per-wearer threshold and are never part of any metric.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import GroupKFold

from src import checks
from src import evaluate as ev
from src import feature_spec as fs
from src import preprocessing as pp
from src.small_models import model_size
from src.train import INNER_FOLDS, f1_threshold, fit_checked, prepare

K_GRID = [0.5, 1, 2, 3, 4, 6, 8, 12, 16]
SMOOTH_GRID = [(1, 1), (2, 2), (2, 3), (3, 3), (3, 4), (3, 5), (4, 5)]
SD_MIN = 0.01
SPEC_MIN = 0.90
OPS = {"OP1": "pred_op1", "OP2_spec90": "pred_op2", "per_wearer": "pred_pw", "smoothed": "pred_sm"}


# ----------------------------------------------------------------------------- normalisation
def paper_floor_z(df_all: pd.DataFrame, rows: pd.DataFrame, features: list) -> pd.DataFrame:
    """Baseline (paper) personal z: SD = max(calib SD, 0.1 x SD over ALL of the subject's windows).
    TRANSDUCTIVE: for the held-out subject that SD uses its own stress windows. Part A reference only."""
    out = rows.copy()
    for s, idx in rows.groupby("subject").groups.items():
        g = df_all[df_all.subject == s]
        cal = g[g.calib]
        assert len(cal) >= 2, f"{s}: fewer than 2 calibration windows"
        mu = cal[features].mean()
        sd = cal[features].std().replace(0, np.nan).fillna(1.0)
        sd = np.maximum(sd, 0.1 * g[features].std().replace(0, 1.0).fillna(1.0))
        out.loc[idx, features] = ((rows.loc[idx, features] - mu) / sd).values
    return out


def prepare_ext(df_all, rows, spec, floor_subjects):
    if spec.personal and getattr(spec, "floor", "train") == "paper":
        return paper_floor_z(df_all, rows, spec.features)
    return prepare(df_all, rows, spec, floor_subjects)          # train.prepare (asserts the training-only floor)


# ----------------------------------------------------------------------------- decision rules
def spec_threshold(y, s, min_spec: float = SPEC_MIN) -> float:
    """Lowest threshold t (stress if s >= t) whose specificity on (y, s) is >= min_spec."""
    y, s = np.asarray(y), np.asarray(s, float)
    neg = np.sort(s[y == 0])
    cands = np.unique(s)
    spec = np.searchsorted(neg, cands, side="left") / len(neg)      # fraction of negatives strictly below t
    ok = cands[spec >= min_spec]
    return float(ok.min()) if len(ok) else float(np.nextafter(s.max(), np.inf))


def smooth(pred, subject, t_start, m: int, n: int, step: float = pp.STEP_S) -> np.ndarray:
    """Causal m-of-n: window t positive if >= m of the windows starting at t, t-step, ..., t-(n-1)step are
    positive. Windows missing from the table (not scored / other labels) count as negative."""
    pred, subject, t_start = np.asarray(pred, int), np.asarray(subject), np.asarray(t_start, float)
    out = np.zeros(len(pred), int)
    for s in np.unique(subject):
        idx = np.flatnonzero(subject == s)
        k = np.round(t_start[idx] / step).astype(int)
        assert len(np.unique(k)) == len(k), "duplicate window start"
        grid = np.zeros(k.max() + 1, int)
        grid[k[pred[idx] == 1]] = 1
        cs = np.r_[0, np.cumsum(grid)]
        out[idx] = (cs[k + 1] - cs[np.maximum(k - n + 1, 0)] >= m).astype(int)
    return out


def per_wearer_pred(s, subjects, cal_stats: dict, k: float) -> np.ndarray:
    thr = np.array([cal_stats[x][0] + k * max(cal_stats[x][1], SD_MIN) for x in subjects])
    return (np.asarray(s) > thr).astype(int)


def _cal_stats(scores: np.ndarray) -> tuple:
    return float(np.mean(scores)), float(np.std(scores, ddof=1))


# ----------------------------------------------------------------------------- inner loop
def tune_ext(df_all: pd.DataFrame, train: pd.DataFrame, y_tr: np.ndarray, spec) -> dict:
    n = len(spec.grid)
    oof = np.full((n, len(train)), np.nan)
    fold_ap = [[] for _ in range(n)]
    cal_scores = [dict() for _ in range(n)]
    for itr, iva in GroupKFold(n_splits=INNER_FOLDS).split(train, groups=train.subject):
        tr, va = train.iloc[itr], train.iloc[iva]
        tr_s, va_s = set(tr.subject), set(va.subject)
        checks.assert_no_subject_overlap(tr_s, va_s)
        X = prepare_ext(df_all, train, spec, tr_s)                   # inner floor: inner-train subjects
        cal = df_all[df_all.calib & df_all.subject.isin(va_s)]
        assert not cal.scored.any()
        Xc = prepare_ext(df_all, cal, spec, tr_s)
        for i, params in enumerate(spec.grid):
            m = fit_checked(spec, X.iloc[itr], y_tr[itr], params, tr_s, va_s)
            s = spec.score(m, X.iloc[iva])
            oof[i, iva] = s
            fold_ap[i].append(average_precision_score(y_tr[iva], s))
            sc = spec.score(m, Xc)
            for subj in va_s:
                cal_scores[i][subj] = sc[cal.subject.values == subj]
    mean_ap = [float(np.mean(a)) for a in fold_ap]
    best = int(np.argmax(mean_ap))                                  # ties -> first (as train.tune)
    s = oof[best]
    assert not np.isnan(s).any()
    subj = train.subject.values
    thr1 = f1_threshold(y_tr, s)
    cal_stats = {x: _cal_stats(v) for x, v in cal_scores[best].items()}
    f1_k = [f1_score(y_tr, per_wearer_pred(s, subj, cal_stats, k), zero_division=0) for k in K_GRID]
    base = (s >= thr1).astype(int)
    f1_mn = [f1_score(y_tr, smooth(base, subj, train.t_start_s.values, m, nn), zero_division=0)
             for m, nn in SMOOTH_GRID]
    return {"params": spec.grid[best], "inner_pr_auc": mean_ap[best], "thr_op1": thr1,
            "thr_op2": spec_threshold(y_tr, s), "k": K_GRID[int(np.argmax(f1_k))], "inner_f1_k": max(f1_k),
            "mn": SMOOTH_GRID[int(np.argmax(f1_mn))], "inner_f1_mn": max(f1_mn),
            "inner_f1_op1": f1_score(y_tr, base, zero_division=0)}


# ----------------------------------------------------------------------------- outer loop
def run_loso_ext(df_all: pd.DataFrame, spec, df_test_source: pd.DataFrame = None, keep_importance: bool = False):
    """LOSO for one spec. df_test_source (optional, same index as df_all for the same windows) supplies the
    held-out subject's rows (e.g. features recomputed at device sampling rates); training rows always come
    from df_all. Returns (predictions DataFrame, per-fold DataFrame)."""
    if spec.role != "shortcut":
        checks.assert_no_time_feature(spec.features)
        if not spec.majority:
            fs.assert_spec_order(spec.features)
    sc = df_all[df_all.scored]
    preds, folds = [], []
    for test_s in sorted(sc.subject.unique(), key=lambda s: int(s[1:])):
        if df_test_source is None:
            fold_all = df_all
        else:
            alt = df_test_source[df_test_source.subject == test_s]
            assert alt.index.isin(df_all.index[df_all.subject == test_s]).all(), "df_test_source index mismatch"
            fold_all = pd.concat([df_all[df_all.subject != test_s], alt]).sort_index()
        train = sc[sc.subject != test_s]
        test = fold_all[fold_all.scored & (fold_all.subject == test_s)]
        test_cal = fold_all[fold_all.calib & (fold_all.subject == test_s)]
        train_subj = set(train.subject)
        checks.assert_no_subject_overlap(train_subj, {test_s})
        checks.assert_no_calibration_rows(train, "training set")
        checks.assert_no_calibration_rows(test, "test set")
        assert test_cal.calib.all() and not test_cal.scored.any() and not test_cal.index.isin(test.index).any()
        y_tr = train.y.values
        row = {"test_subject": test_s, "n_train": len(train), "n_test": len(test), "n_test_calib": len(test_cal)}
        if spec.majority:
            fitted = fit_checked(spec, train, y_tr, None, train_subj, {test_s})
            s = spec.score(fitted, test)
            p = np.full(len(test), fitted["majority"])
            p1 = p2 = ppw = psm = p
            row.update(params="null", thr_op1=np.nan, thr_op2=np.nan, k=np.nan, mn="", inner_pr_auc=np.nan)
        else:
            info = tune_ext(fold_all, train, y_tr, spec)
            X = prepare_ext(fold_all, pd.concat([train, test, test_cal]), spec, train_subj)
            fitted = fit_checked(spec, X.loc[train.index], y_tr, info["params"], train_subj, {test_s})
            s = spec.score(fitted, X.loc[test.index])
            mu, sd = _cal_stats(spec.score(fitted, X.loc[test_cal.index]))
            p1 = (s >= info["thr_op1"]).astype(int)
            p2 = (s >= info["thr_op2"]).astype(int)
            ppw = (s > mu + info["k"] * max(sd, SD_MIN)).astype(int)
            psm = smooth(p1, test.subject.values, test.t_start_s.values, *info["mn"])
            row.update(params=str(info["params"]), thr_op1=info["thr_op1"], thr_op2=info["thr_op2"], k=info["k"],
                       mn=f"{info['mn'][0]}of{info['mn'][1]}", inner_pr_auc=info["inner_pr_auc"],
                       inner_f1_op1=info["inner_f1_op1"], inner_f1_k=info["inner_f1_k"],
                       inner_f1_mn=info["inner_f1_mn"], calib_score_mean=mu, calib_score_sd=sd)
            est = fitted.steps[-1][1] if hasattr(fitted, "steps") else None
            if keep_importance and hasattr(est, "feature_importances_"):
                row.update({f"imp__{f}": v for f, v in zip(spec.features, est.feature_importances_)})
        row.update(model_size(spec, fitted))
        folds.append(row)
        preds.append(pd.DataFrame({
            "subject": test_s, "t_start_s": test.t_start_s.values, "t_end_s": test.t_end_s.values,
            "label": test.label.values, "y": test.y.values, "score": s,
            "pred_op1": p1, "pred_op2": p2, "pred_pw": ppw, "pred_sm": psm,
            "flag_ppg": test.flag_ppg.values, "flag_motion": test.flag_motion.values}))
    return pd.concat(preds, ignore_index=True), pd.DataFrame(folds)


# ----------------------------------------------------------------------------- metrics per operating point
def latency(pred: pd.DataFrame, col: str, onset: dict) -> pd.DataFrame:
    """Per subject: seconds from stress onset to the END of the first alerted scored stress window."""
    rows = []
    for s, g in pred[pred.label == 2].groupby("subject"):
        g = g.sort_values("t_start_s")
        hit = g[g[col] == 1]
        rows.append({"subject": s, "min_possible_s": float(g.t_end_s.iloc[0] - onset[s]),
                     "latency_s": float(hit.t_end_s.iloc[0] - onset[s]) if len(hit) else np.nan,
                     "missed": len(hit) == 0})
    return pd.DataFrame(rows)


def evaluate_ops(pred: pd.DataFrame, onset: dict = None, ops: dict = OPS) -> dict:
    out = {}
    for op, col in ops.items():
        p = pred.assign(pred=pred[col])
        per = ev.per_subject_metrics(p)
        lat = latency(p, col, onset) if onset is not None else None
        if lat is not None:
            per = per.merge(lat, on="subject", how="left")
        out[op] = {"per": per, "pooled": ev.pooled_metrics(p, per)}
    return out


def latency_summary(per: pd.DataFrame) -> dict:
    if "latency_s" not in per:
        return {}
    l = per.latency_s
    return {"latency_median_s": float(l.median()), "latency_mean_s": float(l.mean()),
            "latency_max_s": float(l.max()), "n_missed_episodes": int(per.missed.sum()),
            "min_possible_latency_median_s": float(per.min_possible_s.median())}


def size_summary(folds: pd.DataFrame) -> dict:
    return {"n_params_median": float(folds.n_params.median()), "n_params_max": int(folds.n_params.max()),
            "n_trees_median": float(folds.n_trees.median()), "n_nodes_median": float(folds.n_nodes.median()),
            "n_nodes_max": int(folds.n_nodes.max()), "est_kB_median": float(folds.est_bytes.median() / 1024),
            "est_kB_max": float(folds.est_bytes.max() / 1024)}


# ----------------------------------------------------------------------------- one parallel job
def run_job_ext(df_all, spec, win: int, task: str, onset: dict, tag: str, df_test_source=None,
                keep_importance: bool = False) -> dict:
    """One LOSO run + all operating points. Returns plain data (no Spec) so it travels between processes."""
    pred, folds = run_loso_ext(df_all, spec, df_test_source, keep_importance)
    return {"win": win, "key": spec.key, "task": task, "tag": tag, "pred": pred, "folds": folds,
            "res": evaluate_ops(pred, onset)}
