"""
Step C driver: small deployable models, operating points, per-wearer thresholds, smoothing, time-shortcut
controls, hardware-parity re-evaluation (pre-registered in NOTES.md, Step C).

Run from software/ml-models/wesad_edge_model (in this order):
    python -m src.stepc --phase grid        B (+ D, E, F operating points) for 20/30/60 s, Wilcoxon, selection
    python -m src.stepc --phase followup    C (stress vs amusement, order), three-class, G3/G4 re-evaluation
    python -m src.stepc --phase combine     one combined table + metric checklist (after src.paper_reconcile)
Inputs : results/metrics/features_w{W}.csv, stepc_condition_order.csv (src.protocol),
         features_eda_w{W}.csv and features_hw_*.csv (src.hw_parity; followup only)
Outputs: results/metrics/stepc_*.csv|json, results/graphs/stepc_*.png
"""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import evaluate as ev
from src import feature_spec as fs
from src import protocol
from src.baselines import all_specs, time_only_spec
from src.runner import OPS, latency_summary, run_job_ext, size_summary
from src.small_models import (ALL_NON_HRV, D, d_subsets, hr_spec, model_spec, primary_specs, reference_specs)
from src.train import load_windows, metric_checklist

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
GRAPHS = ROOT / "results" / "graphs"
WINDOWS = (20, 30, 60)
FAMILIES = ("logreg", "tree", "rf", "xgb")


# ----------------------------------------------------------------------------- tables
def spec_cols(spec, win: int, task: str, part: str) -> dict:
    meta = getattr(spec, "meta", {})
    return {"part": part, "task": task, "window_s": win, "key": spec.key, "display": spec.display,
            "model": spec.model, "role": spec.role, "candidate": spec.candidate,
            "scheme": meta.get("scheme", "personal_z" if spec.personal else "universal"),
            "subset": meta.get("subset", ""), "hr_std": meta.get("hr_std"), "motion": meta.get("motion"),
            "temp_slope": meta.get("temp_slope"), "floor": getattr(spec, "floor", "train"),
            "n_features": len(spec.features), "features": " ".join(spec.features),
            "feature_spec_version": fs.VERSION, "order": spec.order}


def tabulate(jobs: list, spec_by: dict, ops=OPS) -> tuple:
    """results (one row per job x operating point), per-subject rows, fold rows."""
    res_rows, per_rows, fold_rows = [], [], []
    for j in jobs:
        cols = spec_cols(spec_by[(j["key"], j["tag"])], j["win"], j["task"], j["tag"])
        f = j["folds"]
        extra = {"params_mode": f.params.mode().iloc[0], "k_mode": f.k.mode().iloc[0] if f.k.notna().any() else np.nan,
                 "mn_mode": f.mn.mode().iloc[0] if (f.mn != "").any() else ""}
        for op in ops:
            r = j["res"][op]
            res_rows.append({**cols, "op": op, **ev.summarise(r["per"], r["pooled"]),
                             **latency_summary(r["per"]), **size_summary(f), **extra})
            per_rows.append(r["per"].assign(**cols, op=op))
        fold_rows.append(f.assign(**cols))
    return pd.DataFrame(res_rows), pd.concat(per_rows, ignore_index=True), pd.concat(fold_rows, ignore_index=True)


def run_parallel(tasks: list, n_jobs: int) -> list:
    """tasks: list of (df, spec, win, task, onset, tag, df_test_source)."""
    return Parallel(n_jobs=n_jobs, verbose=5)(delayed(run_job_ext)(*t) for t in tasks)


def onset_map() -> dict:
    return protocol.load_order().set_index("subject").stress_onset_s.to_dict()


def fmt(r, m="f1", d=3) -> str:
    return f"{r[f'{m}_mean']:.{d}f} ± {r[f'{m}_sd']:.{d}f} [{r[f'{m}_ci_low']:.{d}f}, {r[f'{m}_ci_high']:.{d}f}]"


# ----------------------------------------------------------------------------- selection (pre-registered)
def select(results: pd.DataFrame, families=FAMILIES) -> dict:
    pool = results[(results.role == "primary") & (results.op == "OP1") & (results.task == "binary")
                   & results.model.isin(families)]
    best = pool.sort_values(["f1_mean", "pr_auc_mean"], ascending=False).iloc[0]
    q = pool[(pool.f1_ci_high >= best.f1_ci_low) & (pool.worst_f1 >= best.worst_f1 - 0.05)]
    chosen = q.sort_values(["n_params_median", "window_s", "f1_mean"], ascending=[True, True, False]).iloc[0]
    keep = ["window_s", "key", "display", "model", "scheme", "subset", "features", "f1_mean", "f1_sd", "f1_ci_low",
            "f1_ci_high", "pr_auc_mean", "worst_subject", "worst_f1", "n_params_median", "est_kB_median"]
    return {"families": list(families), "n_pool": int(len(pool)), "n_qualifying": int(len(q)),
            "best": best[keep].to_dict(), "selected": chosen[keep].to_dict(),
            "qualifying": q.sort_values(["n_params_median", "window_s"])[keep].to_dict(orient="records")}


def wilcoxon_table(per: pd.DataFrame) -> pd.DataFrame:
    """Per model and window: universal_all vs personal_z(D) vs relative_only(D), all factors on, OP1."""
    rows = []
    for model in FAMILIES:
        keys = {"universal (all, reference)": model_spec(model, "universal_all", ALL_NON_HRV, "x").key,
                "personal z (D)": model_spec(model, "personal_z", D, "x").key,
                "relative-only (D)": model_spec(model, "relative_only", D, "x").key}
        for w in WINDOWS:
            f1 = {n: per[(per.window_s == w) & (per.key == k) & (per.op == "OP1") & (per.task == "binary")]
                  .set_index("subject").f1 for n, k in keys.items()}
            names = list(keys)
            pairs = [(names[0], names[1]), (names[0], names[2]), (names[1], names[2])]
            fam = [{"model": model, "window_s": w, "a": a, "b": b, **ev.wilcoxon_paired(f1[a], f1[b])}
                   for a, b in pairs]
            for r, p in zip(fam, ev.holm([x["p"] for x in fam])):
                r["p_holm"] = p
            rows += fam
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- phase: grid
def grid_specs() -> list:
    base = all_specs()[:1]                                     # majority class (Step B spec)
    return base + [hr_spec(False), hr_spec(True)] + primary_specs() + reference_specs() + [time_only_spec()]


def phase_grid(n_jobs: int) -> None:
    onset = onset_map()
    dfs = {w: load_windows(w) for w in WINDOWS}
    specs = grid_specs()
    spec_by = {(s.key, "B"): s for s in specs}
    assert len(spec_by) == len(specs), "duplicate spec keys"
    tasks = [(dfs[w], s, w, "binary", onset, "B", None) for w in WINDOWS for s in specs]
    print(f"Step C grid: {len(tasks)} LOSO jobs ({len(specs)} specs x {len(WINDOWS)} windows)", flush=True)
    jobs = run_parallel(tasks, n_jobs)
    results, per, folds = tabulate(jobs, spec_by)
    results = results.sort_values(["window_s", "order", "op"]).reset_index(drop=True)
    results.to_csv(METRICS / "stepc_results_grid.csv", index=False)
    per.to_csv(METRICS / "stepc_per_subject_grid.csv", index=False)
    folds.to_csv(METRICS / "stepc_folds_grid.csv", index=False)

    size_cols = ["window_s", "key", "display", "model", "scheme", "role", "n_features", "features",
                 "params_mode", "n_params_median", "n_params_max", "n_trees_median", "n_nodes_median",
                 "n_nodes_max", "est_kB_median", "est_kB_max"]
    results[results.op == "OP1"][size_cols].to_csv(METRICS / "stepc_model_size.csv", index=False)

    wil = wilcoxon_table(per)
    wil.to_csv(METRICS / "stepc_wilcoxon.csv", index=False)
    sel = {"all_models": select(results), "trees_only": select(results, ("tree", "rf", "xgb"))}
    (METRICS / "stepc_selection.json").write_text(json.dumps(sel, indent=2, default=str))

    # predictions of the selected and best configurations (for plots and the follow-up phase)
    keep = {(sel["all_models"][k]["key"], int(sel["all_models"][k]["window_s"])) for k in ("best", "selected")}
    keep |= {(sel["trees_only"][k]["key"], int(sel["trees_only"][k]["window_s"])) for k in ("best", "selected")}
    pd.concat([j["pred"].assign(key=j["key"], window_s=j["win"]) for j in jobs if (j["key"], j["win"]) in keep]) \
        .to_csv(METRICS / "stepc_predictions_selected.csv", index=False)

    show_grid(results, wil, sel)
    print("\n(checklist below is for stepc_results_grid.csv; the header text comes from train.metric_checklist)")
    metric_checklist(results)


def show_grid(results, wil, sel) -> None:
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 500)
    r = results[results.op == "OP1"].copy()
    r["F1 mean±SD [CI]"] = r.apply(fmt, axis=1)
    cols = ["window_s", "role", "display", "F1 mean±SD [CI]", "pr_auc_mean", "roc_auc_mean", "sensitivity_mean",
            "specificity_mean", "fa_events_per_h_mean", "worst_subject", "worst_f1", "n_params_median"]
    print("\n=== Step C grid, OP1 (best subset per model x scheme x window, plus baselines/references) ===")
    prim = r[r.role == "primary"]
    top = prim.loc[prim.groupby(["window_s", "model", "scheme"]).f1_mean.idxmax()]
    other = r[r.role != "primary"]
    print(pd.concat([top, other]).sort_values(["window_s", "order"])[cols].round(3).to_string(index=False))
    print("\n=== Wilcoxon (per model x window, Holm across 3 pairs) ===")
    print(wil[["model", "window_s", "a", "b", "median_diff", "W", "p", "p_holm", "method"]].round(4).to_string(index=False))
    print("\n=== Selection ===")
    print(json.dumps({k: {"best": v["best"], "selected": v["selected"], "n_qualifying": v["n_qualifying"]}
                      for k, v in sel.items()}, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["grid", "followup", "combine"], required=True)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()
    np.random.seed(0)
    if args.phase == "grid":
        phase_grid(args.jobs)
    elif args.phase == "followup":
        from src.stepc_followup import phase_followup
        phase_followup(args.jobs)
    else:
        from src.stepc_followup import phase_combine
        phase_combine()


if __name__ == "__main__":
    main()
