"""
Step A: chest-ECG reference check for the wrist HR / HRV features (stress detection project).

The chest ECG (RespiBAN, 700 Hz) is used ONLY as a reference to judge the wrist PPG features.
It is never a model input.

Run from software/ml-models/wesad_edge_model:
    python -m src.hrv_check --data "C:\\datasets\\WESAD"

What it does
------------
1. Loads the chest ECG of every subject and detects R-peaks:
     - Butterworth band-pass 5-20 Hz (order 2, zero-phase sosfiltfilt; fine here because this is an
       offline reference, not a deployed feature) -> emphasises the QRS complex.
     - Polarity: per subject, the sign whose extreme values are larger in the 0.5-40 Hz ECG.
     - scipy find_peaks with a 300 ms refractory period (max 200 bpm) and an adaptive height
       threshold (35 % of the 98th percentile of each 10 s block).
     - Each peak is moved to the maximum of the 0.5-40 Hz ECG within +/-50 ms, then refined with
       parabolic interpolation (700 Hz -> 1.4 ms per sample).
     - RR artefact check: 0.3-2.0 s and within 20 % of the running median of 11 RR intervals.
   Peak quality is checked by (a) the fraction of RR intervals that pass the artefact check, per
   subject and condition, and (b) a figure of detected peaks on ECG excerpts for every subject.
2. Computes chest HR, HR SD, RMSSD, SDNN, pNN50 on EXACTLY the window grid of
   results/metrics/features_w{W}.csv (same subject, t_start_s, t_end_s).
3. Compares wrist vs chest: bias, Bland-Altman 95 % limits of agreement (repeated-measures
   corrected), Pearson, Spearman, repeated-measures correlation, MAE, RMSE -- per condition,
   per subject and per window length.
4. Applies the pre-registered keep/drop criteria from NOTES.md and writes the verdict.
5. Sensitivity only (not used for the verdict): repeats the agreement table with the chest window
   shifted by the median measured wrist-behind-chest lag.

Time alignment
--------------
In the WESAD .pkl files the chest (700 Hz) and wrist (E4) streams are already synchronised by the
dataset authors, and both start at sample 0 = time 0. So chest sample i is at i / 700 s and wrist
BVP sample j is at j / 64 s. We assert that every subject's chest and wrist streams have the same
duration, and we check the alignment by cross-correlating 4 Hz instantaneous-HR series from the
chest ECG and from the (causally filtered) wrist BVP. A lag of a fraction of a second is expected
(pulse transit time + causal filter delay) and is small relative to a 20 s window.

Repeated measures
-----------------
Windows within a subject are NOT independent (same person, 5 s step, heavy overlap). Pooled
correlations are therefore descriptive only (no p-values). The limits of agreement use the
Bland & Altman (2007) variance-components correction, and within-subject agreement is summarised
with the repeated-measures correlation (rmcorr).
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import preprocessing as pp
from src.explore import CLASS_COLORS, INK, INK2, MUTED, save   # shared plot style

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
GRAPHS = ROOT / "results" / "graphs"

FS_ECG = 700
QRS_BAND_HZ = (5.0, 20.0)
BROAD_BAND_HZ = (0.5, 40.0)
REFRACTORY_S = 0.30
BLOCK_S = 10.0
HEIGHT_FRAC = 0.35
REFINE_S = 0.05
RR_MIN_S, RR_MAX_S = 0.3, 2.0
RR_MEDIAN_N = 11
RR_TOL = 0.20
CHEST_VALID_MIN_GOOD = 0.90       # chest window valid if >= 90 % of its RR intervals are good

COMPARED = ["hr_mean", "hr_std", "rmssd", "sdnn", "pnn50"]   # hr_std: reported, not a criterion
UNITS = {"hr_mean": "bpm", "hr_std": "bpm", "rmssd": "ms", "sdnn": "ms", "pnn50": "fraction"}
CONDITIONS = {1: "baseline", 2: "stress", 3: "amusement"}

# ----------------------------------------------------------------------------- pre-registered criteria
# Copied from NOTES.md ("Pre-registered keep/drop criteria"). Do not change after seeing results.
HR_CRIT = dict(min_pearson=0.80, max_abs_bias=3.0, loa_limit=10.0, min_rmcorr=0.70)
HRV_CRIT = {
    "rmssd": dict(min_spearman=0.50, max_abs_bias=10.0, loa_limit=40.0, min_rmcorr=0.40),
    "sdnn": dict(min_spearman=0.50, max_abs_bias=10.0, loa_limit=40.0, min_rmcorr=0.40),
    "pnn50": dict(min_spearman=0.50, max_abs_bias=0.10, loa_limit=0.30, min_rmcorr=0.40),
}


# ----------------------------------------------------------------------------- R-peak detection
def load_chest_ecg(pkl_path: Path) -> tuple:
    """Chest ECG (700 Hz), wrist BVP (64 Hz) and labels, with a duration check."""
    with open(pkl_path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    ecg = np.asarray(d["signal"]["chest"]["ECG"], dtype=float).ravel()
    bvp = np.asarray(d["signal"]["wrist"]["BVP"], dtype=float).ravel()
    label = np.asarray(d["label"]).ravel().astype(int)
    assert len(ecg) == len(label), "chest ECG and label lengths differ"
    assert abs(len(bvp) / pp.FS["BVP"] - len(ecg) / FS_ECG) < 1.0 / pp.FS["BVP"], \
        "chest and wrist durations differ -> alignment assumption broken"
    return ecg, bvp, label


def detect_r_peaks(ecg: np.ndarray, fs: float = FS_ECG) -> dict:
    """Returns R-peak times (s), a per-RR 'good' mask and the polarity used."""
    broad = sps.sosfiltfilt(sps.butter(2, BROAD_BAND_HZ, btype="band", fs=fs, output="sos"), ecg)
    # Polarity: is the R wave the large positive or the large negative deflection?
    hi, lo = np.percentile(broad, 99.9), np.percentile(broad, 0.1)
    polarity = 1.0 if abs(hi) >= abs(lo) else -1.0
    broad = polarity * broad
    qrs = polarity * sps.sosfiltfilt(sps.butter(2, QRS_BAND_HZ, btype="band", fs=fs, output="sos"), ecg)

    # Adaptive height threshold, one value per 10 s block
    block = int(BLOCK_S * fs)
    height = np.empty_like(qrs)
    for s in range(0, len(qrs), block):
        seg = qrs[s:s + block]
        height[s:s + block] = HEIGHT_FRAC * np.percentile(seg, 98)
    peaks, _ = sps.find_peaks(qrs, height=height, distance=int(REFRACTORY_S * fs))

    # Move each peak to the local maximum of the broadband ECG and refine to sub-sample time
    r = int(REFINE_S * fs)
    refined = []
    for p in peaks:
        a, b = max(p - r, 1), min(p + r + 1, len(broad) - 1)
        q = a + int(np.argmax(broad[a:b]))
        y0, y1, y2 = broad[q - 1], broad[q], broad[q + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
        refined.append(q + float(np.clip(off, -0.5, 0.5)))
    t = np.unique(np.round(np.asarray(refined) / fs, 6))   # two peaks refined to the same R -> one

    # RR artefact check. good[i] refers to the interval between peak i and peak i+1.
    rr = np.diff(t)
    med = pd.Series(rr).rolling(RR_MEDIAN_N, center=True, min_periods=1).median().values
    good = (rr >= RR_MIN_S) & (rr <= RR_MAX_S) & (np.abs(rr - med) <= RR_TOL * med)
    return {"t": t, "rr": rr, "good": good, "polarity": polarity, "broad": broad}


# ----------------------------------------------------------------------------- per-window chest values
def chest_window_values(t: np.ndarray, rr: np.ndarray, good: np.ndarray, t0: float, t1: float) -> dict:
    """HR / HRV from the R-peaks inside [t0, t1). Same definitions as the wrist features:
    hr_mean = mean of 60/RR, hr_std = SD of 60/RR, sdnn = SD(RR) (population), rmssd and pnn50 from
    successive differences. Unlike the wrist code, successive differences are taken only between
    two ADJACENT good RR intervals (proper NN intervals)."""
    i0, i1 = np.searchsorted(t, t0), np.searchsorted(t, t1)      # peaks with t0 <= t < t1
    idx = np.arange(i0, i1 - 1)                                   # RR intervals fully inside
    out = dict(chest_n_rr=int(len(idx)), chest_good_frac=np.nan, chest_valid=False,
               **{f"chest_{k}": np.nan for k in COMPARED})
    if len(idx) < 4:
        return out
    g = good[idx]
    out["chest_good_frac"] = float(g.mean())
    nn = rr[idx][g]
    both = g[:-1] & g[1:]
    d = np.diff(rr[idx])[both] * 1000.0
    if len(nn) < 4 or len(d) < 3:
        return out
    hr = 60.0 / nn
    out.update(chest_hr_mean=float(hr.mean()), chest_hr_std=float(hr.std()),
               chest_sdnn=float(np.std(nn * 1000.0)), chest_rmssd=float(np.sqrt(np.mean(d ** 2))),
               chest_pnn50=float(np.mean(np.abs(d) > 50)),
               chest_valid=bool(out["chest_good_frac"] >= CHEST_VALID_MIN_GOOD))
    return out


# ----------------------------------------------------------------------------- alignment check
def instantaneous_hr(beat_t: np.ndarray, good: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Instantaneous HR (bpm) at each good beat, linearly interpolated to `grid` (s)."""
    tt = beat_t[1:][good]
    hr = 60.0 / np.diff(beat_t)[good]
    return np.interp(grid, tt, hr, left=np.nan, right=np.nan)


def alignment_lag(chest_t, chest_good, bvp, label, max_lag_s=10.0, fs_grid=4.0) -> dict:
    """Lag (s) that maximises the correlation between chest and wrist instantaneous HR.
    Positive lag = wrist HR series lags behind the chest series."""
    x = pp.CausalSOS(pp.bvp_bandpass_sos()).process(bvp)          # the same causal filter as the model
    fs = pp.FS["BVP"]
    pk, _ = sps.find_peaks(x, distance=int(0.33 * fs), prominence=0.3 * np.std(x))
    wt = pk / fs
    wrr = np.diff(wt)
    wmed = pd.Series(wrr).rolling(RR_MEDIAN_N, center=True, min_periods=1).median().values
    wgood = (wrr > 0.33) & (wrr < 1.5) & (np.abs(wrr - wmed) < 0.3 * wmed)
    # only the labelled conditions 1-3 (seated parts of the protocol)
    keep = np.isin(label[::int(FS_ECG / fs_grid)], list(CONDITIONS))   # label sampled at 4 Hz
    grid = np.arange(len(keep)) / fs_grid
    a = instantaneous_hr(chest_t, chest_good, grid)
    b = instantaneous_hr(wt, wgood, grid)
    a[~keep] = np.nan; b[~keep] = np.nan
    lags = np.arange(-int(max_lag_s * fs_grid), int(max_lag_s * fs_grid) + 1)
    r = []
    for L in lags:                       # b shifted by L samples: compare a[n] with b[n + L]
        aa = a[max(0, -L): len(a) - max(0, L)]
        bb = b[max(0, L): len(b) - max(0, -L)]
        m = ~np.isnan(aa) & ~np.isnan(bb)
        r.append(np.corrcoef(aa[m], bb[m])[0, 1] if m.sum() > 100 else np.nan)
    r = np.asarray(r)
    best = int(np.nanargmax(r))
    return {"lag_s": float(lags[best] / fs_grid), "r_at_lag": float(r[best]),
            "r_at_zero": float(r[lags == 0][0])}


# ----------------------------------------------------------------------------- agreement statistics
def rm_limits_of_agreement(diff: np.ndarray, subject: np.ndarray) -> tuple:
    """Bias and 95 % limits of agreement corrected for repeated measures
    (Bland & Altman 2007, 'true value varies'): var = between-subject + within-subject variance."""
    df = pd.DataFrame({"d": diff, "s": subject})
    g = df.groupby("s")["d"]
    n_i, m_i = g.size(), g.mean()
    N, k = len(df), len(n_i)
    bias = float(df.d.mean())
    msw = float(((df.d - df.s.map(m_i)) ** 2).sum() / (N - k)) if N > k else np.nan
    if k < 2:
        return bias, np.nan, np.nan, np.nan
    msb = float((n_i * (m_i - bias) ** 2).sum() / (k - 1))
    n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    var_b = max(0.0, (msb - msw) / n0)
    sd = float(np.sqrt(var_b + msw))
    return bias, bias - 1.96 * sd, bias + 1.96 * sd, sd


def rmcorr(x: np.ndarray, y: np.ndarray, group: np.ndarray) -> float:
    """Repeated-measures correlation: Pearson r of subject-mean-centred x and y."""
    df = pd.DataFrame({"x": x, "y": y, "g": group})
    xc = df.x - df.groupby("g").x.transform("mean")
    yc = df.y - df.groupby("g").y.transform("mean")
    if xc.std() == 0 or yc.std() == 0:
        return np.nan
    return float(np.corrcoef(xc, yc)[0, 1])


def agreement(w: np.ndarray, c: np.ndarray, subject: np.ndarray) -> dict:
    """All agreement statistics for one set of paired (wrist, chest) windows."""
    d = w - c
    out = dict(n_windows=int(len(d)), n_subjects=int(len(np.unique(subject))))
    if len(d) < 5:
        return out
    bias, lo, hi, sd_rm = rm_limits_of_agreement(d, subject)
    out.update(
        wrist_mean=float(w.mean()), chest_mean=float(c.mean()), bias=bias,
        sd_diff_naive=float(d.std(ddof=1)), loa_low=lo, loa_high=hi, sd_diff_rm=sd_rm,
        loa_low_naive=bias - 1.96 * float(d.std(ddof=1)), loa_high_naive=bias + 1.96 * float(d.std(ddof=1)),
        pearson=float(stats.pearsonr(w, c)[0]) if np.std(w) > 0 and np.std(c) > 0 else np.nan,
        spearman=float(stats.spearmanr(w, c)[0]) if np.std(w) > 0 and np.std(c) > 0 else np.nan,
        rmcorr=rmcorr(w, c, subject), mae=float(np.abs(d).mean()), rmse=float(np.sqrt(np.mean(d ** 2))),
    )
    return out


# ----------------------------------------------------------------------------- verdict
def evaluate_criteria(agr: pd.DataFrame, win: int) -> dict:
    """Apply the pre-registered criteria to the pooled agreement table of one window length."""
    a = agr[agr.window_s == win].set_index(["feature", "condition"])
    res = {"checks": {}}
    hr_ok = True
    for cond in CONDITIONS.values():
        r = a.loc[("hr_mean", cond)]
        chk = {"H1_pearson": bool(r.pearson >= HR_CRIT["min_pearson"]),
               "H2_bias": bool(abs(r.bias) <= HR_CRIT["max_abs_bias"]),
               "H3_loa": bool(r.loa_low >= -HR_CRIT["loa_limit"] and r.loa_high <= HR_CRIT["loa_limit"])}
        res["checks"][f"hr_mean/{cond}"] = chk
        hr_ok &= all(chk.values())
    h4 = bool(a.loc[("hr_mean", "all")].rmcorr >= HR_CRIT["min_rmcorr"])
    res["checks"]["hr_mean/all"] = {"H4_rmcorr": h4}
    hr_ok &= h4

    tracks, agrees = {}, {}
    for f, crit in HRV_CRIT.items():
        t_ok, a_ok = True, True
        for cond in CONDITIONS.values():
            r = a.loc[(f, cond)]
            chk = {"V1_spearman": bool(r.spearman >= crit["min_spearman"]),
                   "V3_bias": bool(abs(r.bias) <= crit["max_abs_bias"]),
                   "V4_loa": bool(r.loa_low >= -crit["loa_limit"] and r.loa_high <= crit["loa_limit"])}
            res["checks"][f"{f}/{cond}"] = chk
            t_ok &= chk["V1_spearman"]
            a_ok &= chk["V3_bias"] and chk["V4_loa"]
        v2 = bool(a.loc[(f, "all")].rmcorr >= crit["min_rmcorr"])
        res["checks"][f"{f}/all"] = {"V2_rmcorr": v2}
        tracks[f], agrees[f] = bool(t_ok and v2), bool(a_ok)

    if not all(tracks.values()):
        verdict = "keep HR only"
    elif not all(agrees.values()):
        verdict = "keep HRV with caveats"
    else:
        verdict = "keep HRV"
    if not hr_ok:
        verdict += " (HR not verified: keep with caveats)"
    res.update(hr_passes=bool(hr_ok), hrv_tracks_reference=tracks, hrv_agrees_in_value=agrees, verdict=verdict)
    return res


def lag_shifted_agreement(feats: dict, subj_peaks: dict, shift_s: float) -> pd.DataFrame:
    """SENSITIVITY ONLY (not used for the verdict): pair each wrist window [t0, t1) with the chest
    window [t0 - shift, t1 - shift), i.e. undo the median wrist-behind-chest lag, and recompute agreement."""
    rows = []
    for w, df in feats.items():
        vals = []
        for name, g in df.groupby("subject"):
            t, rr, good = subj_peaks[name]
            for _, row in g.iterrows():
                vals.append({"subject": name, "label": row.label, **{f: row[f] for f in COMPARED},
                             **chest_window_values(t, rr, good, row.t_start_s - shift_s, row.t_end_s - shift_s)})
        m = pd.DataFrame(vals)
        for f in COMPARED:
            ok = m[f].notna() & m[f"chest_{f}"].notna() & m.chest_valid
            q0 = m[ok]
            for cond_lab, cond in list(CONDITIONS.items()) + [(None, "all")]:
                q = q0 if cond_lab is None else q0[q0.label == cond_lab]
                rows.append({"window_s": w, "feature": f, "condition": cond, "chest_shift_s": shift_s,
                             **agreement(q[f].values, q[f"chest_{f}"].values, q.subject.values)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- plots
def plot_bland_altman(paired: pd.DataFrame, agr: pd.DataFrame, win: int) -> None:
    feats = ["hr_mean", "rmssd", "sdnn", "pnn50"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    for ax, f in zip(axes, feats):
        p = paired[(paired.feature == f)]
        for lab, cond in CONDITIONS.items():
            q = p[p.label == lab]
            ax.scatter((q.wrist + q.chest) / 2, q.wrist - q.chest, s=5, alpha=0.35, color=CLASS_COLORS[lab],
                       label=cond, linewidths=0, rasterized=True)
        r = agr[(agr.window_s == win) & (agr.feature == f) & (agr.condition == "all")].iloc[0]
        ax.axhline(r.bias, color=INK, linewidth=1.5)
        for v in (r.loa_low, r.loa_high):
            ax.axhline(v, color=INK2, linewidth=1, linestyle="--")
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.text(0.02, 0.97, f"bias {r.bias:.2f}\nRM 95% LoA [{r.loa_low:.1f}, {r.loa_high:.1f}]",
                transform=ax.transAxes, va="top", fontsize=8, color=INK)
        ax.set_title(f"{f} ({UNITS[f]})")
        ax.set_xlabel("mean of wrist and chest"); ax.set_ylabel("wrist − chest")
    axes[0].legend(loc="lower left", markerscale=3, fontsize=8)
    fig.suptitle(f"Stress detection features: wrist PPG vs chest ECG reference, Bland-Altman ({win} s windows, "
                 "all subjects; solid = bias, dashed = repeated-measures 95% LoA)", fontweight="bold")
    save(fig, f"hrv_check_bland_altman_w{win}.png")


def plot_peak_examples(examples: list) -> None:
    """One ECG excerpt (8 s, stress condition) per subject with the detected R-peaks."""
    n = len(examples); ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 1.7 * nrow), sharex=False)
    for ax, ex in zip(axes.ravel(), examples):
        ax.plot(ex["t"], ex["ecg"], color=INK2, linewidth=0.7)
        ax.scatter(ex["peaks_t"], ex["peaks_y"], color=CLASS_COLORS[2], s=14, zorder=3)
        ax.set_title(f"{ex['subject']} (polarity {'+' if ex['polarity'] > 0 else '−'}, "
                     f"good RR {ex['good_frac']:.1%})", fontsize=8)
        ax.set_yticks([]); ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle("Stress detection reference: chest-ECG R-peak detection check (8 s excerpt from the stress "
                 "condition per subject; dots = detected R-peaks)", fontweight="bold")
    save(fig, "hrv_check_ecg_peaks.png")


# ----------------------------------------------------------------------------- main
def run(data_root: Path, window_lengths=pp.WINDOW_LENGTHS_S) -> dict:
    feats = {w: pd.read_csv(METRICS / f"features_w{w}.csv") for w in window_lengths}
    subjects = pp.find_subjects(data_root)
    chest_rows = {w: [] for w in window_lengths}
    quality_rows, align_rows, examples = [], [], []
    subj_peaks = {}                      # kept for the lag-shifted sensitivity analysis
    for name, pkl in subjects:
        ecg, bvp, label = load_chest_ecg(pkl)
        rp = detect_r_peaks(ecg)
        t, rr, good = rp["t"], rp["rr"], rp["good"]
        # peak quality per condition (RR interval assigned to the label at its end peak)
        lab_at = label[np.minimum((t[1:] * FS_ECG).astype(int), len(label) - 1)]
        for lab, cond in CONDITIONS.items():
            m = lab_at == lab
            quality_rows.append({"subject": name, "condition": cond, "polarity": rp["polarity"],
                                 "n_rr": int(m.sum()), "good_rr_frac": float(good[m].mean()),
                                 "median_rr_s": float(np.median(rr[m]))})
        align_rows.append({"subject": name, **alignment_lag(t, good, bvp, label)})
        # one 8 s stress excerpt for the figure
        s0 = np.flatnonzero(label == 2)[0] / FS_ECG + 120.0
        a, b = int(s0 * FS_ECG), int((s0 + 8) * FS_ECG)
        pk = t[(t >= s0) & (t < s0 + 8)]
        examples.append({"subject": name, "t": np.arange(a, b) / FS_ECG - s0, "ecg": rp["broad"][a:b],
                         "peaks_t": pk - s0, "peaks_y": rp["broad"][(pk * FS_ECG).round().astype(int)],
                         "polarity": rp["polarity"],
                         "good_frac": float(good[lab_at == 2].mean())})
        for w in window_lengths:
            g = feats[w][feats[w].subject == name]
            for _, row in g.iterrows():
                chest_rows[w].append({"subject": name, "t_start_s": row.t_start_s, "t_end_s": row.t_end_s,
                                      **chest_window_values(t, rr, good, row.t_start_s, row.t_end_s)})
            subj_peaks[name] = (t, rr, good)
        print(f"  {name}: {len(t)} R-peaks, polarity {rp['polarity']:+.0f}, good RR {good.mean():.3f}, "
              f"lag {align_rows[-1]['lag_s']:+.2f} s", flush=True)

    quality = pd.DataFrame(quality_rows); quality.to_csv(METRICS / "hrv_check_ecg_quality.csv", index=False)
    align = pd.DataFrame(align_rows); align.to_csv(METRICS / "hrv_check_alignment.csv", index=False)
    plot_peak_examples(examples)

    agr_rows, subj_rows, summary = [], [], {"windows": {}}
    for w in window_lengths:
        ch = pd.DataFrame(chest_rows[w])
        ch.to_csv(METRICS / f"hrv_check_chest_windows_w{w}.csv", index=False)
        m = feats[w].merge(ch, on=["subject", "t_start_s", "t_end_s"], how="left", validate="one_to_one")
        assert len(m) == len(feats[w]) and m.chest_n_rr.notna().all(), "window grids do not match"
        paired = []
        for f in COMPARED:
            ok = m[f].notna() & m[f"chest_{f}"].notna() & m.chest_valid
            p = m.loc[ok, ["subject", "label"]].assign(feature=f, wrist=m.loc[ok, f], chest=m.loc[ok, f"chest_{f}"])
            paired.append(p)
            for cond_lab, cond in list(CONDITIONS.items()) + [(None, "all")]:
                q = p if cond_lab is None else p[p.label == cond_lab]
                agr_rows.append({"window_s": w, "feature": f, "condition": cond,
                                 **agreement(q.wrist.values, q.chest.values, q.subject.values)})
                for s, qs in q.groupby("subject"):
                    d = qs.wrist - qs.chest
                    subj_rows.append({"window_s": w, "feature": f, "condition": cond, "subject": s,
                                      "n_windows": len(qs), "bias": d.mean(), "mae": d.abs().mean(),
                                      "rmse": np.sqrt((d ** 2).mean()),
                                      "pearson": qs.wrist.corr(qs.chest),
                                      "spearman": qs.wrist.corr(qs.chest, method="spearman")})
        paired = pd.concat(paired, ignore_index=True)
        summary["windows"][str(w)] = {
            "n_windows": int(len(m)), "chest_valid": int(m.chest_valid.sum()),
            "chest_invalid": int((~m.chest_valid).sum()),
            "wrist_hr_nan": int(m.hr_mean.isna().sum()),
            "paired_hr": int((paired.feature == "hr_mean").sum()),
        }
        agr_partial = pd.DataFrame(agr_rows)
        plot_bland_altman(paired, agr_partial, w)

    agr = pd.DataFrame(agr_rows); agr.to_csv(METRICS / "hrv_check_agreement.csv", index=False)
    shift = float(align.lag_s.median())
    lag_shifted_agreement(feats, subj_peaks, shift).to_csv(
        METRICS / "hrv_check_agreement_lag_shifted.csv", index=False)
    pd.DataFrame(subj_rows).to_csv(METRICS / "hrv_check_per_subject.csv", index=False)
    for w in window_lengths:
        summary["windows"][str(w)].update(evaluate_criteria(agr, w))
    summary["alignment_lag_s"] = {"min": float(align.lag_s.min()), "max": float(align.lag_s.max()),
                                  "median": float(align.lag_s.median())}
    summary["ecg_good_rr_frac"] = {"min": float(quality.good_rr_frac.min()),
                                   "median": float(quality.good_rr_frac.median())}
    summary["criteria"] = {"hr": HR_CRIT, "hrv": HRV_CRIT}
    (METRICS / "hrv_check_verdict.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder containing WESAD S2 ... S17 (read only)")
    ap.add_argument("--windows", type=int, nargs="+", default=list(pp.WINDOW_LENGTHS_S))
    args = ap.parse_args()
    s = run(Path(args.data), tuple(args.windows))
    for w, v in s["windows"].items():
        print(f"{w} s: verdict = {v['verdict']}")


if __name__ == "__main__":
    main()
