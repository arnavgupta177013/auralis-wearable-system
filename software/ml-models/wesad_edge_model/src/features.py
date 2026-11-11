"""
Window features for the Auralis stress detection model (WESAD wrist data).

Run from software/ml-models/wesad_edge_model:
    python -m src.features --data "C:\\datasets\\WESAD"

Writes, for every window length W in WINDOW_LENGTHS_S:
    results/metrics/features_w{W}.csv            one row per window (raw features + flags)
    results/metrics/hrv_causal_vs_zerophase_w{W}.csv   per-window HR/HRV, causal vs filtfilt
    results/metrics/zscore_investigation_w{W}.csv      calibration SDs, floors, extreme z-scores
and results/metrics/feature_build_summary.json.

Every feature is computed from data inside the window only, after a single causal
filter pass over the recording (see preprocessing.py), so it can be ported to streaming C.
Outliers are never removed: quality problems are flagged in columns flag_*.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps

from src import preprocessing as pp

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"

# ----------------------------------------------------------------------------- feature lists
FEATURES = [
    "hr_mean", "hr_std", "rmssd", "sdnn", "pnn50",                 # PPG
    "scl_mean", "eda_std", "eda_slope", "scr_count", "scr_amp_mean",  # EDA
    "temp_mean", "temp_slope",                                     # skin temperature
    "acc_mag_mean", "motion_energy",                               # motion
]
# Features that do not depend on the absolute EDA level or absolute temperature
# (what the Auralis hardware can provide without µS / °C calibration against the E4).
# Caveat: scr_count uses a 0.01 µS prominence and eda_slope is in µS/s, so both still
# depend on the EDA gain; they are kept because they do not depend on the absolute level.
RELATIVE_FEATURES = [
    "hr_mean", "hr_std", "rmssd", "sdnn", "pnn50",
    "scr_count", "eda_slope", "temp_slope",
    "acc_mag_mean", "motion_energy",
]
QUALITY_COLS = ["ppg_quality", "n_beats", "eda_range"]
NORMALISATIONS = ("universal", "personal_z", "relative_only")

# ----------------------------------------------------------------------------- quality thresholds
PPG_QUALITY_MIN = 0.5        # fraction of beat intervals that pass the artefact check
MIN_BEATS_PER_10S = 5        # fewer beats than this per 10 s -> unusable PPG
MOTION_ENERGY_MAX_G = 0.1    # SD of |acc| over the window, in g
EDA_FLAT_RANGE_US = 0.005    # max - min of raw EDA in the window below this -> flat line
EDA_MIN_US = 0.05            # mean EDA below this -> probably no skin contact

# ----------------------------------------------------------------------------- PPG / HRV
def _parabolic_peak_times(x: np.ndarray, peaks: np.ndarray, fs: float) -> np.ndarray:
    """Sub-sample peak times (s) by fitting a parabola through each peak and its neighbours.
    At 64 Hz one sample is 15.6 ms, so this matters for RMSSD and pNN50."""
    p = peaks[(peaks > 0) & (peaks < len(x) - 1)]
    y0, y1, y2 = x[p - 1], x[p], x[p + 1]
    denom = y0 - 2 * y1 + y2
    offset = np.where(np.abs(denom) > 1e-12, 0.5 * (y0 - y2) / denom, 0.0)
    return (p + np.clip(offset, -0.5, 0.5)) / fs


def ppg_features(bvp_filtered: np.ndarray, fs: float = pp.FS["BVP"]) -> dict:
    """HR / HRV from an already band-passed BVP window."""
    nan = dict(hr_mean=np.nan, hr_std=np.nan, rmssd=np.nan, sdnn=np.nan, pnn50=np.nan,
               ppg_quality=0.0, n_beats=0)
    x = bvp_filtered
    if len(x) < fs * 10 or np.std(x) == 0:
        return nan
    peaks, _ = sps.find_peaks(x, distance=int(0.33 * fs), prominence=0.3 * np.std(x))
    if len(peaks) < 5:
        return dict(nan, n_beats=int(len(peaks)))
    ibi = np.diff(_parabolic_peak_times(x, peaks, fs))
    ibi = ibi[(ibi > 0.33) & (ibi < 1.5)]                 # 40-180 bpm
    if len(ibi) < 4:
        return dict(nan, n_beats=int(len(peaks)))
    med = np.median(ibi)
    good = np.abs(ibi - med) < 0.3 * med                  # reject artefact beats
    quality = float(good.mean())
    ibi = ibi[good]
    if len(ibi) < 4:
        return dict(nan, ppg_quality=quality, n_beats=int(len(peaks)))
    hr = 60.0 / ibi
    d = np.diff(ibi) * 1000.0                             # successive differences, ms
    return dict(
        hr_mean=float(hr.mean()), hr_std=float(hr.std()),
        rmssd=float(np.sqrt(np.mean(d ** 2))), sdnn=float(np.std(ibi * 1000.0)),
        pnn50=float(np.mean(np.abs(d) > 50)), ppg_quality=quality, n_beats=int(len(peaks)),
    )


def ppg_features_zero_phase(bvp_raw_window: np.ndarray, fs: float = pp.FS["BVP"]) -> dict:
    """Same HR/HRV code, but the band-pass is applied with filtfilt on the window
    (as in wesad_baseline). Used only for the causal-vs-zero-phase comparison."""
    if len(bvp_raw_window) < fs * 10:
        return ppg_features(np.array([]), fs)
    sos = pp.bvp_bandpass_sos(fs)
    return ppg_features(sps.sosfiltfilt(sos, bvp_raw_window), fs)


# ----------------------------------------------------------------------------- EDA / TEMP / ACC
def _slope_per_s(x: np.ndarray, fs: float) -> float:
    if len(x) < 3:
        return np.nan
    t = np.arange(len(x)) / fs
    return float(np.polyfit(t, x, 1)[0])


def eda_features(eda_lp, tonic, phasic, eda_raw, fs: float = pp.FS["EDA"]) -> dict:
    pk, props = sps.find_peaks(phasic, prominence=0.01, distance=int(1 * fs))   # SCRs >= 0.01 µS
    return dict(
        scl_mean=float(tonic.mean()),
        eda_std=float(eda_lp.std()),
        eda_slope=_slope_per_s(eda_lp, fs),                       # µS / s
        scr_count=float(len(pk)),
        scr_amp_mean=float(props["prominences"].mean()) if len(pk) else 0.0,
        eda_range=float(eda_raw.max() - eda_raw.min()) if len(eda_raw) else np.nan,
        _eda_mean_raw=float(eda_raw.mean()) if len(eda_raw) else np.nan,
    )


def temp_features(temp, fs: float = pp.FS["TEMP"]) -> dict:
    return dict(temp_mean=float(temp.mean()), temp_slope=_slope_per_s(temp, fs))   # °C, °C/s


def acc_features(acc) -> dict:
    mag = np.linalg.norm(acc, axis=1)
    return dict(acc_mag_mean=float(mag.mean()), motion_energy=float(mag.std()))


def quality_flags(r: dict, win_s: float) -> dict:
    flag_ppg = bool(np.isnan(r["hr_mean"]) or r["ppg_quality"] < PPG_QUALITY_MIN
                    or r["n_beats"] < MIN_BEATS_PER_10S * win_s / 10)
    flag_motion = bool(r["motion_energy"] > MOTION_ENERGY_MAX_G)
    flag_eda_flat = bool(r["eda_range"] < EDA_FLAT_RANGE_US)
    flag_eda_low = bool(r["_eda_mean_raw"] < EDA_MIN_US)
    return dict(flag_ppg=flag_ppg, flag_motion=flag_motion, flag_eda_flat=flag_eda_flat,
                flag_eda_low=flag_eda_low,
                flag_any=flag_ppg or flag_motion or flag_eda_flat or flag_eda_low)


# ----------------------------------------------------------------------------- one subject
def subject_windows(name: str, streams: dict, win_s: float):
    """Returns (features DataFrame, causal-vs-zero-phase DataFrame) for one subject."""
    c0, c1 = pp.calibration_period_s(streams["label"])
    rows, cmp_rows = [], []
    for w in pp.make_windows(streams["label"], win_s):
        seg = pp.slice_window(streams, w["t_start_s"], w["t_end_s"])
        r = {"subject": name, **w, "calib_start_s": c0, "calib_end_s": c1}
        r.update(ppg_features(seg["BVP"]))
        r.update(eda_features(seg["EDA"], seg["EDA_tonic"], seg["EDA_phasic"], seg["EDA_raw"]))
        r.update(temp_features(seg["TEMP"]))
        r.update(acc_features(seg["ACC"]))
        r.update(quality_flags(r, win_s))
        r.pop("_eda_mean_raw")
        rows.append(r)
        zp = ppg_features_zero_phase(seg["BVP_raw"])
        cmp_rows.append({"subject": name, "t_start_s": w["t_start_s"], "label": w["label"],
                         **{f"{k}_causal": r[k] for k in ("hr_mean", "rmssd", "sdnn", "pnn50", "n_beats")},
                         **{f"{k}_zerophase": zp[k] for k in ("hr_mean", "rmssd", "sdnn", "pnn50", "n_beats")}})
    return pd.DataFrame(rows), pd.DataFrame(cmp_rows)


# ----------------------------------------------------------------------------- normalisation
def calibration_sd_floor(df: pd.DataFrame, features=FEATURES) -> pd.DataFrame:
    """Per subject and feature: floor = median calibration SD of the OTHER subjects.

    Uses calibration (rest) windows only, never labels or the subject's own stress data,
    so it cannot leak test information. (The old baseline floored with 0.1 x the subject's
    SD over ALL its windows, which includes that subject's stress windows.)
    """
    cal = df[df.calib]
    sd = cal.groupby("subject")[features].std()
    floors = {}
    for s in sd.index:
        floors[s] = sd.drop(index=s).median()
    return pd.DataFrame(floors).T


def normalise(df: pd.DataFrame, scheme: str) -> tuple:
    """Return (DataFrame with feature columns transformed, list of feature columns used).

    universal     : raw features, all 14
    personal_z    : (x - calib mean) / max(calib SD, floor), all 14
    relative_only : raw values of RELATIVE_FEATURES only (no absolute EDA level / temperature)
    """
    if scheme == "universal":
        return df.copy(), list(FEATURES)
    if scheme == "relative_only":
        return df.copy(), list(RELATIVE_FEATURES)
    if scheme != "personal_z":
        raise ValueError(scheme)
    out = df.copy()
    floor = calibration_sd_floor(df)
    for s, g in df.groupby("subject"):
        cal = g[g.calib]
        assert len(cal) >= 2, f"{s}: fewer than 2 calibration windows"
        mu = cal[FEATURES].mean()
        sd = np.maximum(cal[FEATURES].std().fillna(0), floor.loc[s, FEATURES].fillna(0))
        sd = sd.replace(0, 1.0)                       # only if every subject's calib SD is 0
        out.loc[g.index, FEATURES] = ((g[FEATURES] - mu) / sd).values
    return out, list(FEATURES)


def zscore_investigation(df: pd.DataFrame) -> pd.DataFrame:
    """Why did the baseline produce z-scores of 50-319? Compare no floor, the baseline's floor,
    and the leave-subject-out floor, for the median stress-window z of each subject and feature."""
    floor = calibration_sd_floor(df)
    rows = []
    for s, g in df.groupby("subject"):
        cal = g[g.calib]; st = g[(g.label == 2) & g.scored]
        mu, sd = cal[FEATURES].mean(), cal[FEATURES].std()
        sd_baseline_style = np.maximum(sd.replace(0, np.nan).fillna(1.0),
                                       0.1 * g[FEATURES].std().replace(0, 1.0).fillna(1.0))
        sd_new = np.maximum(sd.fillna(0), floor.loc[s, FEATURES].fillna(0)).replace(0, 1.0)
        for f in FEATURES:
            shift = st[f].median() - mu[f]
            rows.append({
                "subject": s, "feature": f, "n_calib": len(cal),
                "calib_mean": mu[f], "calib_sd": sd[f], "floor_lso": floor.loc[s, f],
                "sd_used": sd_new[f], "floor_applied": bool(floor.loc[s, f] > sd[f]),
                "stress_median_minus_calib_mean": shift,
                "z_stress_nofloor": shift / sd[f] if sd[f] > 0 else np.nan,
                "z_stress_baseline_floor": shift / sd_baseline_style[f],
                "z_stress_lso_floor": shift / sd_new[f],
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main
def build(data_root: Path, window_lengths=pp.WINDOW_LENGTHS_S, out_dir: Path = METRICS) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects = pp.find_subjects(data_root)
    if not subjects:
        raise SystemExit(f"No S*/S*.pkl under {data_root}")
    feats = {w: [] for w in window_lengths}
    cmps = {w: [] for w in window_lengths}
    stream_err = []
    t_start = time.time()
    for name, pkl in subjects:
        raw = pp.load_subject(pkl)
        stream_err.append(pp.check_streaming_equivalence(raw["BVP"][: 64 * 600]))
        streams = pp.preprocess(raw)
        for w in window_lengths:
            f, c = subject_windows(name, streams, w)
            feats[w].append(f); cmps[w].append(c)
        print(f"  {name}: " + ", ".join(f"{w}s={len(feats[w][-1])}" for w in window_lengths), flush=True)
    assert max(stream_err) < 1e-9, f"chunked != one-pass filtering ({max(stream_err)})"

    summary = {"n_subjects": len(subjects), "step_s": pp.STEP_S, "purity": pp.PURITY,
               "calib_min": pp.CALIB_MIN, "streaming_max_abs_err": max(stream_err), "windows": {}}
    for w in window_lengths:
        df = pd.concat(feats[w], ignore_index=True)
        pp.assert_calibration_never_scored(df)
        df.to_csv(out_dir / f"features_w{w}.csv", index=False)
        pd.concat(cmps[w], ignore_index=True).to_csv(out_dir / f"hrv_causal_vs_zerophase_w{w}.csv", index=False)
        zscore_investigation(df).to_csv(out_dir / f"zscore_investigation_w{w}.csv", index=False)
        sc = df[df.scored]
        summary["windows"][str(w)] = {
            "total": int(len(df)), "calib": int(df.calib.sum()),
            "excluded_overlap_calib": int(df.in_calib_period.sum()), "scored": int(len(sc)),
            "scored_by_label": {pp.LABEL_NAMES[k]: int(v) for k, v in sc.label.value_counts().sort_index().items()},
            "scored_stress_fraction": round(float((sc.label == 2).mean()), 4),
        }
    summary["build_seconds"] = round(time.time() - t_start, 1)
    (out_dir / "feature_build_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder containing WESAD S2 ... S17")
    ap.add_argument("--windows", type=int, nargs="+", default=list(pp.WINDOW_LENGTHS_S))
    args = ap.parse_args()
    print(json.dumps(build(Path(args.data), tuple(args.windows)), indent=2))


if __name__ == "__main__":
    main()
