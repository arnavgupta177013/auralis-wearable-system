"""
Step C hardware parity (G2, G3, G4).

Run from software/ml-models/wesad_edge_model:
    python -m src.hw_parity --table                                       G2 feasibility table
    python -m src.hw_parity --eda --data "C:\\datasets\\WESAD"             G3 EDA gain/offset features (20/30/60 s)
    python -m src.hw_parity --device 60 --data "C:\\datasets\\WESAD"       G4 device-rate features for one window
Outputs: results/metrics/hardware_feasibility.csv, features_eda_w{W}.csv, stepc_eda_gain_offset.csv,
         stepc_eda_invariance.csv, features_hw_w{W}__{variant}.csv, stepc_device_feature_shift.csv

Every recomputation first re-derives the E4-rate features with the same code and asserts they equal
features_w{W}.csv, so differences come only from the perturbation / rate change.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps

from src import features as ft
from src import preprocessing as pp

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
EDA_FEATS = ["scl_mean", "eda_std", "eda_slope", "scr_count", "scr_amp_mean"]
REL_FEATS = ["eda_slope_rel", "scr_count_rel"]
GAIN_RANGE = (0.3, 3.0)          # log-uniform
OFFSET_RANGE = (-0.5, 0.5)       # uS, uniform
SCR_REL_K = 1.0                  # prominence >= 1.0 x calibration SD of the phasic EDA
EDA_LSB_US = 20.0 / 4096         # "12-bit equivalent" over a 0-20 uS full scale (WESAD max 15.9 uS)
TEMP_STEP_C = 0.1
SEED = 0


# ----------------------------------------------------------------------------- G2
def feasibility_table() -> pd.DataFrame:
    P = "fingertip on prototype, wrist in training data"
    E = "finger on prototype, wrist in training data"
    ppg = ("MAX30102 IR channel over I2C (100 Hz; or decimate). Causal biquad band-pass 0.7-3.5 Hz designed at the "
           "device rate; peak detection (0.33 s refractory, prominence 0.3 x window SD), parabolic refinement, IBI "
           "rules as feature_spec.json; HR = 60/IBI over the last W s")
    gsr = ("GSR module on GPIO34, analogReadMilliVolts at >= 4 Hz (mV, ESP32 ADC nonlinear; NOT uS). Causal 1 Hz "
           "biquad low-pass at 4 Hz, EMA tonic (tau 8 s), phasic = LP - tonic")
    tmp = "MAX30205 over I2C (16-bit, 0.0039 degC LSB), read at 1 Hz"
    imu = "MPU6050 accelerometer over I2C at 50 Hz, +/-2 g (16384 LSB/g) -> g; magnitude sqrt(x^2+y^2+z^2)"
    rows = [
        ("hr_mean", ppg + "; mean HR", "no (time only)", "UNVALIDATED", P,
         "E4 wrist HR tracks chest ECG within person (rmcorr 0.86 at 20 s) but stress LoA +/-25-29 bpm (Step A). "
         "Never checked on MAX30102 fingertip data."),
        ("hr_std", ppg + "; SD of HR", "no", "UNVALIDATED", P,
         "Did not track chest ECG in stress (Spearman 0.06, Step A); kept only as an on/off factor."),
        ("rmssd", ppg + "; RMSSD", "no", "NOT TRANSFERABLE", P,
         "Excluded: wrist RMSSD rmcorr 0.05 with chest ECG, wrong sign under stress (Step A); prototype serial log "
         "RMSSD 250-337 ms at HR ~70 (reported by the user), not physiological."),
        ("sdnn", ppg + "; SDNN", "no", "NOT TRANSFERABLE", P, "Excluded (Step A)."),
        ("pnn50", ppg + "; pNN50", "no", "NOT TRANSFERABLE", P, "Excluded (Step A)."),
        ("scl_mean", gsr + "; mean tonic", "yes (uS level)", "NOT TRANSFERABLE", E,
         "Absolute level: mV is not uS without a per-device conductance calibration; site differs."),
        ("eda_std", gsr + "; SD of LP EDA", "yes (uS gain)", "NOT TRANSFERABLE", E, "Gain-dependent (G3)."),
        ("eda_slope", gsr + "; LS slope", "yes (uS/s gain)", "NOT TRANSFERABLE", E,
         "Offset-invariant but gain-dependent (G3). Usable only through personal z or as eda_slope_rel."),
        ("eda_slope_rel", gsr + "; LS slope / SD of LP EDA over the wearer's calibration period", "no",
         "UNVALIDATED", E, "Gain/offset-invariant by construction; confirmed for linear gain+offset (G3). ADC "
                           "nonlinearity and finger site not tested."),
        ("scr_count", gsr + "; phasic peaks with prominence >= 0.01 uS", "yes (fixed uS prominence)",
         "NOT TRANSFERABLE", E, "Fixed absolute prominence -> gain-dependent (G3)."),
        ("scr_count_rel", gsr + "; phasic peaks with prominence >= 1.0 x calibration SD of phasic EDA", "no",
         "UNVALIDATED", E, "Gain/offset-invariant by construction; confirmed for linear gain+offset (G3)."),
        ("scr_amp_mean", gsr + "; mean SCR prominence", "yes (uS)", "NOT TRANSFERABLE", E, "Absolute amplitude."),
        ("temp_mean", tmp + "; mean", "yes (degC level)", "NOT TRANSFERABLE",
         "contact sensor (prototype) vs IR thermopile at the wrist (training)",
         "Sensor offset differs; WESAD temp drifts with session time (shortcut)."),
        ("temp_slope", tmp + "; LS slope over the window", "degC scale only (MAX30205 is calibrated)", "UNVALIDATED",
         "contact sensor (prototype) vs IR thermopile at the wrist (training)",
         "Offset-free; contact vs IR sensor dynamics differ; WESAD temp slope also follows the session (Stage 2)."),
        ("acc_mag_mean", imu + "; mean magnitude", "g scale (MPU6050 calibrated)", "UNVALIDATED",
         "breadboard (prototype) vs wrist (training)", "Placement differs; E4 range +/-2 g at 32 Hz."),
        ("motion_energy", imu + "; SD of magnitude", "g scale (MPU6050 calibrated)", "UNVALIDATED",
         "breadboard (prototype) vs wrist (training)",
         "In WESAD it mostly marks TSST speaking/gesturing (Stage 2), not physiology."),
    ]
    return pd.DataFrame(rows, columns=["feature", "esp32_computation", "needs_absolute_units", "status", "site",
                                       "notes"])


# ----------------------------------------------------------------------------- shared helpers
def eda_streams(eda: np.ndarray, fs: float = pp.FS["EDA"]) -> tuple:
    lp = pp.CausalSOS(pp.eda_lowpass_sos(fs)).process(eda)
    tonic = pp.CausalEMA(pp.EDA_TONIC_TAU_S, fs).process(lp)
    return lp, tonic, lp - tonic


def calib_scales(lp, phasic, label, fs=pp.FS["EDA"]) -> tuple:
    c0, c1 = pp.calibration_period_s(label)
    i0, i1 = int(round(c0 * fs)), int(round(c1 * fs))
    return float(np.std(lp[i0:i1])), float(np.std(phasic[i0:i1]))


def eda_window(lp, tonic, phasic, raw, t0, t1, s_lp, s_ph, fs=pp.FS["EDA"]) -> dict:
    i0, i1 = int(round(t0 * fs)), int(round(t1 * fs))
    f = ft.eda_features(lp[i0:i1], tonic[i0:i1], phasic[i0:i1], raw[i0:i1], fs)
    pk, _ = sps.find_peaks(phasic[i0:i1], prominence=SCR_REL_K * s_ph, distance=int(1 * fs))
    f.update(scr_count_rel=float(len(pk)), eda_slope_rel=f["eda_slope"] / s_lp if s_lp > 0 else np.nan)
    return f


def _check_equal(new: pd.DataFrame, ref: pd.DataFrame, cols: list, what: str) -> None:
    for c in cols:
        a, b = new[c].values.astype(float), ref[c].values.astype(float)
        ok = np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True)
        assert ok, f"{what}: recomputed {c} differs from features csv (max diff {np.nanmax(np.abs(a - b))})"


# ----------------------------------------------------------------------------- G3
def build_eda(data_root: Path, windows=pp.WINDOW_LENGTHS_S) -> None:
    feats = {w: pd.read_csv(METRICS / f"features_w{w}.csv") for w in windows}
    rng = np.random.default_rng(SEED)
    out = {w: [] for w in windows}
    gains = []
    for name, pkl in pp.find_subjects(data_root):
        raw = pp.load_subject(pkl)
        gain = float(np.exp(rng.uniform(*np.log(GAIN_RANGE))))
        offset = float(rng.uniform(*OFFSET_RANGE))
        versions = {"clean": raw["EDA"], "pert": gain * raw["EDA"] + offset}
        streams = {}
        for v, eda in versions.items():
            lp, tonic, ph = eda_streams(eda)
            streams[v] = (lp, tonic, ph, eda, *calib_scales(lp, ph, raw["label"]))
        gains.append({"subject": name, "gain": gain, "offset_uS": offset,
                      "sigma_lp_cal_clean": streams["clean"][4], "sigma_ph_cal_clean": streams["clean"][5],
                      "sigma_lp_cal_pert": streams["pert"][4], "sigma_ph_cal_pert": streams["pert"][5]})
        for w in windows:
            g = feats[w][feats[w].subject == name]
            rows = []
            for t0, t1 in zip(g.t_start_s, g.t_end_s):
                r = {"subject": name, "t_start_s": t0}
                for v, (lp, tonic, ph, eda, s_lp, s_ph) in streams.items():
                    f = eda_window(lp, tonic, ph, eda, t0, t1, s_lp, s_ph)
                    r.update({f"{k}__{v}": f[k] for k in EDA_FEATS + REL_FEATS})
                rows.append(r)
            d = pd.DataFrame(rows, index=g.index)
            _check_equal(d.rename(columns={f"{k}__clean": k for k in EDA_FEATS}), g, EDA_FEATS, f"{name} w{w}")
            out[w].append(d)
        print(f"  {name}: gain {gain:.3f} offset {offset:+.3f} uS", flush=True)
    inv = []
    for w in windows:
        d = pd.concat(out[w]).sort_index()
        d.to_csv(METRICS / f"features_eda_w{w}.csv", index=True, index_label="row")
        for k in EDA_FEATS + REL_FEATS:
            a, b = d[f"{k}__clean"], d[f"{k}__pert"]
            sd = a.std()
            diff = (b - a).abs() / (sd if sd > 0 else 1.0)
            inv.append({"window_s": w, "feature": k, "max_abs_diff_over_sd": float(diff.max()),
                        "median_abs_diff_over_sd": float(diff.median()),
                        "frac_windows_changed": float((diff > 1e-6).mean()),
                        "pearson_clean_vs_pert": float(a.corr(b)), "spearman_clean_vs_pert": float(a.corr(b, "spearman")),
                        "changes": bool(diff.max() > 1e-6)})
    pd.DataFrame(gains).to_csv(METRICS / "stepc_eda_gain_offset.csv", index=False)
    inv = pd.DataFrame(inv)
    inv.to_csv(METRICS / "stepc_eda_invariance.csv", index=False)
    print(inv.round(4).to_string(index=False))


# ----------------------------------------------------------------------------- G4
DEVICE_VARIANTS = {
    "e4_identity": {},                                         # self-check: must equal features csv
    "eda_12bit": {"eda_q": True},
    "temp_0.1C": {"temp_q": True},
    "bvp_32Hz": {"bvp_fs": 32},
    "bvp_100Hz": {"bvp_fs": 100},
    "acc_50Hz": {"acc_fs": 50},
    "device_all": {"eda_q": True, "temp_q": True, "bvp_fs": 100, "acc_fs": 50},
}
ALL_FEATS = ft.FEATURES


def _resample(x, fs_new, fs_old):
    if fs_new == fs_old:
        return x
    from fractions import Fraction
    fr = Fraction(fs_new, fs_old)
    return sps.resample_poly(x, fr.numerator, fr.denominator, axis=0)


def device_windows(raw: dict, g: pd.DataFrame, win_s: float, opt: dict) -> pd.DataFrame:
    fs_b, fs_a = opt.get("bvp_fs", pp.FS["BVP"]), opt.get("acc_fs", pp.FS["ACC"])
    eda = np.round(raw["EDA"] / EDA_LSB_US) * EDA_LSB_US if opt.get("eda_q") else raw["EDA"]
    temp4 = np.round(raw["TEMP"] / TEMP_STEP_C) * TEMP_STEP_C if opt.get("temp_q") else raw["TEMP"]
    bvp = pp.CausalSOS(pp.bvp_bandpass_sos(fs_b)).process(_resample(raw["BVP"], fs_b, pp.FS["BVP"]))
    acc = _resample(raw["ACC"], fs_a, pp.FS["ACC"])
    lp, tonic, ph = eda_streams(eda)
    temp = pp.block_mean(temp4, pp.FS_E4["TEMP"] // pp.FS["TEMP"])
    rows = []
    for t0, t1 in zip(g.t_start_s, g.t_end_s):
        sl = lambda x, fs: x[int(round(t0 * fs)):int(round(t1 * fs))]
        r = {}
        r.update(ft.ppg_features(sl(bvp, fs_b), fs_b))
        r.update(ft.eda_features(sl(lp, 4), sl(tonic, 4), sl(ph, 4), sl(eda, 4)))
        r.update(ft.temp_features(sl(temp, pp.FS["TEMP"])))
        r.update(ft.acc_features(sl(acc, fs_a)))
        r.update(ft.quality_flags(r, win_s))
        r.pop("_eda_mean_raw")
        rows.append(r)
    return pd.DataFrame(rows, index=g.index)


def build_device(data_root: Path, win: int) -> None:
    feats = pd.read_csv(METRICS / f"features_w{win}.csv")
    meta = [c for c in feats.columns if c not in ALL_FEATS + ft.QUALITY_COLS and not c.startswith("flag_")]
    out = {v: [] for v in DEVICE_VARIANTS}
    for name, pkl in pp.find_subjects(data_root):
        raw = pp.load_subject(pkl)
        g = feats[feats.subject == name]
        for v, opt in DEVICE_VARIANTS.items():
            d = device_windows(raw, g, win, opt)
            if v == "e4_identity":
                _check_equal(d, g, ALL_FEATS, f"{name} e4_identity")
            out[v].append(pd.concat([g[meta], d], axis=1))
        print(f"  {name} done", flush=True)
    shift = []
    for v in DEVICE_VARIANTS:
        d = pd.concat(out[v]).sort_index()
        d.to_csv(METRICS / f"features_hw_w{win}__{v}.csv", index=True, index_label="row")
        for k in ALL_FEATS:
            a, b = feats[k], d[k]
            sd = a.std()
            diff = (b - a).abs()
            shift.append({"window_s": win, "variant": v, "feature": k,
                          "median_abs_diff": float(diff.median()), "p95_abs_diff": float(diff.quantile(0.95)),
                          "median_abs_diff_over_sd": float(diff.median() / sd) if sd > 0 else np.nan,
                          "pearson_vs_e4": float(a.corr(b)), "n_nan_e4": int(a.isna().sum()),
                          "n_nan_variant": int(b.isna().sum())})
    shift = pd.DataFrame(shift)
    shift.to_csv(METRICS / "stepc_device_feature_shift.csv", index=False)
    print(shift[shift.variant != "e4_identity"].round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--eda", action="store_true")
    ap.add_argument("--device", type=int, default=None, help="window length for the G4 device-rate features")
    ap.add_argument("--data", default=None)
    a = ap.parse_args()
    if a.table:
        t = feasibility_table()
        t.to_csv(METRICS / "hardware_feasibility.csv", index=False)
        print(t[["feature", "needs_absolute_units", "status", "site"]].to_string(index=False))
    if a.eda:
        build_eda(Path(a.data))
    if a.device:
        build_device(Path(a.data), a.device)


if __name__ == "__main__":
    main()
