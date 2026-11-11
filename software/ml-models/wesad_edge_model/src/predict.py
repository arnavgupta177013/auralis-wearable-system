"""
Host-side reference predictor for the deployed stress detection model (Step D, config D).

The model files in models/ are written by src/final_model.py:
    stress_lr_D_w30.joblib   sklearn pipeline (median imputer -> standard scaler -> logistic regression) + metadata
    stress_lr_D_w30.json     plain-number export for a C implementation (folded weights, bias, medians, thresholds)
    feature_spec.json        feature names, ORDER, units and exact definitions

Two scorers that must agree:
    proba_joblib(bundle, X)  the sklearn pipeline
    proba_json(coef, X)      pure numpy from the JSON (what a C port computes):
                             x_i = NaN ? median_i : x_i;  z = bias_folded + sum_i w_folded_i * x_i;  p = 1/(1+e^-z)
Both accept ONLY a pandas DataFrame whose columns are exactly the model's feature order; anything else
(wrong order, missing/extra columns, a bare array without names) raises ValueError.

Fresh-process round-trip test (run from software/ml-models/wesad_edge_model):
    python -m src.predict --roundtrip

Inference (Step E; no retraining, everything loaded from models/):
    python -m src.predict --features-csv FILE.csv            rows of model inputs (header = feature_spec order,
                                                             optional leading t_start_s column)
    python -m src.predict --signals CAL_DIR TEST_DIR         raw BVP + EDA (+ optional ACC) CSVs, run through the
                                                             SAME feature code as training (src.features,
                                                             src.hw_parity); calibration = last 300 s of CAL_DIR
    python -m src.predict --demo-wesad S5 [--data C:\\datasets\\WESAD]
                                                             read-only demo on a WESAD (TRAINING) subject
Signal CSVs (one folder each for calibration and test; the test recording must directly follow the calibration
recording, because the causal filters run continuously over both, as in training):
    bvp.csv   column "bvp"   (PPG / BVP, any linear unit; model rate 64 Hz)
    eda.csv   column "eda_uS" (microsiemens, as WESAD E4) or "eda_mV" (ESP32 analogReadMilliVolts; UNVALIDATED)
    acc.csv   columns "acc_x_g,acc_y_g,acc_z_g" in g (optional; used ONLY for the motion check, not by the model)
    Sampling rate: first line "# fs_hz=64" in the file, or --bvp-fs / --eda-fs / --acc-fs (must agree if both).

Output per 30 s window (5 s step): raw score, threshold (OP1), window alert (stress / no stress), and the
4-of-5 smoothed alert (the shipped default). The score is NOT a calibrated probability (class-balanced
training; results/graphs/final_calibration.png): never show it to a wearer as a "probability of stress".
Inputs that fail a validation check are rejected (exit code 2) or the window is withheld, with the reason.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MODEL_STEM = "stress_lr_D_w30"
ALLOWED_ROLES = ("deployable", "deployable_redefined")


# ----------------------------------------------------------------------------- loading and checks
def check_feature_order(order: list, spec: dict) -> None:
    """The model's features must follow feature_spec.json order and be deployable."""
    roles = {f["name"]: f["role"] for f in spec["features"]}
    unknown = [f for f in order if f not in roles]
    if unknown:
        raise ValueError(f"features not in feature_spec.json: {unknown}")
    expected = [f for f in spec["order"] if f in set(order)]
    if list(order) != expected:
        raise ValueError(f"model feature order {list(order)} differs from feature_spec.json order {expected}")
    bad = [f for f in order if roles[f] not in ALLOWED_ROLES]
    if bad:
        raise ValueError(f"non-deployable features in the model: {bad}")


def load(models_dir: Path = MODELS) -> tuple:
    import joblib
    bundle = joblib.load(models_dir / f"{MODEL_STEM}.joblib")
    coef = json.loads((models_dir / f"{MODEL_STEM}.json").read_text(encoding="utf-8"))
    spec = json.loads((models_dir / "feature_spec.json").read_text(encoding="utf-8"))
    check_feature_order(coef["feature_order"], spec)
    if list(bundle["feature_order"]) != list(coef["feature_order"]):
        raise ValueError("joblib and JSON feature orders differ")
    if coef["feature_spec_version"] != spec["version"]:
        raise ValueError("feature_spec version mismatch")
    return bundle, coef, spec


def _matrix(X, order: list) -> np.ndarray:
    if not isinstance(X, pd.DataFrame):
        raise ValueError("input must be a pandas DataFrame with named columns in the model's feature order")
    if list(X.columns) != list(order):
        raise ValueError(f"input columns {list(X.columns)} != model feature order {list(order)}")
    return X.to_numpy(dtype=float)


# ----------------------------------------------------------------------------- scoring
def proba_joblib(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    return bundle["pipeline"].predict_proba(_matrix(X, bundle["feature_order"]))[:, 1]


def proba_json(coef: dict, X: pd.DataFrame, dtype=np.float64) -> np.ndarray:
    x = _matrix(X, coef["feature_order"]).astype(dtype)
    med = np.asarray(coef["imputer_medians"], dtype=dtype)
    x = np.where(np.isnan(x), med, x)
    z = dtype(coef["bias_folded"]) + x @ np.asarray(coef["weights_folded"], dtype=dtype)
    return (dtype(1) / (dtype(1) + np.exp(-z))).astype(dtype)


def alert(p: np.ndarray, threshold: float) -> np.ndarray:
    """Per-window decision: stress if p >= threshold (same rule as the harness)."""
    return np.asarray(p) >= threshold


# ----------------------------------------------------------------------------- round trip
def _expect_rejected(fn, X, what: str) -> None:
    try:
        fn(X)
    except ValueError:
        return
    raise AssertionError(f"wrongly formatted input was NOT rejected: {what}")


def roundtrip(models_dir: Path = MODELS) -> dict:
    bundle, coef, spec = load(models_dir)
    order = coef["feature_order"]
    ref = pd.read_csv(models_dir / "roundtrip_windows.csv")
    X = ref[order]
    p = proba_joblib(bundle, X)
    assert np.array_equal(p, ref.p_ref.values), \
        f"reloaded joblib predictions differ from the in-memory model (max diff {np.max(np.abs(p - ref.p_ref.values))})"
    for op, col in (("threshold_op1", "alert_op1_ref"), ("threshold_op2", "alert_op2_ref")):
        assert np.array_equal(alert(p, coef[op]), ref[col].values.astype(bool)), f"{op} decisions differ"
    pj = proba_json(coef, X)
    d_json = float(np.max(np.abs(pj - p)))
    assert d_json <= 1e-12, f"JSON (numpy) probabilities differ from the pipeline by {d_json}"
    assert np.array_equal(alert(pj, coef["threshold_op1"]), alert(p, coef["threshold_op1"])), "JSON OP1 decisions differ"
    p32 = proba_json(coef, X, np.float32)
    n32 = int((alert(p32, np.float32(coef["threshold_op1"])) != alert(p, coef["threshold_op1"])).sum())

    # feature order: JSON vs feature_spec.json, and rejection of wrongly ordered input
    assert order == [f for f in spec["order"] if f in set(order)], "feature order differs from feature_spec.json"
    _expect_rejected(lambda o: check_feature_order(o, spec), order[::-1], "reversed order vs feature_spec.json")
    swapped = [order[1], order[0]] + order[2:]
    for name, bad in [("reversed columns", X[order[::-1]]), ("first two swapped", X[swapped]),
                      ("bare numpy array", X.to_numpy()), ("missing column", X[order[:-1]]),
                      ("renamed column", X.rename(columns={order[0]: order[0] + "_x"}))]:
        _expect_rejected(lambda z: proba_joblib(bundle, z), bad, f"joblib: {name}")
        _expect_rejected(lambda z: proba_json(coef, z), bad, f"json: {name}")
    out = {"n_windows": int(len(X)), "n_with_nan": int(X.isna().any(axis=1).sum()),
           "joblib_exact_match": True, "json_max_abs_diff": d_json,
           "float32_op1_decision_disagreements": n32, "wrong_order_inputs_rejected": 5 * 2 + 1}
    return out


# ============================================================================= inference (Step E)
BANNER = ("Stress detection (research prototype, lab-stress model trained on WESAD; not panic/anxiety detection, "
          "not a medical diagnosis).")
NOT_VALIDATED = [
    "Finger EDA in mV through the ESP32 ADC (training: E4 wrist EDA in uS). The relative EDA features are invariant "
    "only to LINEAR gain/offset (Step C, G3); ADC nonlinearity and the finger site were never tested.",
    "Fingertip PPG (MAX30102) vs the E4 wrist BVP used in training; hr_mean was never checked on MAX30102 data.",
    "Anything depending on the prototype hardware: sampling rates other than BVP 64 Hz / EDA 4 Hz (resampled here), "
    "the 60 s warm-up rule, the calibration validity rules in this script, sensor placement, real-life stressors.",
    "No prototype or human recording has been used (none may be before IEC approval). Expected performance is the "
    "LOSO result of config D on WESAD (results/metrics/final_results.csv), itself optimistic.",
]
DEFAULT_WESAD = Path(r"C:\datasets\WESAD")
EXIT_REJECTED = 2

# Range of the deployed features over all 6421 WESAD 30 s windows (results/metrics/features_w30.csv +
# features_eda_w30.csv "__clean"; asserted in tests/test_predict.py). Outside it -> note, not rejection.
TRAIN_RANGE = {"hr_mean": (50.193352, 152.995839), "eda_slope_rel": (-0.924785, 2.320958),
               "scr_count_rel": (0.0, 23.0)}
# Validation rules written for this script (NOT validated on any device; WESAD values for reference):
SLOPE_REL_FAR = 10.0          # |eda_slope_rel| > 10 x the largest WESAD |value| -> implausible (bad calibration?)
RAIL_FRAC_SEGMENT = 0.01      # > 1 % of a segment's samples exactly at its min or max -> railed (WESAD ~0.004 %)
RAIL_FRAC_WINDOW = 0.02       # > 2 % of a window's samples at the segment min/max -> window withheld
PPG_RAIL_FRAC_WINDOW = 0.02   # Step F gate G1 (= ppg_gate.RAIL_FRAC): > 2 % of a window's raw BVP samples REPEATING
                              # the WINDOW max/min, or peak-to-peak 0 -> flat/railed/clipped PPG, window withheld.
                              # Step F: caught 100 % of SYNTHETIC flat/railed/clipped windows; 0 of the 17292 WESAD
                              # 30 s windows (5 s grid, whole recordings) flagged (max 0.00104). G2/G3 NOT added.
CAL_MOTION_MAX_FRAC = 0.25    # > 25 % of calibration windows with motion_energy > 0.1 g -> reject (WESAD max 0.185)
CAL_PPG_BAD_MAX_FRAC = 0.5    # > 50 % of calibration windows with flag_ppg -> reject (WESAD max 0.148)
SIGMA_FAR = 10.0              # calibration sigma outside [WESAD min / 10, WESAD max x 10] (uS) -> reject
EDA_US_MAX = 100.0            # uS; WESAD E4 max 15.9 uS. Larger values look like mV or ADC counts
EDA_MV_MAX = 3300.0           # mV; ESP32 analogReadMilliVolts cannot exceed ~3.3 V
EDA_MV_RAIL = 3100.0          # mV; near the 11 dB-attenuation top of the ESP32 ADC (ASSUMED)
EDA_MV_FLAT = 2.0             # mV range per window below this -> flat (ASSUMED: ~2-3 ADC LSB)
ACC_MAG_G = (0.3, 4.0)        # median |acc| at the wrist is ~1 g (WESAD 0.98); outside -> not in g
WARMUP_S = 60.0               # calibration_procedure.md: >= 60 s of filter warm-up before calibrating (untested)


class InputRejected(ValueError):
    """A validation check failed: `check` (rule name), `message` (what is wrong), `todo` (what to do)."""

    def __init__(self, check: str, message: str, todo: str):
        super().__init__(f"{check}: {message}")
        self.check, self.message, self.todo = check, message, todo


def spec_limits(spec: dict, window_s: float) -> dict:
    """Hard limits derived from the feature definitions in feature_spec.json."""
    d = {f["name"]: f for f in spec["features"]}
    m = re.search(r"keep ([0-9.]+) s < IBI < ([0-9.]+) s", d["hr_mean"]["definition"])
    k = re.search(r"distance >= ([0-9.]+) s", d["scr_count"]["definition"])
    assert m and k, "feature_spec.json definitions changed: cannot derive the HR / SCR limits"
    ibi_lo, ibi_hi = float(m.group(1)), float(m.group(2))
    return {"hr_mean": (60.0 / ibi_hi, 60.0 / ibi_lo),        # what the feature code can produce (40-181.8 bpm)
            "hr_physiological": (30.0, 220.0),
            "scr_count_rel_max": window_s / float(k.group(1)),   # peaks >= 1 s apart in one window
            "fs": {"BVP": d["hr_mean"]["fs_hz"], "EDA": d["eda_slope_rel"]["fs_hz"]}}


def load_deployment(models_dir: Path = MODELS) -> dict:
    """Everything inference needs, from models/ only: pipeline, JSON, spec, window, threshold, smoothing,
    calibration procedure. Raises if the files disagree with each other."""
    bundle, coef, spec = load(models_dir)
    win = json.loads((models_dir / "window.json").read_text(encoding="utf-8"))
    sm = coef["smoothing"]
    if (win["window_s"], win["step_s"]) != (coef["window_s"], coef["step_s"]) or \
            list(win["smoothing_m_of_n"]) != [sm["m"], sm["n"]]:
        raise RuntimeError("window.json disagrees with the model JSON (window, step or smoothing)")
    if bundle.get("threshold_op1", coef["threshold_op1"]) != coef["threshold_op1"]:
        raise RuntimeError("joblib and JSON OP1 thresholds differ")
    cal_md = (models_dir / "calibration_procedure.md").read_text(encoding="utf-8")
    cal = coef["calibration"]
    for s in cal["stored_on_device"]:
        if s not in cal_md:
            raise RuntimeError(f"calibration_procedure.md does not describe {s}")
    rates = coef["input_rates_hz"]
    lim = spec_limits(spec, coef["window_s"])
    if lim["fs"] != rates:
        raise RuntimeError(f"input rates in the JSON {rates} differ from feature_spec.json {lim['fs']}")
    return {"bundle": bundle, "coef": coef, "spec": spec, "order": list(coef["feature_order"]),
            "window_s": float(coef["window_s"]), "step_s": float(coef["step_s"]), "m": int(sm["m"]),
            "n": int(sm["n"]), "threshold": float(coef["threshold_op1"]), "cal": cal, "rates": rates,
            "limits": lim, "models_dir": Path(models_dir)}


def score(M: dict, X: pd.DataFrame) -> np.ndarray:
    """Model score from the joblib pipeline; the JSON (C-port) scorer is checked against it every call."""
    p = proba_joblib(M["bundle"], X)
    if len(p):
        d = float(np.max(np.abs(proba_json(M["coef"], X) - p)))
        assert d <= 1e-12, f"JSON scorer disagrees with the pipeline by {d}"
    return p


# ----------------------------------------------------------------------------- (a) features CSV
def _rows_msg(rows: list, limit: int = 8) -> str:
    s = ", ".join(str(r) for r in rows[:limit])
    return s + (f" ... ({len(rows)} rows)" if len(rows) > limit else "")


def read_features_csv(path: Path, M: dict) -> pd.DataFrame:
    """Parse and validate a CSV of model inputs. Header = feature_spec order, optional leading t_start_s."""
    order = M["order"]
    todo_hdr = f"Write the header exactly as: {','.join(order)} (optionally t_start_s first), in feature_spec.json order."
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f) if r and not (len(r) == 1 and not r[0].strip())]
    if not rows:
        raise InputRejected("wrong shape", f"{path} is empty (no header, no rows)", todo_hdr)
    cols = [c.strip() for c in rows[0]]
    allowed = ["t_start_s"] + order
    missing = [c for c in order if c not in cols]
    extra = [c for c in cols if c not in allowed]
    if missing:
        raise InputRejected("missing columns", f"columns {missing} are missing (header: {cols})", todo_hdr)
    if extra or len(set(cols)) != len(cols):
        raise InputRejected("wrong shape", f"unexpected or duplicate columns {extra or cols} (model inputs are "
                                           f"exactly {order})", todo_hdr)
    if cols not in (order, allowed):
        raise InputRejected("wrong column order", f"columns are {cols}; the model needs {order} in this order "
                                                  "(feature_spec.json)", todo_hdr)
    data = rows[1:]
    if not data:
        raise InputRejected("wrong shape", "header only, no data rows", "Add one row per 30 s window.")
    bad = [i + 2 for i, r in enumerate(data) if len(r) != len(cols)]
    if bad:
        raise InputRejected("wrong shape", f"rows with a different number of fields than the header ({len(cols)}): "
                                           f"file lines {_rows_msg(bad)}", "Every row needs one value per column.")
    vals, nonnum = np.full((len(data), len(cols)), np.nan), []
    for i, r in enumerate(data):
        for j, s in enumerate(r):
            s = s.strip()
            if s.lower() in ("", "nan", "na", "null", "none"):
                continue
            try:
                vals[i, j] = float(s)
            except ValueError:
                nonnum.append(f"line {i + 2} {cols[j]}={s!r}")
    if nonnum:
        raise InputRejected("non-numeric value", f"{_rows_msg(nonnum)}", "All values must be plain numbers.")
    df = pd.DataFrame(vals, columns=cols)
    nan = [i + 2 for i in np.flatnonzero(~np.isfinite(vals).all(axis=1))]
    if nan:
        raise InputRejected("missing value (NaN/inf)", f"file lines {_rows_msg(nan)} have empty, NaN or infinite "
                            "values", "Recompute those windows or drop them (a window without enough heart beats "
                                      "cannot be scored by this script).")
    check_feature_values(df, M)
    if "t_start_s" in df:
        t = df.t_start_s.values
        k = t / M["step_s"]
        if np.any(np.diff(t) <= 0) or not np.allclose(k, np.round(k)):
            raise InputRejected("t_start_s invalid", f"t_start_s must increase in multiples of {M['step_s']:g} s",
                                "Use window start times on the 5 s grid, or drop the column (rows are then taken "
                                "as consecutive 5 s steps).")
    else:
        df.insert(0, "t_start_s", np.arange(len(df)) * M["step_s"])
    return df


def check_feature_values(df: pd.DataFrame, M: dict) -> None:
    lim = M["limits"]
    hr = df.hr_mean.values
    lo, hi = lim["hr_mean"]
    plo, phi = lim["hr_physiological"]
    if np.median(hr) > 250:
        raise InputRejected("wrong units (hr_mean)", f"median hr_mean {np.median(hr):.1f}: looks like an inter-beat "
                            "interval in ms; hr_mean must be in bpm", "Convert: bpm = 60000 / IBI_ms.")
    if np.median(hr) < 4:
        raise InputRejected("wrong units (hr_mean)", f"median hr_mean {np.median(hr):.3f}: looks like Hz or "
                            "beats/s; hr_mean must be in bpm", "Convert: bpm = 60 x Hz.")
    out = [i + 2 for i in np.flatnonzero((hr < lo) | (hr > hi))]
    if out:
        raise InputRejected("hr_mean out of range", f"file lines {_rows_msg(out)}: values "
                            f"{_rows_msg([round(float(x), 1) for x in hr[(hr < lo) | (hr > hi)]])} bpm. Physiological "
                            f"range {plo:g}-{phi:g} bpm; the feature code (feature_spec.json IBI rule) can only produce "
                            f"{lo:.1f}-{hi:.1f} bpm", "Check the PPG and the beat detection; hr_mean must be in bpm.")
    sc = df.scr_count_rel.values
    bad = (sc < 0) | (sc != np.round(sc)) | (sc > lim["scr_count_rel_max"])
    if bad.any():
        raise InputRejected("scr_count_rel invalid", f"file lines {_rows_msg([i + 2 for i in np.flatnonzero(bad)])}: "
                            f"must be a whole number 0-{lim['scr_count_rel_max']:g} (peaks >= 1 s apart in a "
                            f"{M['window_s']:g} s window)", "scr_count_rel is a COUNT of SCR peaks per window.")
    sl = df.eda_slope_rel.values
    far = SLOPE_REL_FAR * max(abs(v) for v in TRAIN_RANGE["eda_slope_rel"])
    bad = np.abs(sl) > far
    if bad.any():
        raise InputRejected("eda_slope_rel implausible", f"file lines {_rows_msg([i + 2 for i in np.flatnonzero(bad)])}: "
                            f"|eda_slope_rel| > {far:.1f} calibration SDs per s (WESAD range "
                            f"{TRAIN_RANGE['eda_slope_rel'][0]:g} to {TRAIN_RANGE['eda_slope_rel'][1]:g})",
                            "eda_slope_rel = EDA slope / sigma_lp_cal; check that it was divided by the calibration "
                            "SD and that the calibration was valid.")


def range_notes(df: pd.DataFrame) -> list:
    notes = []
    for _, r in df.iterrows():
        o = [f for f, (lo, hi) in TRAIN_RANGE.items() if np.isfinite(r[f]) and not lo <= r[f] <= hi]
        notes.append(("outside WESAD range: " + ",".join(o)) if o else "")
    return notes


# ----------------------------------------------------------------------------- (b) raw signals
SIGNAL_COLUMNS = {"bvp": [["bvp"]], "eda": [["eda_uS"], ["eda_mV"]], "acc": [["acc_x_g", "acc_y_g", "acc_z_g"]]}


def read_signal_csv(path: Path, kind: str, fs_flag) -> tuple:
    """-> (samples, fs, unit). Header line names the column(s) and therefore the unit."""
    with open(path, encoding="utf-8-sig") as f:
        head, n_comment = [], 0
        for line in f:
            if line.startswith("#"):
                head.append(line); n_comment += 1
            else:
                header = [c.strip() for c in line.strip().split(",")]
                break
        else:
            raise InputRejected("wrong shape", f"{path}: no header row", "First non-comment line = column names.")
    fs_head = None
    for line in head:
        m = re.search(r"fs_hz\s*=\s*([0-9.]+)", line)
        if m:
            fs_head = float(m.group(1))
    if fs_head is None and fs_flag is None:
        raise InputRejected("sampling rate missing", f"{path}: no '# fs_hz=...' line and no --{kind}-fs flag",
                            f"State the sampling rate (model rates: BVP 64 Hz, EDA 4 Hz; ACC any).")
    if fs_head is not None and fs_flag is not None and abs(fs_head - fs_flag) > 1e-9:
        raise InputRejected("sampling rate mismatch", f"{path}: header says {fs_head:g} Hz, --{kind}-fs says "
                            f"{fs_flag:g} Hz", "Give one consistent sampling rate.")
    fs = float(fs_flag if fs_flag is not None else fs_head)
    if fs <= 0 or fs != round(fs):
        raise InputRejected("sampling rate invalid", f"{path}: {fs:g} Hz (whole positive Hz required)",
                            "Use an integer sampling rate.")
    hdr_ok = [h for h in SIGNAL_COLUMNS[kind] if [c.lower() for c in header] == [c.lower() for c in h]]
    if not hdr_ok:
        units = {"bvp": "column 'bvp' (any linear PPG unit)",
                 "eda": "column 'eda_uS' (microsiemens, as WESAD E4) or 'eda_mV' (ESP32 analogReadMilliVolts, "
                        "UNVALIDATED). Raw ADC counts are NOT accepted: convert to mV first",
                 "acc": "columns 'acc_x_g,acc_y_g,acc_z_g' in g (E4: raw/64; MPU6050 +/-2 g: raw/16384)"}[kind]
        raise InputRejected(f"wrong units / header ({kind})", f"{path}: header {header}; expected {units}",
                            "Rename the column(s) to state the unit, after converting to that unit.")
    unit = hdr_ok[0][0].split("_")[-1] if kind == "eda" else ("g" if kind == "acc" else "a.u.")
    try:
        x = pd.read_csv(path, skiprows=n_comment, dtype=float).to_numpy()
    except ValueError as e:
        raise InputRejected("non-numeric value", f"{path}: {e}", "All samples must be plain numbers.")
    if x.shape[1] != len(header):
        raise InputRejected("wrong shape", f"{path}: {x.shape[1]} columns", "One column per header name.")
    if not np.isfinite(x).all():
        raise InputRejected("gaps (too little data)", f"{path}: {int((~np.isfinite(x)).sum())} missing/NaN samples "
                            "(sensor drop-out?)", "Re-record without gaps; this script does not interpolate.")
    return (x[:, 0] if kind != "acc" else x), fs, unit


def _railed(x: np.ndarray) -> float:
    """Fraction of samples REPEATING the segment's max or min value (the single extreme sample itself is not
    counted, so short segments are not flagged by construction)."""
    if not len(x):
        return 0.0
    return float(max(np.sum(x == x.max()), np.sum(x == x.min())) - 1) / len(x)


def ppg_flat_railed(raw_window: np.ndarray) -> bool:
    """Step F gate G1 on one window of RAW BVP samples at the sensor's own rate (before any resampling, which would
    smear a clipped plateau): flat (peak-to-peak 0, or no samples) or railed/clipped (> PPG_RAIL_FRAC_WINDOW of the
    samples repeat the window's own max or min). Same rule as ppg_gate.gate()["G1_flat_railed"]. It does NOT detect
    noise: band-limited noise still passes (test_KNOWN_LIMITATION_band_limited_noise_passes_ppg_quality)."""
    x = np.asarray(raw_window, float)
    if not len(x) or np.ptp(x) <= 0:
        return True
    return _railed(x) > PPG_RAIL_FRAC_WINDOW


def check_signal_units(kind: str, x: np.ndarray, unit: str, where: str) -> None:
    if kind == "eda":
        if unit == "uS":
            if np.nanmax(x) > EDA_US_MAX:
                looks = "raw ADC counts (integers)" if np.all(x == np.round(x)) else "mV"
                raise InputRejected("wrong units (EDA)", f"{where}: EDA up to {np.nanmax(x):.1f} but the column says "
                                    f"uS (WESAD E4 range ~0-16 uS; skin conductance rarely > {EDA_US_MAX:g} uS); "
                                    f"looks like {looks}", "Label mV data 'eda_mV'; convert ADC counts to mV "
                                    "(analogReadMilliVolts) first.")
            if np.nanmin(x) < 0:
                raise InputRejected("wrong units (EDA)", f"{where}: negative conductance ({np.nanmin(x):.3f} uS)",
                                    "EDA in uS must be >= 0; check the conversion.")
        else:
            if np.nanmax(x) > EDA_MV_MAX or np.nanmin(x) < 0:
                raise InputRejected("wrong units (EDA)", f"{where}: EDA {np.nanmin(x):.0f}-{np.nanmax(x):.0f} labelled "
                                    f"mV, outside the ESP32 0-{EDA_MV_MAX:g} mV range; looks like raw ADC counts "
                                    "(0-4095) or uV", "Use analogReadMilliVolts (mV), not analogRead (counts).")
            if np.mean((x >= EDA_MV_RAIL) | (x <= 0)) > RAIL_FRAC_SEGMENT:
                raise InputRejected("saturated / railed signal (EDA)", f"{where}: > {RAIL_FRAC_SEGMENT:g} of samples "
                                    f">= {EDA_MV_RAIL:g} mV or <= 0 mV (ADC at its limit; open circuit?)",
                                    "Check the electrodes and the GSR module gain.")
    if kind == "acc":
        mag = np.median(np.linalg.norm(x, axis=1))
        if not ACC_MAG_G[0] <= mag <= ACC_MAG_G[1]:
            raise InputRejected("wrong units (ACC)", f"{where}: median |acc| = {mag:.3g}; at rest |acc| is ~1 g. Raw "
                                "LSB? (E4: 64 LSB/g; MPU6050 +/-2 g: 16384 LSB/g)", "Convert ACC to g.")
    if kind in ("bvp", "eda"):
        r = _railed(x)
        if r > RAIL_FRAC_SEGMENT:
            raise InputRejected(f"saturated / railed signal ({kind.upper()})", f"{where}: {r:.3f} of samples sit "
                                f"exactly at the segment min or max (limit {RAIL_FRAC_SEGMENT:g}; WESAD ~0.00004)",
                                "The sensor is clipping: reduce LED current / gain, or reseat the sensor.")


def _resample_to(x, fs_in: float, fs_out: float):
    if fs_in == fs_out:
        return x
    from src.hw_parity import _resample
    return _resample(x, int(fs_out), int(fs_in))


def window_starts(t_from: float, t_to: float, M: dict) -> np.ndarray:
    """Window starts on the 5 s grid from the start of the stream, windows fully inside [t_from, t_to]."""
    step, w = M["step_s"], M["window_s"]
    k0 = int(np.ceil(t_from / step - 1e-9))
    starts = []
    k = k0
    while k * step + w <= t_to + 1e-9:
        starts.append(k * step); k += 1
    return np.asarray(starts, float)


def stream_features(bvp, eda, acc, acc_fs, c0: float, c1: float, starts, M: dict, eda_unit: str = "uS",
                    bvp_native=None) -> tuple:
    """The training feature code on one continuous stream (BVP 64 Hz, EDA 4 Hz). Calibration = [c0, c1).
    bvp_native = (samples, fs) of the BVP before resampling, for the flat/railed check (default: bvp at 64 Hz).
    Returns (windows DataFrame with features + quality flags, calibration dict)."""
    from src import features as ft
    from src import hw_parity as hwp
    from src import preprocessing as pp
    fs_b, fs_e = M["rates"]["BVP"], M["rates"]["EDA"]
    bvp_f = pp.CausalSOS(pp.bvp_bandpass_sos(fs_b)).process(bvp)
    lp, tonic, ph = hwp.eda_streams(eda, fs_e)
    i0, i1 = int(round(c0 * fs_e)), int(round(c1 * fs_e))          # = hw_parity.calib_scales
    s_lp, s_ph = float(np.std(lp[i0:i1])), float(np.std(ph[i0:i1]))
    seg_max = {"BVP": (bvp.min(), bvp.max()), "EDA": (eda.min(), eda.max())}
    bvp_n, fs_n = bvp_native if bvp_native is not None else (bvp, fs_b)

    def window(t0):
        t1 = t0 + M["window_s"]
        sb = bvp_f[int(round(t0 * fs_b)):int(round(t1 * fs_b))]
        rb = bvp[int(round(t0 * fs_b)):int(round(t1 * fs_b))]
        r = {"t_start_s": t0, "t_end_s": t1}
        r.update(ft.ppg_features(sb, fs_b))
        f = hwp.eda_window(lp, tonic, ph, eda, t0, t1, s_lp, s_ph, fs_e)
        r.update(eda_slope_rel=f["eda_slope_rel"], scr_count_rel=f["scr_count_rel"], eda_range=f["eda_range"],
                 _eda_mean_raw=f["_eda_mean_raw"])
        if acc is not None:
            r.update(ft.acc_features(acc[int(round(t0 * acc_fs)):int(round(t1 * acc_fs))]))
        else:
            r.update(acc_mag_mean=np.nan, motion_energy=np.nan)
        q = ft.quality_flags(r, M["window_s"])
        if eda_unit == "mV":                                  # features.py EDA thresholds are in uS
            q["flag_eda_flat"], q["flag_eda_low"] = bool(r["eda_range"] < EDA_MV_FLAT), False
        re_ = eda[int(round(t0 * fs_e)):int(round(t1 * fs_e))]
        q["flag_railed"] = bool(max(np.mean(np.isin(rb, seg_max["BVP"])), np.mean(np.isin(re_, seg_max["EDA"])))
                                > RAIL_FRAC_WINDOW)
        q["flag_ppg_flat_railed"] = ppg_flat_railed(bvp_n[int(round(t0 * fs_n)):int(round(t1 * fs_n))])
        r.update(q)
        return r

    rows = pd.DataFrame([window(t) for t in starts]) if len(starts) else pd.DataFrame()
    cal_starts = window_starts(c0, c1, M)
    cal_rows = pd.DataFrame([window(t) for t in cal_starts])
    eda_cal = eda[i0:i1]
    cal = {"c0_s": c0, "c1_s": c1, "sigma_lp_cal": s_lp, "sigma_ph_cal": s_ph, "unit": eda_unit,
           "n_cal_windows": int(len(cal_rows)),
           "frac_motion": float((cal_rows.motion_energy > ft.MOTION_ENERGY_MAX_G).mean()) if acc is not None else None,
           "frac_ppg_bad": float(cal_rows.flag_ppg.mean()) if len(cal_rows) else 1.0,
           "eda_range": float(np.ptp(eda_cal)) if len(eda_cal) else 0.0,
           "eda_mean": float(np.mean(eda_cal)) if len(eda_cal) else np.nan,
           "hr_median": float(cal_rows.hr_mean.median()) if len(cal_rows) else np.nan}
    return rows, cal


def check_calibration(cal: dict, M: dict, warnings: list) -> None:
    from src import features as ft
    ref = M["cal"]["wesad_reference_values_uS"]
    todo = ("Repeat the 5-min calibration: seated, relaxed, hand still, no talking, after >= 60 s of wear "
            "(models/calibration_procedure.md).")
    if cal["c1_s"] - cal["c0_s"] < M["cal"]["duration_s"] - 1e-9:
        raise InputRejected("calibration too short", f"{cal['c1_s'] - cal['c0_s']:.1f} s < "
                            f"{M['cal']['duration_s']} s", todo)
    flat = EDA_MV_FLAT if cal["unit"] == "mV" else ft.EDA_FLAT_RANGE_US
    if cal["eda_range"] < flat:
        raise InputRejected("flat EDA during calibration", f"EDA range over the calibration {cal['eda_range']:.4g} "
                            f"{cal['unit']} < {flat:g} {cal['unit']}: near-constant signal (no skin contact?)",
                            "Check that both electrodes touch the skin, then " + todo[0].lower() + todo[1:])
    if cal["unit"] == "uS" and cal["eda_mean"] < ft.EDA_MIN_US:
        raise InputRejected("EDA too low during calibration", f"mean EDA {cal['eda_mean']:.4f} uS < "
                            f"{ft.EDA_MIN_US:g} uS (no skin contact?)", "Check electrode contact. " + todo)
    for s in ("sigma_lp_cal", "sigma_ph_cal"):                         # the saved (untested) validity rule
        if not np.isfinite(cal[s]) or cal[s] <= 0:
            raise InputRejected("calibration statistics invalid", f"{s} = {cal[s]} (0 or not finite)", todo)
    if cal["frac_ppg_bad"] > CAL_PPG_BAD_MAX_FRAC:
        raise InputRejected("very low PPG quality during calibration", f"{cal['frac_ppg_bad']:.2f} of the calibration "
                            f"windows have unusable PPG (limit {CAL_PPG_BAD_MAX_FRAC:g}; WESAD max 0.148)",
                            "Reseat the PPG sensor (firm contact, no ambient light). " + todo)
    if cal["frac_motion"] is None:
        warnings.append("No ACC given: motion during calibration was NOT checked.")
    elif cal["frac_motion"] > CAL_MOTION_MAX_FRAC:
        raise InputRejected("extreme motion during calibration", f"{cal['frac_motion']:.2f} of the calibration windows "
                            f"have motion_energy > {ft.MOTION_ENERGY_MAX_G:g} g (limit {CAL_MOTION_MAX_FRAC:g}; "
                            "WESAD max 0.185)", todo)
    if cal["unit"] == "uS":
        for s in ("sigma_lp_cal", "sigma_ph_cal"):
            lo, hi = ref[f"{s}_min"], ref[f"{s}_max"]
            if not lo / SIGMA_FAR <= cal[s] <= hi * SIGMA_FAR:
                raise InputRejected("calibration statistics far outside the training range",
                                    f"{s} = {cal[s]:.4g} uS; WESAD range {lo:.4g}-{hi:.4g} uS (limit: 10x outside)",
                                    "Check EDA units and contact. " + todo)
            if not lo <= cal[s] <= hi:
                warnings.append(f"{s} = {cal[s]:.4g} uS is outside the WESAD range {lo:.4g}-{hi:.4g} uS "
                                "(not rejected; less than 10x outside).")
    else:
        warnings.append("EDA in mV: calibration sigmas cannot be compared with the WESAD uS range (UNVALIDATED).")
    if cal["c0_s"] < WARMUP_S - 1e-9:
        warnings.append(f"Only {cal['c0_s']:.0f} s of signal before the 5-min calibration. The saved procedure "
                        f"recommends discarding / waiting at least the first {WARMUP_S:.0f} s after power-on (filter "
                        "warm-up) before calibrating (rule untested). Record >= 360 s for the calibration segment.")


def window_status(rows: pd.DataFrame) -> list:
    out = []
    for _, r in rows.iterrows():
        why = []
        if r.flag_ppg or not np.isfinite(r.hr_mean):
            why.append("low PPG quality")
        if r.get("flag_ppg_flat_railed", False):
            why.append("flat/railed/clipped PPG")
        if r.flag_eda_flat or r.flag_eda_low:
            why.append("flat/low EDA (no skin contact?)")
        if r.flag_railed:
            why.append("saturated/railed signal")
        out.append(why)
    return out


def read_signals(cal_dir: Path, test_dir: Path, fs_flags: dict, M: dict) -> dict:
    sig = {}
    for kind in ("bvp", "eda", "acc"):
        pc, pt = Path(cal_dir) / f"{kind}.csv", Path(test_dir) / f"{kind}.csv"
        if kind == "acc" and not pc.exists() and not pt.exists():
            continue
        for p in (pc, pt):
            if not p.exists():
                raise InputRejected("missing file", f"{p} not found", f"Provide {kind}.csv in both folders "
                                    "(acc.csv is optional but must then be in both).")
        (xc, fc, uc), (xt, ft_, ut) = (read_signal_csv(p, kind, fs_flags.get(kind)) for p in (pc, pt))
        if fc != ft_ or uc != ut:
            raise InputRejected("calibration/test mismatch", f"{kind}: calibration {fc:g} Hz {uc}, test {ft_:g} Hz "
                                f"{ut}", "Use the same sensor settings for both segments.")
        for x, where in ((xc, pc), (xt, pt)):
            check_signal_units(kind, x, uc, str(where))
        sig[kind] = (xc, xt, fc, uc)
    return sig


def run_signals(cal_dir: Path, test_dir: Path, fs_flags: dict, M: dict) -> dict:
    warnings = []
    sig = read_signals(cal_dir, test_dir, fs_flags, M)
    dur = {k: (len(v[0]) / v[2], len(v[1]) / v[2]) for k, v in sig.items()}
    for k in ("bvp", "acc"):
        if k in dur and max(abs(dur[k][0] - dur["eda"][0]), abs(dur[k][1] - dur["eda"][1])) > 1.0:
            raise InputRejected("signal lengths disagree", f"{k} covers {dur[k][0]:.1f} s + {dur[k][1]:.1f} s, EDA "
                                f"{dur['eda'][0]:.1f} s + {dur['eda'][1]:.1f} s (calibration + test)",
                                "All signals must cover the same time span.")
    c1 = dur["eda"][0]
    if c1 < M["cal"]["duration_s"] - 1e-9:
        raise InputRejected("calibration too short", f"calibration segment is {c1:.1f} s; the model needs "
                            f"{M['cal']['duration_s']} s (5 min) of rest", "Record the full 5 minutes (plus >= 60 s "
                            "warm-up) before the test segment.")
    if dur["eda"][1] < M["window_s"]:
        raise InputRejected("too little data", f"test segment is {dur['eda'][1]:.1f} s, shorter than one "
                            f"{M['window_s']:g} s window", "Record at least 30 s after the calibration.")
    rates = {"bvp": M["rates"]["BVP"], "eda": M["rates"]["EDA"]}
    streams = {}
    for k in ("bvp", "eda"):
        xc, xt, fs, _ = sig[k]
        x = np.r_[xc, xt]
        if fs != rates[k]:
            warnings.append(f"{k.upper()} at {fs:g} Hz resampled to the model rate {rates[k]:g} Hz "
                            "(scipy resample_poly; UNVALIDATED).")
        streams[k] = _resample_to(x, fs, rates[k])
    acc, acc_fs = (np.r_[sig["acc"][0], sig["acc"][1]], sig["acc"][2]) if "acc" in sig else (None, None)
    end = len(streams["eda"]) / rates["eda"]
    starts = window_starts(c1, end, M)
    rows, cal = stream_features(streams["bvp"], streams["eda"], acc, acc_fs, c1 - M["cal"]["duration_s"], c1,
                                starts, M, sig["eda"][3], bvp_native=(np.r_[sig["bvp"][0], sig["bvp"][1]],
                                                                      sig["bvp"][2]))
    check_calibration(cal, M, warnings)
    if sig["eda"][3] == "mV":
        warnings.append("EDA in mV: finger EDA through the ESP32 ADC is UNVALIDATED for this model.")
    return predict_rows(rows, M, warnings, cal)


# ----------------------------------------------------------------------------- (c) WESAD demo
# WESAD readme: 0 = not defined / transient, 1 baseline, 2 stress, 3 amusement, 4 meditation, 5-7 to be ignored.
# Only 1-3 were used in training; a window is "mixed" if no label covers >= 90 % of it (training purity rule).
DEMO_LABELS = {0: "transient(0)", 1: "baseline", 2: "stress", 3: "amusement", 4: "meditation(4)"}

def run_demo(subject: str, data_root: Path, M: dict) -> dict:
    from src import preprocessing as pp
    pkl = Path(data_root) / subject / f"{subject}.pkl"
    if not pkl.exists():
        raise InputRejected("demo data not found", f"{pkl} does not exist", "Pass --data with the WESAD folder.")
    raw = pp.load_subject(pkl)                                   # read-only
    warnings = []
    for kind, key in (("bvp", "BVP"), ("eda", "EDA"), ("acc", "ACC")):
        check_signal_units(kind, raw[key], "uS" if kind == "eda" else "", f"{subject} {key}")
    c0, c1 = pp.calibration_period_s(raw["label"])
    dur = len(raw["label"]) / pp.FS_LABEL
    starts = window_starts(c1, dur, M)
    rows, cal = stream_features(raw["BVP"], raw["EDA"], raw["ACC"], pp.FS["ACC"], c0, c1, starts, M, "uS")
    check_calibration(cal, M, warnings)
    lab = raw["label"]
    true = []
    for t0, t1 in zip(rows.t_start_s, rows.t_end_s):
        seg = lab[int(round(t0 * pp.FS_LABEL)):int(round(t1 * pp.FS_LABEL))]
        cnt = np.bincount(seg, minlength=8)
        k = int(cnt.argmax())
        true.append(DEMO_LABELS.get(k, f"ignored({k})") if cnt[k] / len(seg) >= pp.PURITY else "mixed")
    rows["true_label"] = true
    return predict_rows(rows, M, warnings, cal)


# ----------------------------------------------------------------------------- shared output
def predict_rows(rows: pd.DataFrame, M: dict, warnings: list, cal=None) -> dict:
    from src.runner import smooth
    order, thr = M["order"], M["threshold"]
    rows = rows.reset_index(drop=True)
    why = window_status(rows) if "flag_ppg" in rows else [[] for _ in range(len(rows))]
    ok = np.array([not w for w in why], bool)
    s = np.full(len(rows), np.nan)
    if ok.any():
        s[ok] = score(M, rows.loc[ok, order])
    pred = (ok & (s >= thr)).astype(int)                         # withheld windows count as negative
    sm = smooth(pred, np.zeros(len(rows), int), rows.t_start_s.values, M["m"], M["n"], M["step_s"])
    notes = range_notes(rows)
    if "flag_motion" in rows:
        notes = [("; ".join(x for x in (n, "motion > 0.1 g") if x) if fm else n)
                 for n, fm in zip(notes, rows.flag_motion)]
    out = rows[["t_start_s"] + (["t_end_s"] if "t_end_s" in rows else []) + order].copy()
    if "t_end_s" not in out:
        out.insert(1, "t_end_s", out.t_start_s + M["window_s"])
    out["score"] = s
    out["threshold"] = thr
    out["alert"] = np.where(~ok, "WITHHELD", np.where(pred == 1, "stress", "no stress"))
    out["smoothed_alert"] = np.where(sm == 1, "stress", "no stress")
    out["note"] = ["; ".join(w) if w else n for w, n in zip(why, notes)]
    if "true_label" in rows:
        out["true_label"] = rows.true_label.values
    return {"table": out, "warnings": warnings, "cal": cal}


def print_report(res: dict, M: dict, mode: str, print_every: int = 1, subject: str = None) -> None:
    t = res["table"]
    print(f"Model: models/{MODEL_STEM} (config D, logistic regression), inputs {', '.join(M['order'])}; "
          f"{M['window_s']:g} s windows, {M['step_s']:g} s step")
    print(f"Threshold (OP1): {M['threshold']:.4f} on the score. Smoothed alert: stress if >= {M['m']} of the last "
          f"{M['n']} window alerts are stress (shipped default).")
    print("The score is NOT calibrated and is not a probability: only the thresholded alert is meaningful.")
    if mode == "demo":
        print(f"DEMO on WESAD {subject}: {subject} is a TRAINING subject of the deployed model D (D was fitted on all "
              "15 WESAD subjects), so this demo is NOT an accuracy estimate. The true label is shown for the demo only. "
              "Expected performance = the LOSO result of D (results/metrics/final_results.csv).")
    c = res["cal"]
    if c:
        print(f"Calibration: {c['c0_s']:.1f}-{c['c1_s']:.1f} s of the stream, sigma_lp_cal {c['sigma_lp_cal']:.6g} "
              f"{c['unit']}, sigma_ph_cal {c['sigma_ph_cal']:.6g} {c['unit']}, median HR {c['hr_median']:.1f} bpm, "
              f"{c['n_cal_windows']} calibration windows. Checks passed.")
    for w in res["warnings"]:
        print("WARNING: " + w)
    cols = ["t_start_s", "t_end_s"] + M["order"] + ["score", "threshold", "alert", "smoothed_alert"] + \
           (["true_label"] if "true_label" in t else []) + ["note"]
    fmt = {"t_start_s": "{:.0f}".format, "t_end_s": "{:.0f}".format, "hr_mean": "{:.1f}".format,
           "eda_slope_rel": "{:.4f}".format, "scr_count_rel": "{:.0f}".format,
           "score": lambda v: "-" if not np.isfinite(v) else f"{v:.4f}", "threshold": "{:.4f}".format}
    shown = t.iloc[::max(1, print_every)]
    if print_every > 1:
        print(f"(printing every {print_every}th window of {len(t)}; use --print-every 1 or --out for all)")
    print(shown[cols].to_string(index=False, formatters=fmt, na_rep="-"))
    n_w = int((t.alert == "WITHHELD").sum())
    print(f"Windows: {len(t)}; withheld: {n_w}; window alerts 'stress': {int((t.alert == 'stress').sum())}; "
          f"smoothed alerts 'stress': {int((t.smoothed_alert == 'stress').sum())}")
    if "true_label" in t:
        tab = pd.crosstab(t.true_label, t.smoothed_alert)
        print("Smoothed alert by true WESAD label (training subject, NOT an accuracy estimate):")
        print(tab.to_string())
    print("NOT validated:")
    for s in NOT_VALIDATED:
        print("  - " + s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--roundtrip", action="store_true", help="fresh-process round-trip test of models/")
    g.add_argument("--features-csv", metavar="FILE", help="CSV of model inputs (hr_mean,eda_slope_rel,scr_count_rel)")
    g.add_argument("--signals", nargs=2, metavar=("CAL_DIR", "TEST_DIR"),
                   help="folders with bvp.csv, eda.csv (+ optional acc.csv): calibration segment, test segment")
    g.add_argument("--demo-wesad", metavar="SUBJECT", help="read-only demo on a WESAD (training) subject, e.g. S5")
    ap.add_argument("--models", default=str(MODELS))
    ap.add_argument("--data", default=str(DEFAULT_WESAD), help="WESAD folder for --demo-wesad (read-only)")
    ap.add_argument("--bvp-fs", type=float, default=None, help="BVP sampling rate (Hz) if not in the file header")
    ap.add_argument("--eda-fs", type=float, default=None, help="EDA sampling rate (Hz) if not in the file header")
    ap.add_argument("--acc-fs", type=float, default=None, help="ACC sampling rate (Hz) if not in the file header")
    ap.add_argument("--out", default=None, help="also write every window to this CSV")
    ap.add_argument("--print-every", type=int, default=None,
                    help="print every k-th window (default: all; 6 for --demo-wesad)")
    a = ap.parse_args(argv)
    print(BANNER)
    if a.roundtrip:
        out = roundtrip(Path(a.models))
        print("ROUNDTRIP PASS " + json.dumps(out))
        return 0
    M = load_deployment(Path(a.models))
    try:
        if a.features_csv:
            mode, df = "features", read_features_csv(Path(a.features_csv), M)
            res = predict_rows(df, M, [])
        elif a.signals:
            mode = "signals"
            res = run_signals(Path(a.signals[0]), Path(a.signals[1]),
                              {"bvp": a.bvp_fs, "eda": a.eda_fs, "acc": a.acc_fs}, M)
        else:
            mode = "demo"
            res = run_demo(a.demo_wesad, Path(a.data), M)
    except InputRejected as e:
        print(f"INPUT REJECTED - check failed: {e.check}")
        print(f"  Problem: {e.message}")
        print(f"  What to do: {e.todo}")
        print("No prediction was made.")
        return EXIT_REJECTED
    pe = a.print_every if a.print_every is not None else (6 if mode == "demo" else 1)
    print_report(res, M, mode, pe, a.demo_wesad)
    if a.out:
        res["table"].to_csv(a.out, index=False, float_format="%.17g")
        print(f"All windows written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
