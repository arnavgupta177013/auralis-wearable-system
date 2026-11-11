"""
Feature specification (Step C, G1): the single source of truth for feature names, ORDER, units and definitions.

Every model in Step C takes a subset of SPEC_ORDER, in SPEC_ORDER order (asserted by `assert_spec_order`).
Run from software/ml-models/wesad_edge_model to (re)write feature_spec.json:
    python -m src.feature_spec

Definitions below describe exactly what src/preprocessing.py and src/features.py compute (plus the two
gain-invariant EDA features added in Step C, computed in src/hw_parity.py).
"""

import json
from pathlib import Path

from src import preprocessing as pp

ROOT = Path(__file__).resolve().parents[1]
SPEC_JSON = ROOT / "feature_spec.json"
VERSION = "stepc-1"

WINDOW = {"lengths_s": list(pp.WINDOW_LENGTHS_S), "step_s": pp.STEP_S,
          "grid": "windows start at 0, 5, 10, ... s from the start of the recording; features use only samples "
                  "in [t_start, t_end) after one causal filter pass over the whole stream"}

_BVP = (f"BVP {pp.FS['BVP']} Hz; causal Butterworth band-pass {pp.BVP_BAND_HZ[0]}-{pp.BVP_BAND_HZ[1]} Hz, order "
        f"{pp.BVP_ORDER} per edge, SOS with state carried across the stream (steady-state start). In the window: "
        "scipy find_peaks(distance = int(0.33*fs) samples, prominence = 0.3 x SD of the filtered window); peak time "
        "refined by a 3-point parabola; IBI = successive differences; keep 0.33 s < IBI < 1.5 s; drop IBIs with "
        "|IBI - median| >= 0.3 x median; need >= 5 peaks and >= 4 IBIs else NaN; HR = 60 / IBI")
_EDA_LP = (f"EDA {pp.FS['EDA']} Hz (uS); causal Butterworth low-pass {pp.EDA_LOWPASS_HZ} Hz, order {pp.EDA_ORDER}, "
           "SOS with state")
_PHASIC = (f"phasic = low-passed EDA - tonic, tonic = causal EMA with tau {pp.EDA_TONIC_TAU_S} s "
           f"(a = 1 - exp(-1/(tau*fs)), fs {pp.FS['EDA']} Hz)")
_CAL = ("wearer's calibration period = first 5 min of the rest (baseline) condition; the per-wearer constant is "
        "computed from the stream samples in that period and stored on the device")
_TEMP = "TEMP 4 Hz (degC) -> 1 Hz by block mean of 4 samples (causal)"
_ACC = "ACC 32 Hz, raw / 64 -> g; magnitude = sqrt(x^2 + y^2 + z^2)"

# role: deployable (pool D) | deployable_redefined (G3 gain-invariant) | reference_absolute | excluded_hrv
FEATURES = [
    dict(name="hr_mean", unit="bpm", input="BVP", fs_hz=pp.FS["BVP"], role="deployable",
         definition=_BVP + "; hr_mean = mean of HR over the kept IBIs"),
    dict(name="hr_std", unit="bpm", input="BVP", fs_hz=pp.FS["BVP"], role="deployable",
         definition=_BVP + "; hr_std = population SD (ddof 0) of HR over the kept IBIs"),
    dict(name="rmssd", unit="ms", input="BVP", fs_hz=pp.FS["BVP"], role="excluded_hrv",
         definition=_BVP + "; RMS of successive IBI differences (ms). EXCLUDED: does not track chest ECG (Step A)"),
    dict(name="sdnn", unit="ms", input="BVP", fs_hz=pp.FS["BVP"], role="excluded_hrv",
         definition=_BVP + "; SD (ddof 0) of kept IBIs (ms). EXCLUDED (Step A)"),
    dict(name="pnn50", unit="fraction", input="BVP", fs_hz=pp.FS["BVP"], role="excluded_hrv",
         definition=_BVP + "; fraction of |successive IBI differences| > 50 ms. EXCLUDED (Step A)"),
    dict(name="scl_mean", unit="uS", input="EDA", fs_hz=pp.FS["EDA"], role="reference_absolute",
         definition=_EDA_LP + "; " + _PHASIC + "; scl_mean = mean of the tonic in the window. Absolute level"),
    dict(name="eda_std", unit="uS", input="EDA", fs_hz=pp.FS["EDA"], role="reference_absolute",
         definition=_EDA_LP + "; SD (ddof 0) of the low-passed EDA in the window. Gain-dependent"),
    dict(name="eda_slope", unit="uS/s", input="EDA", fs_hz=pp.FS["EDA"], role="deployable",
         definition=_EDA_LP + "; least-squares slope of the low-passed EDA vs time (s) over the window. "
                              "Offset-invariant, gain-DEPENDENT"),
    dict(name="eda_slope_rel", unit="1/s (calibration SDs per s)", input="EDA", fs_hz=pp.FS["EDA"],
         role="deployable_redefined",
         definition="eda_slope / sigma_lp_cal, sigma_lp_cal = SD (ddof 0) of the low-passed EDA samples over the "
                    "calibration period; " + _CAL + ". Gain- and offset-invariant"),
    dict(name="scr_count", unit="count per window", input="EDA", fs_hz=pp.FS["EDA"], role="deployable",
         definition=_EDA_LP + "; " + _PHASIC + "; number of find_peaks(phasic, prominence >= 0.01 uS, distance >= "
                              "1 s) in the window. Offset-invariant, gain-DEPENDENT (fixed absolute prominence)"),
    dict(name="scr_count_rel", unit="count per window", input="EDA", fs_hz=pp.FS["EDA"],
         role="deployable_redefined",
         definition="as scr_count but prominence >= 1.0 x sigma_ph_cal, sigma_ph_cal = SD (ddof 0) of the phasic "
                    "samples over the calibration period; " + _CAL + ". Gain- and offset-invariant"),
    dict(name="scr_amp_mean", unit="uS", input="EDA", fs_hz=pp.FS["EDA"], role="reference_absolute",
         definition="mean prominence of the scr_count peaks (0 if none). Gain-dependent absolute amplitude"),
    dict(name="temp_mean", unit="degC", input="TEMP", fs_hz=pp.FS["TEMP"], role="reference_absolute",
         definition=_TEMP + "; mean over the window. Absolute level (sensor offset, session drift)"),
    dict(name="temp_slope", unit="degC/s", input="TEMP", fs_hz=pp.FS["TEMP"], role="deployable",
         definition=_TEMP + "; least-squares slope vs time (s) over the window"),
    dict(name="acc_mag_mean", unit="g", input="ACC", fs_hz=pp.FS["ACC"], role="deployable",
         definition=_ACC + "; mean of the magnitude over the window"),
    dict(name="motion_energy", unit="g", input="ACC", fs_hz=pp.FS["ACC"], role="deployable",
         definition=_ACC + "; population SD (ddof 0) of the magnitude over the window"),
]
SPEC_ORDER = [f["name"] for f in FEATURES]
DEPLOYABLE = [f["name"] for f in FEATURES if f["role"] == "deployable"]
REDEFINED = {"eda_slope": "eda_slope_rel", "scr_count": "scr_count_rel"}

PERSONAL_Z = ("(x - mean_cal) / max(sd_cal, floor); mean_cal, sd_cal (ddof 1) over the wearer's calibration "
              "windows (windows fully inside the first 5 min of rest, same length and step); floor = median sd_cal "
              "of the TRAINING subjects (a model constant shipped with the model)")


def ordered(features) -> list:
    """Return `features` in spec order (raises if a name is not in the spec)."""
    unknown = [f for f in features if f not in SPEC_ORDER]
    assert not unknown, f"features not in feature_spec: {unknown}"
    return [f for f in SPEC_ORDER if f in set(features)]


def assert_spec_order(features) -> None:
    assert list(features) == ordered(features), f"feature order differs from feature_spec: {list(features)}"


def spec_dict() -> dict:
    return {"version": VERSION, "order": SPEC_ORDER, "deployable_pool_D": DEPLOYABLE,
            "window": WINDOW, "personal_z": PERSONAL_Z,
            "features": [dict(index=i, window_s=WINDOW["lengths_s"], step_s=WINDOW["step_s"], **f)
                         for i, f in enumerate(FEATURES)]}


def write(path: Path = SPEC_JSON) -> Path:
    path.write_text(json.dumps(spec_dict(), indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(f"wrote {write()}")
