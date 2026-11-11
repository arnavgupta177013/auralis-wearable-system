"""
Preprocessing for the Auralis stress detection model (WESAD wrist / Empatica E4).

WESAD "stress" is a lab stressor (TSST). This is stress detection, not panic or anxiety.

What this module does
---------------------
1. Loads one WESAD subject (.pkl) and keeps only the wrist (E4) signals + labels.
2. Converts units and resamples to the rates the model uses:
       BVP 64 Hz (as E4)  EDA 4 Hz (as E4)  TEMP 4 -> 1 Hz  ACC 32 Hz, 1/64 g -> g
   The Auralis device must resample to the same rates (PPG 100->64, EDA 10->4, IMU 50->32).
3. Filters every signal CAUSALLY in a single forward pass over the whole recording,
   exactly as a streaming device would (no filtfilt, no centred filters):
       BVP: Butterworth band-pass 0.7-3.5 Hz (SOS, with state)
       EDA: Butterworth low-pass 1 Hz, then tonic = exponential moving average (EMA),
            phasic = low-passed EDA - tonic
4. Cuts fixed-step windows (length configurable, 5 s step), labels each window by its
   majority WESAD label (purity >= 90 %, labels 1/2/3 only) and marks calibration windows.

Calibration (personal baseline) = first CALIB_MIN minutes of the subject's baseline condition.
  calib          : window lies fully inside that period -> used to compute personal statistics
  in_calib_period: window overlaps that period at all   -> NEVER scored (for any method)
  scored         : not in_calib_period
"""

import pickle
from pathlib import Path

import numpy as np
from scipy import signal as sps

# ----------------------------------------------------------------------------- configuration
FS_E4 = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32}   # rates stored in WESAD
FS = {"BVP": 64, "EDA": 4, "TEMP": 1, "ACC": 32}      # rates the model uses
FS_LABEL = 700                                        # WESAD labels are at the chest rate
ACC_UNIT_G = 1 / 64.0                                 # E4 ACC unit is 1/64 g

WINDOW_LENGTHS_S = (20, 30, 60)
STEP_S = 5
PURITY = 0.90
KEEP_LABELS = (1, 2, 3)                               # 1 baseline, 2 stress, 3 amusement
LABEL_NAMES = {1: "baseline", 2: "stress", 3: "amusement"}
CALIB_MIN = 5

BVP_BAND_HZ = (0.7, 3.5)      # 42-210 bpm
BVP_ORDER = 3                 # order per edge -> 6th-order band-pass
EDA_LOWPASS_HZ = 1.0
EDA_ORDER = 2
EDA_TONIC_TAU_S = 8.0         # EMA time constant for the tonic (slow) EDA level


# ----------------------------------------------------------------------------- causal filters
class CausalSOS:
    """Second-order-sections IIR filter that keeps its state between calls.

    Feeding a signal in one call or in many chunks gives identical output - this is
    what makes it portable to a streaming C implementation (one biquad cascade).
    """

    def __init__(self, sos: np.ndarray):
        self.sos = sos
        self.zi = None

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.zi is None:                       # start in steady state for the first sample
            self.zi = sps.sosfilt_zi(self.sos) * x[0]
        y, self.zi = sps.sosfilt(self.sos, x, zi=self.zi)
        return y


class CausalEMA:
    """y[n] = y[n-1] + a * (x[n] - y[n-1]),  a = 1 - exp(-1 / (tau * fs)).  One line of C."""

    def __init__(self, tau_s: float, fs: float):
        self.a = 1.0 - np.exp(-1.0 / (tau_s * fs))
        self.y = None

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.y is None:
            self.y = x[0]
        b, a = [self.a], [1.0, -(1.0 - self.a)]
        y, zf = sps.lfilter(b, a, x, zi=[(1.0 - self.a) * self.y])
        self.y = y[-1]
        return y


def bvp_bandpass_sos(fs: float = FS["BVP"]) -> np.ndarray:
    return sps.butter(BVP_ORDER, BVP_BAND_HZ, btype="band", fs=fs, output="sos")


def eda_lowpass_sos(fs: float = FS["EDA"]) -> np.ndarray:
    return sps.butter(EDA_ORDER, EDA_LOWPASS_HZ, btype="low", fs=fs, output="sos")


def block_mean(x: np.ndarray, factor: int) -> np.ndarray:
    """Causal decimation: each output sample = mean of the last `factor` input samples."""
    n = len(x) // factor
    return np.asarray(x[: n * factor], dtype=float).reshape(n, factor).mean(axis=1)


# ----------------------------------------------------------------------------- loading
def find_subjects(root: Path) -> list:
    root = Path(root)
    subs = [p for p in root.glob("S*") if p.name[1:].isdigit() and (p / f"{p.name}.pkl").exists()]
    return [(p.name, p / f"{p.name}.pkl") for p in sorted(subs, key=lambda p: int(p.name[1:]))]


def load_subject(pkl_path: Path) -> dict:
    """Raw E4 wrist signals and the 700 Hz label vector, units converted (ACC in g)."""
    with open(pkl_path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    w = d["signal"]["wrist"]
    return {
        "BVP": np.asarray(w["BVP"], dtype=float).ravel(),
        "EDA": np.asarray(w["EDA"], dtype=float).ravel(),             # µS
        "TEMP": np.asarray(w["TEMP"], dtype=float).ravel(),           # °C
        "ACC": np.asarray(w["ACC"], dtype=float) * ACC_UNIT_G,        # g, shape (n, 3)
        "label": np.asarray(d["label"]).ravel().astype(int),
    }


# ----------------------------------------------------------------------------- preprocessing
def preprocess(raw: dict) -> dict:
    """Single causal pass over the whole recording. Returns the streams used by features.py."""
    bvp_f = CausalSOS(bvp_bandpass_sos()).process(raw["BVP"])
    eda_lp = CausalSOS(eda_lowpass_sos()).process(raw["EDA"])
    eda_tonic = CausalEMA(EDA_TONIC_TAU_S, FS["EDA"]).process(eda_lp)
    temp = block_mean(raw["TEMP"], FS_E4["TEMP"] // FS["TEMP"])
    return {
        "BVP_raw": raw["BVP"],          # kept for the zero-phase comparison only
        "BVP": bvp_f,
        "EDA_raw": raw["EDA"],
        "EDA": eda_lp,
        "EDA_tonic": eda_tonic,
        "EDA_phasic": eda_lp - eda_tonic,
        "TEMP": temp,
        "ACC": raw["ACC"],
        "label": raw["label"],
    }


def check_streaming_equivalence(x: np.ndarray, seed: int = 0) -> float:
    """Filter x in one call and in random chunks; return max abs difference (should be ~1e-12)."""
    rng = np.random.default_rng(seed)
    full = CausalSOS(bvp_bandpass_sos()).process(x)
    filt = CausalSOS(bvp_bandpass_sos())
    cuts = np.sort(rng.choice(np.arange(1, len(x)), size=min(50, len(x) - 1), replace=False))
    chunked = np.concatenate([filt.process(c) for c in np.split(x, cuts)])
    ema_full = CausalEMA(EDA_TONIC_TAU_S, 4).process(x)
    ema = CausalEMA(EDA_TONIC_TAU_S, 4)
    ema_chunked = np.concatenate([ema.process(c) for c in np.split(x, cuts)])
    return float(max(np.abs(full - chunked).max(), np.abs(ema_full - ema_chunked).max()))


# ----------------------------------------------------------------------------- windows
def calibration_period_s(label: np.ndarray) -> tuple:
    """(start, end) in seconds of the first CALIB_MIN minutes of the baseline condition."""
    idx = np.flatnonzero(label == 1)
    if len(idx) == 0:
        return (np.nan, np.nan)
    start = idx[0] / FS_LABEL
    return (start, start + CALIB_MIN * 60)


def make_windows(label: np.ndarray, win_s: float, step_s: float = STEP_S) -> list:
    """List of dicts: t_start_s, t_end_s, label, purity, calib flags. Only kept windows."""
    dur = len(label) / FS_LABEL
    c0, c1 = calibration_period_s(label)
    out = []
    t0 = 0.0
    while t0 + win_s <= dur + 1e-9:
        t1 = t0 + win_s
        seg = label[int(round(t0 * FS_LABEL)):int(round(t1 * FS_LABEL))]
        counts = np.bincount(seg, minlength=8)
        lab = int(counts.argmax())
        purity = counts[lab] / len(seg)
        if purity >= PURITY and lab in KEEP_LABELS:
            overlaps_calib = (t0 < c1) and (t1 > c0)
            out.append({
                "t_start_s": t0, "t_end_s": t1, "label": lab, "purity": round(float(purity), 4),
                "calib": bool(lab == 1 and t0 >= c0 and t1 <= c1),
                "in_calib_period": bool(overlaps_calib),
                "scored": bool(not overlaps_calib),
            })
        t0 += step_s
    return out


def slice_window(streams: dict, t0: float, t1: float) -> dict:
    """Cut every stream to [t0, t1) at its own sampling rate."""
    rates = {"BVP_raw": FS["BVP"], "BVP": FS["BVP"], "EDA_raw": FS["EDA"], "EDA": FS["EDA"],
             "EDA_tonic": FS["EDA"], "EDA_phasic": FS["EDA"], "TEMP": FS["TEMP"], "ACC": FS["ACC"]}
    return {k: streams[k][int(round(t0 * fs)):int(round(t1 * fs))] for k, fs in rates.items()}


def assert_calibration_never_scored(df) -> None:
    """Hard check: no scored window may overlap the calibration period or be a calibration window."""
    scored = df[df["scored"]]
    assert not scored["calib"].any(), "calibration window marked as scored"
    assert not scored["in_calib_period"].any(), "window overlapping the calibration period is scored"
    for subj, g in scored.groupby("subject"):
        c0, c1 = g["calib_start_s"].iloc[0], g["calib_end_s"].iloc[0]
        overlap = (g["t_start_s"] < c1) & (g["t_end_s"] > c0)
        assert not overlap.any(), f"{subj}: scored window overlaps calibration period"
