"""
Streaming reference implementation of the three model-D features (stress detection; WESAD lab stressor, NOT panic or
anxiety), written so that every step ports line-for-line to C on the ESP32.  NOT firmware: a Python model of it.

    hr_mean        BVP  -> causal band-pass biquad cascade -> ring buffer (30 s) -> peaks -> IBIs -> mean HR
    eda_slope_rel  EDA  -> causal low-pass biquad -> ring buffer (30 s) -> least-squares slope / sigma_lp_cal
    scr_count_rel  EDA  -> low-pass -> EMA tonic -> phasic ring buffer (30 s) -> peaks (prominence >= sigma_ph_cal)
    sigma_lp_cal, sigma_ph_cal: Welford running mean/variance over the 5-min calibration (no buffer of 1200 samples)

Per sample only: one biquad cascade step per signal, one EMA step, one ring-buffer write, (during calibration) one
Welford update.  Every 5 s (window step), once 30 s of samples exist: window features from the ring buffers.
The arithmetic type is a parameter: T=float (IEEE double, as the offline code) or T=numpy.float32 (as the ESP32 FPU).
scipy is used ONLY at design time (filter coefficients, steady-state initial conditions = constants in a C header);
nothing in the per-sample or per-window path calls numpy/scipy.

DEVIATIONS from the offline code (src/preprocessing.py, src/features.py, src/hw_parity.py), all deliberate:
  D1  find_peaks: hand-written port of scipy.signal.find_peaks with the SAME rules and order (local maxima with
      plateau midpoints -> distance selection by height priority -> prominence with wlen=None).  The one known
      difference: scipy orders equal-height peaks with numpy argsort (quicksort, not stable for > 16 peaks); the port
      uses a stable order (height, then index).  Only matters if two local maxima within `distance` have bit-identical
      heights.
  D2  Window SD / mean / median / slope: sequential sums (C loops) instead of numpy pairwise summation and
      np.polyfit (QR least squares); differences are rounding-level.  The window SD is two-pass over the ring buffer
      (mean first, then squared deviations), NOT a running sum with add-new/subtract-old, because that drifts in
      float32 over hours.
  D3  Calibration SDs by Welford's running algorithm instead of np.std over a stored 1200-sample array.
  D4  Before the calibration period has ended, eda_slope_rel and scr_count_rel are NaN ("calibrating").  The offline
      code computes them for calibration-period windows with sigmas taken from the whole period (i.e. partly from the
      future).  Those windows are never scored (in_calib_period), so no scored window is affected.
  D5  Ring buffers are read in time order through (head + i) mod N; in this Python model they are copied to a list
      once per window for readability (a C port indexes the ring directly; same values).
  D6  Window alert computed as logit z >= logit_threshold (no exp() on the device) instead of p >= threshold.  Both
      are evaluated and any disagreement is counted (src/stream_ref.py parity run).
  D7  Smoothing on the device runs over EVERY 5 s window of the stream; the offline evaluation (runner.smooth) runs
      over the labelled table, where unlabelled/missing windows count as negative.  Compared separately.

Run (from software/ml-models/wesad_edge_model, data read-only):
    python -m src.stream_ref --data "C:\\datasets\\WESAD"          parity, float64 and float32, all 15 subjects
"""

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src import preprocessing as pp

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
MODELS = ROOT / "models"

FS_BVP, FS_EDA = pp.FS["BVP"], pp.FS["EDA"]
WIN_S, STEP_S = 30, 5
PEAK_DIST_BVP = int(0.33 * FS_BVP)          # 21 samples (features.ppg_features)
PEAK_PROM_K_BVP = 0.3                       # prominence >= 0.3 x SD of the filtered window
PEAK_DIST_EDA = int(1 * FS_EDA)             # 4 samples (hw_parity.eda_window)
SCR_REL_K = 1.0                             # prominence >= 1.0 x sigma_ph_cal (hw_parity.SCR_REL_K)
IBI_MIN_S, IBI_MAX_S, IBI_DEV = 0.33, 1.5, 0.3
MIN_PEAKS, MIN_IBIS = 5, 4


# ============================================================================= design-time constants (-> C header)
def design_constants() -> dict:
    """Filter coefficients and steady-state initial-condition vectors (zi per unit input), as plain lists."""
    from scipy import signal as sps
    sb, se = pp.bvp_bandpass_sos(FS_BVP), pp.eda_lowpass_sos(FS_EDA)
    return {"bvp_sos": sb.tolist(), "bvp_zi": sps.sosfilt_zi(sb).tolist(),
            "eda_sos": se.tolist(), "eda_zi": sps.sosfilt_zi(se).tolist(),
            "ema_a": float(1.0 - np.exp(-1.0 / (pp.EDA_TONIC_TAU_S * FS_EDA)))}


# ============================================================================= per-sample building blocks
class Biquads:
    """Cascade of second-order sections, transposed direct form II, with explicit state (2 numbers per section).
    Same operation order as scipy's sosfilt, so float64 output matches it to the last bit or close to it.
    First sample: state = zi_unit * x0 (scipy sosfilt_zi steady-state start, as preprocessing.CausalSOS)."""

    def __init__(self, sos, zi_unit, T=float):
        self.T = T
        self.c = [[T(v) for v in (s[0], s[1], s[2], s[4], s[5])] for s in sos]   # b0 b1 b2 a1 a2 (a0 = 1)
        self.zi_unit = [[T(v) for v in z] for z in zi_unit]
        self.z = None

    def step(self, x):
        if self.z is None:
            self.z = [[zu[0] * x, zu[1] * x] for zu in self.zi_unit]
        for (b0, b1, b2, a1, a2), z in zip(self.c, self.z):
            y = b0 * x + z[0]
            z[0] = b1 * x - a1 * y + z[1]
            z[1] = b2 * x - a2 * y
            x = y
        return x


class EMA:
    """y[n] = a*x[n] + (1-a)*y[n-1], y[-1] = x[0]  (= preprocessing.CausalEMA / scipy lfilter, same op order)."""

    def __init__(self, a, T=float):
        self.a, self.b = T(a), T(1.0 - a)
        self.y = None

    def step(self, x):
        if self.y is None:
            self.y = x
        self.y = self.a * x + self.b * self.y
        return self.y


class Ring:
    """Fixed-size ring buffer (C: static T buf[N]; uint16 head; uint32 count)."""

    def __init__(self, n, T=float):
        self.n, self.buf, self.head, self.count = n, [T(0)] * n, 0, 0

    def push(self, x):
        self.buf[self.head] = x
        self.head = (self.head + 1) % self.n
        self.count += 1

    def ordered(self) -> list:
        h = self.head
        return self.buf[h:] + self.buf[:h]          # C: buf[(head + i) % N]


class Welford:
    """Running mean / population variance (ddof 0). C: three numbers (n, mean, M2)."""

    def __init__(self, T=float):
        self.T, self.n, self.mean, self.m2 = T, 0, T(0), T(0)

    def push(self, x):
        self.n += 1
        d = x - self.mean
        self.mean = self.mean + d / self.T(self.n)
        self.m2 = self.m2 + d * (x - self.mean)

    def sd(self):
        return self.T(math.sqrt(self.m2 / self.T(self.n))) if self.n else self.T(float("nan"))


# ============================================================================= per-window building blocks
def find_peaks_port(x: list, distance: int, prominence_min, ops: dict = None, tie_order: str = "stable") -> list:
    """Port of scipy.signal.find_peaks(x, distance=distance, prominence=prominence_min) (wlen=None, no height).
    Same three stages in the same order as scipy 1.17: _local_maxima_1d, _select_by_peak_distance, peak_prominences +
    _select_by_property.  Returns sorted peak indices.  `ops` (optional) counts inner-loop iterations."""
    n = len(x)
    # 1. local maxima; a flat plateau counts once, at its midpoint (rounded down)
    peaks = []
    i, i_max = 1, n - 1
    while i < i_max:
        if x[i - 1] < x[i]:
            ia = i + 1
            while ia < i_max and x[ia] == x[i]:
                ia += 1
            if x[ia] < x[i]:
                peaks.append((i + ia - 1) // 2)
                i = ia
        i += 1
    if ops is not None:
        ops["scan"] += n
        ops["local_max"] += len(peaks)
    # 2. distance: visit peaks from highest to lowest; each kept peak removes lower neighbours closer than distance
    m = len(peaks)
    keep = [True] * m
    if tie_order == "numpy":                   # test only: reproduce scipy's (unstable) argsort tie order exactly
        order = list(np.argsort(np.asarray([x[p] for p in peaks], dtype=float)))
    else:                                      # stable (D1); C: insertion sort of <= n/2 indices
        order = sorted(range(m), key=lambda k: (x[peaks[k]], k))
    for j in reversed(order):
        if not keep[j]:
            continue
        k = j - 1
        while k >= 0 and peaks[j] - peaks[k] < distance:
            keep[k] = False; k -= 1
        k = j + 1
        while k < m and peaks[k] - peaks[j] < distance:
            keep[k] = False; k += 1
    peaks = [p for p, kp in zip(peaks, keep) if kp]
    # 3. prominence: walk left/right while samples are <= the peak, track the minimum on each side
    out = []
    steps = 0
    for p in peaks:
        xp = x[p]
        i, lmin = p, xp
        while i >= 0 and x[i] <= xp:
            if x[i] < lmin:
                lmin = x[i]
            i -= 1
        steps += p - i
        i, rmin = p, xp
        while i <= n - 1 and x[i] <= xp:
            if x[i] < rmin:
                rmin = x[i]
            i += 1
        steps += i - p
        if prominence_min <= xp - (lmin if lmin > rmin else rmin):
            out.append(p)
    if ops is not None:
        ops["after_distance"] += len(peaks)
        ops["prominence_steps"] += steps
        ops["sort_n"] += m
    return out


def mean_sd(x: list, T=float) -> tuple:
    """Two-pass mean and population SD (D2)."""
    n = len(x)
    s = T(0)
    for v in x:
        s = s + v
    mu = s / T(n)
    q = T(0)
    for v in x:
        d = v - mu
        q = q + d * d
    return mu, T(math.sqrt(q / T(n)))


def median_sorted(v: list, T=float):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / T(2)


def hr_window(x: list, T=float, fs: int = FS_BVP, ops: dict = None) -> dict:
    """hr_mean (+ ppg_quality, n_beats for the existing quality flag) from one band-passed BVP window.
    Same rules as features.ppg_features."""
    nan = T(float("nan"))
    res = {"hr_mean": nan, "ppg_quality": T(0), "n_beats": 0}
    if len(x) < fs * 10:
        return res
    _, sd = mean_sd(x, T)
    if sd == 0:
        return res
    pk = find_peaks_port(x, PEAK_DIST_BVP, T(PEAK_PROM_K_BVP) * sd, ops)
    res["n_beats"] = len(pk)
    if len(pk) < MIN_PEAKS:
        return res
    t = []
    half, lim = T(0.5), T(1e-12)
    for p in pk:
        if 0 < p < len(x) - 1:
            y0, y1, y2 = x[p - 1], x[p], x[p + 1]
            den = y0 - T(2) * y1 + y2
            off = half * (y0 - y2) / den if abs(den) > lim else T(0)
            off = max(-half, min(half, off))
            t.append((T(p) + off) / T(fs))
    ibi = [b - a for a, b in zip(t[:-1], t[1:])]
    ibi = [v for v in ibi if T(IBI_MIN_S) < v < T(IBI_MAX_S)]
    if len(ibi) < MIN_IBIS:
        return res
    med = median_sorted(ibi, T)
    tol = T(IBI_DEV) * med
    good = [v for v in ibi if abs(v - med) < tol]
    res["ppg_quality"] = T(len(good)) / T(len(ibi))
    if len(good) < MIN_IBIS:
        return res
    s = T(0)
    for v in good:
        s = s + T(60) / v
    res["hr_mean"] = s / T(len(good))
    return res


def slope_per_s(x: list, T=float, fs: int = FS_EDA):
    """Closed-form least-squares slope (per s) on t = i / fs.  C: the i-mean and sum (i - ibar)^2 are constants."""
    n = len(x)
    ibar = T((n - 1) / 2.0)
    sii = T(n * (n * n - 1) / 12.0)
    s = T(0)
    for i, v in enumerate(x):
        s = s + (T(i) - ibar) * v
    return s / sii * T(fs)


def autocorr_sqi(x: list, T=float, fs: int = FS_BVP, ops: dict = None) -> tuple:
    """Normalised autocorrelation of a band-passed PPG window over the beat-interval lags 0.33-1.5 s (the IBI rule).
    Returns (r_max, lag_samples) of the highest LOCAL maximum of r(k); (0, 0) if none.
    r(k) = sum x[i] x[i+k] / sqrt(sum_{i<N-k} x[i]^2 * sum_{i>=k} x[i]^2).  Used by the proposed gate (src/ppg_gate.py)."""
    n = len(x)
    k0, k1 = int(math.ceil(IBI_MIN_S * fs)), int(IBI_MAX_S * fs)
    # prefix sums of squares make the two energies O(1) per lag
    cs = [T(0)] * (n + 1)
    for i, v in enumerate(x):
        cs[i + 1] = cs[i] + v * v
    r = []
    for k in range(k0 - 1, k1 + 2):
        s = T(0)
        for i in range(n - k):
            s = s + x[i] * x[i + k]
        e = T(math.sqrt((cs[n - k] - cs[0]) * (cs[n] - cs[k])))
        r.append(s / e if e > 0 else T(0))
    if ops is not None:
        ops["autocorr_mac"] += sum(n - k for k in range(k0 - 1, k1 + 2))
    best, lag = T(0), 0
    for j in range(1, len(r) - 1):
        if r[j] >= r[j - 1] and r[j] >= r[j + 1] and r[j] > best:
            best, lag = r[j], k0 - 1 + j
    return best, lag


# ============================================================================= the streaming processor
class StreamD:
    """Model-D front end, one sample at a time.  cal_i0 / cal_i1 = EDA sample indices [i0, i1) of the 5-min rest
    calibration (on a device: set by the calibration routine; here: from the WESAD labels, as in training)."""

    def __init__(self, consts: dict, model: dict, cal_i0: int, cal_i1: int, T=float, ops: bool = False, sqi=False):
        self.T = T
        self.bvp_f = Biquads(consts["bvp_sos"], consts["bvp_zi"], T)
        self.eda_f = Biquads(consts["eda_sos"], consts["eda_zi"], T)
        self.ema = EMA(consts["ema_a"], T)
        nb, ne = WIN_S * FS_BVP, WIN_S * FS_EDA
        self.r_bvp, self.r_lp, self.r_ph, self.r_raw = Ring(nb, T), Ring(ne, T), Ring(ne, T), Ring(ne, T)
        self.cal_i0, self.cal_i1 = cal_i0, cal_i1
        self.w_lp, self.w_ph = Welford(T), Welford(T)
        self.s_lp = self.s_ph = None
        self.n_bvp = self.n_eda = 0
        self.w = [T(v) for v in model["weights_folded"]]
        self.b = T(model["bias_folded"])
        self.med = [T(v) for v in model["imputer_medians"]]
        self.zthr = T(model["logit_threshold_op1"])
        self.pthr = float(model["threshold_op1"])
        self.m, self.n = model["smoothing"]["m"], model["smoothing"]["n"]
        self.hist = [0] * self.n                    # last n window decisions (C: one uint8 bit field)
        self.ops = {k: 0 for k in ("scan", "local_max", "after_distance", "prominence_steps", "sort_n",
                                   "autocorr_mac", "windows")} if ops else None
        self.sqi = sqi

    def push_bvp(self, x):
        self.r_bvp.push(self.bvp_f.step(x))
        self.n_bvp += 1

    def push_eda(self, x):
        """Returns a window dict when a 30 s window ends at this EDA sample (every 5 s), else None."""
        lp = self.eda_f.step(x)
        ph = lp - self.ema.step(lp)
        self.r_lp.push(lp); self.r_ph.push(ph); self.r_raw.push(x)
        i = self.n_eda
        if self.cal_i0 <= i < self.cal_i1:
            self.w_lp.push(lp); self.w_ph.push(ph)
        self.n_eda += 1
        if self.n_eda == self.cal_i1:
            self.s_lp, self.s_ph = self.w_lp.sd(), self.w_ph.sd()
        ne, step = WIN_S * FS_EDA, STEP_S * FS_EDA
        if self.n_eda >= ne and (self.n_eda - ne) % step == 0 and self.n_bvp >= self.n_eda * FS_BVP // FS_EDA:
            return self.window()
        return None

    def window(self) -> dict:
        T = self.T
        t_end = self.n_eda / FS_EDA
        x = self.r_bvp.ordered()
        if self.n_bvp > self.n_eda * FS_BVP // FS_EDA:      # BVP ran ahead: never in run_stream (interleaved)
            raise RuntimeError("BVP and EDA out of step")
        h = hr_window(x, T, ops=self.ops)
        lp, ph, raw = self.r_lp.ordered(), self.r_ph.ordered(), self.r_raw.ordered()
        nan = T(float("nan"))
        if self.s_lp is not None:
            sl = slope_per_s(lp, T) / self.s_lp if self.s_lp > 0 else nan
            sc = T(len(find_peaks_port(ph, PEAK_DIST_EDA, T(SCR_REL_K) * self.s_ph, self.ops)))
        else:
            sl = sc = nan                               # D4: calibrating
        feats = [h["hr_mean"], sl, sc]
        xin = [m if v != v else v for v, m in zip(feats, self.med)]          # NaN -> imputer median
        z = self.b
        for wi, xi in zip(self.w, xin):
            z = z + wi * xi
        a_z = int(z >= self.zthr)
        p = 1.0 / (1.0 + math.exp(-float(z)))
        a_p = int(p >= self.pthr)
        if self.s_lp is None:
            a_z = a_p = 0                               # no alert before calibration exists
        self.hist = self.hist[1:] + [a_z]
        out = {"t_start_s": t_end - WIN_S, "hr_mean": float(h["hr_mean"]), "eda_slope_rel": float(sl),
               "scr_count_rel": float(sc), "ppg_quality": float(h["ppg_quality"]), "n_beats": h["n_beats"],
               "eda_range": float(max(raw) - min(raw)), "z": float(z), "p": p, "alert_z": a_z, "alert_p": a_p,
               "alert_smoothed_device": int(sum(self.hist) >= self.m)}
        if self.sqi:
            out["r_ac"], out["ac_lag"] = (float(v) for v in autocorr_sqi(x, T, ops=self.ops))
        if self.ops is not None:
            self.ops["windows"] += 1
        return out


def run_stream(bvp, eda, cal_i0: int, cal_i1: int, consts: dict, model: dict, T=float, ops=False, sqi=False):
    """Feed BVP (64 Hz) and EDA (4 Hz) interleaved as a device would receive them (16 BVP samples per EDA sample).
    Returns (windows DataFrame, calibration dict, op counts or None)."""
    s = StreamD(consts, model, cal_i0, cal_i1, T, ops, sqi)
    cast = T
    bvp = [cast(v) for v in np.asarray(bvp, float).tolist()]
    eda = [cast(v) for v in np.asarray(eda, float).tolist()]
    r = FS_BVP // FS_EDA
    nb = len(bvp)
    rows = []
    jb = 0
    for je, e in enumerate(eda):
        tgt = min(r * (je + 1), nb)
        while jb < tgt:
            s.push_bvp(bvp[jb]); jb += 1
        w = s.push_eda(e)
        if w is not None:
            rows.append(w)
    cal = {"sigma_lp_cal": float(s.s_lp) if s.s_lp is not None else np.nan,
           "sigma_ph_cal": float(s.s_ph) if s.s_ph is not None else np.nan}
    return pd.DataFrame(rows), cal, s.ops


# ============================================================================= parity run (all 15 WESAD subjects)
FEATS = ["hr_mean", "eda_slope_rel", "scr_count_rel"]


def _subject_job(args):
    name, pkl, dtype_name = args
    T = {"float64": float, "float32": np.float32}[dtype_name]
    raw = pp.load_subject(pkl)
    c0, c1 = pp.calibration_period_s(raw["label"])
    i0, i1 = int(round(c0 * FS_EDA)), int(round(c1 * FS_EDA))          # = hw_parity.calib_scales
    model = json.loads((MODELS / "stress_lr_D_w30.json").read_text(encoding="utf-8"))
    t = time.perf_counter()
    df, cal, ops = run_stream(raw["BVP"], raw["EDA"], i0, i1, design_constants(), model, T,
                              ops=(dtype_name == "float64"))
    sec = time.perf_counter() - t
    df.insert(0, "subject", name)
    return name, dtype_name, df, cal, ops, sec, len(raw["BVP"]) + len(raw["EDA"])


def offline_table() -> pd.DataFrame:
    f = pd.read_csv(METRICS / "features_w30.csv")
    e = pd.read_csv(METRICS / "features_eda_w30.csv", index_col="row")
    assert (e.t_start_s.values == f.t_start_s.values).all() and (e.subject.values == f.subject.values).all()
    f["eda_slope_rel"], f["scr_count_rel"] = e["eda_slope_rel__clean"].values, e["scr_count_rel__clean"].values
    return f


def alerts(df: pd.DataFrame, coef: dict, scored_only: bool = True) -> pd.DataFrame:
    from src.predict import proba_json
    from src.runner import smooth
    d = df[df.scored].copy() if scored_only else df.copy()
    d["p_model"] = proba_json(coef, d[FEATS])
    d["a_win"] = (d.p_model >= coef["threshold_op1"]).astype(int)
    sm = coef["smoothing"]
    d["a_s45"] = smooth(d.a_win.values, d.subject.values, d.t_start_s.values, sm["m"], sm["n"])
    return d


def compare(off: pd.DataFrame, st: pd.DataFrame, coef: dict, label: str) -> tuple:
    keys = ["subject", "t_start_s"]
    m = off.merge(st, on=keys, how="left", suffixes=("", "_st"), validate="one_to_one", indicator=True)
    miss = int((m[m.scored]._merge != "both").sum())
    assert miss == 0, f"{miss} scored offline windows have no stream window"
    sc = m[m.scored].copy()
    rows = []
    for f in FEATS + ["ppg_quality", "n_beats"]:
        a, b = sc[f].values.astype(float), sc[f + "_st"].values.astype(float)
        both = np.isfinite(a) & np.isfinite(b)
        d = np.abs(a - b)[both]
        rows.append({"arithmetic": label, "feature": f, "n_scored_windows": len(sc), "n_both_finite": int(both.sum()),
                     "n_nan_mismatch": int((np.isfinite(a) != np.isfinite(b)).sum()),
                     "max_abs_diff": float(d.max()) if len(d) else np.nan,
                     "median_abs_diff": float(np.median(d)) if len(d) else np.nan,
                     "n_windows_diff_gt_0": int((d > 0).sum()),
                     "max_rel_diff": float((d / np.maximum(np.abs(a[both]), 1e-300)).max()) if len(d) else np.nan})
    ao = alerts(m, coef)
    s_cols = m[keys + [f + "_st" for f in FEATS] + ["scored"]].rename(columns={f + "_st": f for f in FEATS})
    as_ = alerts(s_cols, coef)
    assert (ao.index == as_.index).all()
    flag_off = sc.hr_mean.isna() | (sc.ppg_quality < 0.5) | (sc.n_beats < 15)
    flag_st = sc.hr_mean_st.isna() | (sc.ppg_quality_st < 0.5) | (sc.n_beats_st < 15)
    al = {"arithmetic": label, "n_scored_windows": len(sc),
          "n_offline_windows_missing_in_stream": miss,
          "window_alert_differs": int((ao.a_win != as_.a_win).sum()),
          "smoothed_4of5_alert_differs": int((ao.a_s45 != as_.a_s45).sum()),
          "frac_window_alert_differs": float((ao.a_win != as_.a_win).mean()),
          "frac_smoothed_alert_differs": float((ao.a_s45 != as_.a_s45).mean()),
          "max_abs_p_diff": float(np.abs(ao.p_model - as_.p_model).max()),
          "flag_ppg_differs": int((flag_off != flag_st).sum()),
          "stream_alert_z_vs_p_disagree": int((sc.alert_z != sc.alert_p).sum()),
          "device_smoothing_vs_offline_smoothing_differs": int((sc.alert_smoothed_device.values != ao.a_s45.values).sum()),
          "stream_window_alert_vs_offline_differs": int((sc.alert_z.values != ao.a_win.values).sum())}
    return pd.DataFrame(rows), al, m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--subjects", nargs="*")
    a = ap.parse_args()
    subs = [s for s in pp.find_subjects(Path(a.data)) if not a.subjects or s[0] in a.subjects]
    coef = json.loads((MODELS / "stress_lr_D_w30.json").read_text(encoding="utf-8"))
    off = offline_table()
    off = off[off.subject.isin([s[0] for s in subs])]
    tasks = [(n, p, d) for d in ("float64", "float32") for n, p in subs]
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(_subject_job, tasks))
    print(f"stream runs done in {time.time() - t0:.0f} s", flush=True)
    feat_rows, alert_rows, cal_rows, ops_rows, per_subj = [], [], [], [], []
    for dt in ("float64", "float32"):
        st = pd.concat([r[2] for r in res if r[1] == dt], ignore_index=True)
        st.to_csv(METRICS / f"stream_windows_{dt}.csv", index=False)
        fr, al, m = compare(off, st, coef, dt)
        feat_rows.append(fr); alert_rows.append(al)
        for s, g in m[m.scored].groupby("subject"):
            per_subj.append({"arithmetic": dt, "subject": s, "n_scored": len(g),
                             **{f"max_abs_diff_{f}": float(np.nanmax(np.abs(g[f] - g[f + "_st"]))) for f in FEATS}})
        for r in res:
            if r[1] == dt:
                cal_rows.append({"arithmetic": dt, "subject": r[0], **{k + "_stream": v for k, v in r[3].items()},
                                 "stream_seconds_pc": r[5], "n_samples": r[6],
                                 "us_per_sample_pc": 1e6 * r[5] / r[6]})
                if r[4]:
                    ops_rows.append({"subject": r[0], **r[4]})
    cal = pd.DataFrame(cal_rows)
    ref = pd.read_csv(METRICS / "stepc_eda_gain_offset.csv")[["subject", "sigma_lp_cal_clean", "sigma_ph_cal_clean"]]
    cal = cal.merge(ref, on="subject")
    cal["sigma_lp_abs_diff"] = (cal.sigma_lp_cal_stream - cal.sigma_lp_cal_clean).abs()
    cal["sigma_ph_abs_diff"] = (cal.sigma_ph_cal_stream - cal.sigma_ph_cal_clean).abs()
    pd.concat(feat_rows).to_csv(METRICS / "stream_parity_features.csv", index=False)
    pd.DataFrame(alert_rows).to_csv(METRICS / "stream_parity_alerts.csv", index=False)
    pd.DataFrame(per_subj).to_csv(METRICS / "stream_parity_per_subject.csv", index=False)
    cal.to_csv(METRICS / "stream_parity_calibration.csv", index=False)
    pd.DataFrame(ops_rows).to_csv(METRICS / "stream_op_counts.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(pd.concat(feat_rows).to_string(index=False))
        print(pd.DataFrame(alert_rows).T.to_string())
        print(cal[["arithmetic", "subject", "sigma_lp_abs_diff", "sigma_ph_abs_diff", "us_per_sample_pc"]]
              .groupby("arithmetic").max(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
