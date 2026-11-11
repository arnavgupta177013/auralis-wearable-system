"""
Proposed PPG signal-quality gate for the hr_mean input of model D (stress detection; WESAD, not panic/anxiety).
GATE ONLY: the model, the features and src/predict.py are NOT changed.  This module evaluates a proposal.

Problem (Step E): the training PPG quality rule (features.quality_flags: >= 5 beats / 10 s, >= 50 % regular IBIs) is
scale-invariant, so band-passed noise passes it and gives a fake HR.

Proposed gate, per 30 s window of the causally band-passed BVP (all streaming-friendly; see stream_ref.autocorr_sqi):
  G0  existing flag_ppg (unchanged)
  G1  flat / railed raw window: peak-to-peak == 0, or > 2 % of raw samples repeat the window max or min
  G2  periodicity: r_ac = highest local maximum of the normalised autocorrelation over lags 0.33-1.5 s (the IBI rule's
      range) must be >= T_AC
  G3  agreement: the autocorrelation HR (60 / lag) must agree with hr_mean within HR_AGREE (relative), allowing the
      autocorrelation to lock on twice the beat period (dicrotic / respiratory modulation): min over {lag, lag/2}
Window passes if none of G0-G3 fires.

Evaluation design (fixed BEFORE any evaluation number was computed; see NOTES.md Step F):
  * Real windows: all 6421 labelled 30 s WESAD windows.  Reference = chest ECG HR from Step A
    (hrv_check_chest_windows_w30.csv): "good" = chest_valid and |wrist hr_mean - chest HR| <= 5 bpm;
    "bad" = wrist HR NaN or |difference| > 15 bpm; the rest is "ambiguous".
  * Subjects split in two halves (alternating in numeric order): TUNE = S2 S4 S6 S8 S10 S13 S15 S17,
    EVAL = S3 S5 S7 S9 S11 S14 S16.  Thresholds come from TUNE good windows only:
        T_AC     = 1st percentile of r_ac over TUNE good windows that pass G0 and G1
        HR_AGREE = 99th percentile of the G3 relative disagreement over the same windows
    No noise window is used to set a threshold.
  * Synthetic noise (SYNTHETIC, generated here, not recordings): TUNING seeds 0-99 of white Gaussian noise are only
    inspected (sanity check); TEST sets use seeds >= 1000 and include noise types never inspected before freezing.
  * Reported: flagged fraction of real windows (TUNE and EVAL separately, by label), false rejections of good windows,
    catches of bad windows, catch rate per synthetic noise type.

Run (from software/ml-models/wesad_edge_model, data read-only):
    python -m src.ppg_gate --data "C:\\datasets\\WESAD"
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import features as ft
from src import preprocessing as pp
from src.stream_ref import FS_BVP, IBI_MAX_S, IBI_MIN_S

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
WIN = 30
TUNE = ["S2", "S4", "S6", "S8", "S10", "S13", "S15", "S17"]
EVAL = ["S3", "S5", "S7", "S9", "S11", "S14", "S16"]
GOOD_BPM, BAD_BPM = 5.0, 15.0
RAIL_FRAC = 0.02                                   # = predict.RAIL_FRAC_WINDOW
K0, K1 = int(np.ceil(IBI_MIN_S * FS_BVP)), int(IBI_MAX_S * FS_BVP)


# ============================================================================= statistics (numpy == stream_ref port)
def autocorr_sqi(x: np.ndarray, fs: int = FS_BVP) -> tuple:
    """Vectorised twin of stream_ref.autocorr_sqi (same definition; tested equal)."""
    x = np.asarray(x, float)
    n = len(x)
    cs = np.r_[0.0, np.cumsum(x * x)]
    ks = np.arange(K0 - 1, K1 + 2)
    r = np.empty(len(ks))
    for j, k in enumerate(ks):
        e = np.sqrt((cs[n - k] - cs[0]) * (cs[n] - cs[k]))
        r[j] = np.dot(x[: n - k], x[k:]) / e if e > 0 else 0.0
    loc = np.flatnonzero((r[1:-1] >= r[:-2]) & (r[1:-1] >= r[2:])) + 1
    if not len(loc):
        return 0.0, 0
    j = loc[np.argmax(r[loc])]
    if r[j] <= 0:
        return 0.0, 0
    return float(r[j]), int(ks[j])


def railed_frac(raw: np.ndarray) -> float:
    if not len(raw):
        return 1.0
    return float(max(np.sum(raw == raw.max()), np.sum(raw == raw.min())) - 1) / len(raw)


def hr_disagreement(hr_mean: float, lag: int, fs: int = FS_BVP) -> float:
    if not np.isfinite(hr_mean) or lag <= 0:
        return np.inf
    hr_ac = 60.0 * fs / lag
    return float(min(abs(hr_mean - hr_ac), abs(hr_mean - 2 * hr_ac)) / hr_mean)


def window_stats(filt: np.ndarray, raw: np.ndarray) -> dict:
    h = ft.ppg_features(filt)
    r, lag = autocorr_sqi(filt)
    flag_ppg = bool(np.isnan(h["hr_mean"]) or h["ppg_quality"] < ft.PPG_QUALITY_MIN
                    or h["n_beats"] < ft.MIN_BEATS_PER_10S * WIN / 10)
    return {"hr_mean": h["hr_mean"], "ppg_quality": h["ppg_quality"], "n_beats": h["n_beats"], "flag_ppg": flag_ppg,
            "raw_ptp": float(np.ptp(raw)) if len(raw) else 0.0, "railed_frac": railed_frac(raw),
            "r_ac": r, "ac_lag": lag, "hr_disagree": hr_disagreement(h["hr_mean"], lag)}


def gate(df: pd.DataFrame, t_ac: float, hr_agree: float) -> pd.DataFrame:
    g = pd.DataFrame(index=df.index)
    g["G0_flag_ppg"] = df.flag_ppg.astype(bool)
    g["G1_flat_railed"] = (df.raw_ptp <= 0) | (df.railed_frac > RAIL_FRAC)
    g["G2_low_periodicity"] = df.r_ac < t_ac
    g["G3_hr_disagree"] = ~(df.hr_disagree <= hr_agree)
    g["new_only"] = (g.G1_flat_railed | g.G2_low_periodicity | g.G3_hr_disagree) & ~g.G0_flag_ppg
    g["flagged"] = g[["G0_flag_ppg", "G1_flat_railed", "G2_low_periodicity", "G3_hr_disagree"]].any(axis=1)
    return g


# ============================================================================= real WESAD windows
def real_table(data_root: Path) -> pd.DataFrame:
    f = pd.read_csv(METRICS / "features_w30.csv")
    ch = pd.read_csv(METRICS / "hrv_check_chest_windows_w30.csv")
    assert (ch.subject.values == f.subject.values).all() and (ch.t_start_s.values == f.t_start_s.values).all()
    rows = []
    for name, pkl in pp.find_subjects(data_root):
        raw = pp.load_subject(pkl)
        filt = pp.CausalSOS(pp.bvp_bandpass_sos()).process(raw["BVP"])
        g = f[f.subject == name]
        for i, t0 in zip(g.index, g.t_start_s):
            a, b = int(round(t0 * FS_BVP)), int(round((t0 + WIN) * FS_BVP))
            s = window_stats(filt[a:b], raw["BVP"][a:b])
            assert (np.isnan(s["hr_mean"]) and np.isnan(f.hr_mean[i])) or abs(s["hr_mean"] - f.hr_mean[i]) < 1e-9
            rows.append({"row": i, **s})
    t = pd.DataFrame(rows).set_index("row").sort_index()
    t.insert(0, "subject", f.subject); t.insert(1, "t_start_s", f.t_start_s); t.insert(2, "label", f.label)
    t.insert(3, "scored", f.scored)
    t["chest_hr"] = np.where(ch.chest_valid, ch.chest_hr_mean, np.nan)
    d = (t.hr_mean - t.chest_hr).abs()
    t["ref"] = np.where(t.chest_hr.isna(), "no_ecg_reference",
                        np.where(d <= GOOD_BPM, "good", np.where(t.hr_mean.isna() | (d > BAD_BPM), "bad", "ambiguous")))
    t["half"] = np.where(t.subject.isin(TUNE), "TUNE", "EVAL")
    return t


def freeze_thresholds(t: pd.DataFrame) -> dict:
    g = t[(t.half == "TUNE") & (t.ref == "good") & ~t.flag_ppg & (t.raw_ptp > 0) & (t.railed_frac <= RAIL_FRAC)]
    return {"T_AC": float(np.percentile(g.r_ac, 1)), "HR_AGREE": float(np.percentile(g.hr_disagree, 99)),
            "n_tune_good_windows": int(len(g)), "rule": "T_AC = 1st percentile of r_ac, HR_AGREE = 99th percentile of "
            "hr_disagree, over TUNE good windows passing G0 and G1"}


# ============================================================================= SYNTHETIC noise / fault windows
def _filtered(raw: np.ndarray, warm_s: float = 60.0) -> tuple:
    """Prepend warm-up (same generator continued), filter causally as the device would, keep the last 30 s."""
    y = pp.CausalSOS(pp.bvp_bandpass_sos()).process(raw)
    n = WIN * FS_BVP
    return y[-n:], raw[-n:]


def _pulse(t, hr, amp=50.0):
    ph = 2 * np.pi * hr / 60.0 * t
    return amp * np.sin(ph) + 0.4 * amp * np.sin(2 * ph + 0.5)


def synth(kind: str, seed: int) -> tuple:
    """One SYNTHETIC 90 s raw BVP-like segment (60 s warm-up + 30 s window). Returns (filtered window, raw window)."""
    rng = np.random.default_rng(seed)
    n = 90 * FS_BVP
    t = np.arange(n) / FS_BVP
    if kind == "white_gauss":
        x = rng.normal(0, 50, n)
    elif kind == "white_uniform":
        x = rng.uniform(-80, 80, n)
    elif kind == "pink":                                   # 1/f power spectrum
        X = np.fft.rfft(rng.normal(0, 1, n)); f = np.fft.rfftfreq(n, 1 / FS_BVP); X[1:] /= np.sqrt(f[1:]); X[0] = 0
        x = 50 * np.fft.irfft(X, n) / np.std(np.fft.irfft(X, n))
    elif kind == "brown":                                  # random walk (sensor drift / motion wander)
        x = np.cumsum(rng.normal(0, 5, n))
    elif kind == "spikes":                                 # sparse impulsive artefacts on low noise
        x = rng.normal(0, 2, n); idx = rng.choice(n, n // 50, replace=False); x[idx] += rng.normal(0, 300, len(idx))
    elif kind == "random_interval_pulses":                 # pulse shapes at random intervals 0.35-1.45 s
        x = np.zeros(n); tk = 0.0
        while tk < 90:
            x += 60 * np.exp(-0.5 * ((t - tk) / 0.08) ** 2); tk += rng.uniform(0.35, 1.45)
        x += rng.normal(0, 2, n)
    elif kind == "flat_constant":
        x = np.full(n, 1000.0 + seed)
    elif kind == "railed_full_scale":                      # ADC stuck at full scale (MAX30102 18-bit max)
        x = np.full(n, 262143.0)
    elif kind == "clipped_pulse":                          # real-looking pulse clipped at 40 % of its range
        x = _pulse(t, rng.uniform(55, 100)) + rng.normal(0, 1, n); x = np.minimum(x, np.percentile(x, 60))
    elif kind == "periodic_motion_1p6Hz":                  # periodic artefact IN the HR band (no cardiac content)
        x = 80 * np.sin(2 * np.pi * 1.6 * t + rng.uniform(0, 6.3)) + rng.normal(0, 5, n)
    elif kind == "clean_pulse":                            # control: should PASS
        x = _pulse(t, rng.uniform(50, 120)) + rng.normal(0, 3, n)
    elif kind == "clean_pulse_hrv":                        # control: pulse with 5 % beat-to-beat IBI variability
        x = np.zeros(n); tk = 0.0; ibi0 = 60 / rng.uniform(55, 110)
        while tk < 90:
            x += 60 * np.exp(-0.5 * ((t - tk) / 0.1) ** 2); tk += ibi0 * (1 + rng.normal(0, 0.05))
        x += rng.normal(0, 2, n)
    else:
        raise ValueError(kind)
    return _filtered(x)


TUNING_NOISE = [("white_gauss", range(0, 100))]
TEST_SETS = [  # (kind, seeds, expected) ; expected = "catch" (should be flagged) or "pass" (control)
    ("white_gauss", range(1000, 1100), "catch"), ("white_uniform", range(2000, 2100), "catch"),
    ("pink", range(3000, 3100), "catch"), ("brown", range(4000, 4100), "catch"),
    ("spikes", range(5000, 5100), "catch"), ("random_interval_pulses", range(6000, 6100), "catch"),
    ("flat_constant", range(7000, 7010), "catch"), ("railed_full_scale", range(7100, 7110), "catch"),
    ("clipped_pulse", range(7200, 7300), "catch"),
    ("periodic_motion_1p6Hz", range(8000, 8100), "catch (known: not separable from a pulse without ACC)"),
    ("clean_pulse", range(9000, 9100), "pass"), ("clean_pulse_hrv", range(9100, 9200), "pass")]


def synth_table(sets) -> pd.DataFrame:
    rows = []
    for kind, seeds, *exp in sets:
        for s in seeds:
            filt, raw = synth(kind, s)
            rows.append({"kind": kind, "seed": s, "expected": exp[0] if exp else "inspect", **window_stats(filt, raw)})
    return pd.DataFrame(rows)


# ============================================================================= main
def summarise_real(t: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for half in ("TUNE", "EVAL", "ALL"):
        m = (t.half == half) if half != "ALL" else np.ones(len(t), bool)
        for sel, mm in (("all windows", m), ("scored", m & t.scored), ("stress", m & (t.label == 2)),
                        ("baseline", m & (t.label == 1)), ("amusement", m & (t.label == 3)),
                        ("ref good (ECG agrees <= 5 bpm)", m & (t.ref == "good")),
                        ("ref bad (> 15 bpm or NaN)", m & (t.ref == "bad")), ("ref ambiguous", m & (t.ref == "ambiguous"))):
            mm = np.asarray(mm)
            rows.append({"half": half, "subset": sel, "n": int(mm.sum()),
                         "flagged_existing_G0": int(g.G0_flag_ppg[mm].sum()),
                         "flagged_proposed_any": int(g.flagged[mm].sum()),
                         "flagged_new_only": int(g.new_only[mm].sum()),
                         "frac_flagged_proposed": float(g.flagged[mm].mean()) if mm.sum() else np.nan,
                         "frac_flagged_existing": float(g.G0_flag_ppg[mm].mean()) if mm.sum() else np.nan,
                         "G1": int(g.G1_flat_railed[mm].sum()), "G2": int(g.G2_low_periodicity[mm].sum()),
                         "G3": int(g.G3_hr_disagree[mm].sum())})
    return pd.DataFrame(rows)


def tradeoff(t: pd.DataFrame, syn: pd.DataFrame, th: dict) -> pd.DataFrame:
    """POST-HOC, descriptive only: false rejection of EVAL good windows vs catch rate of the TEST noise types, over a
    range of T_AC (HR_AGREE kept frozen).  It looks at the evaluation data, so it must NOT be used to pick T_AC."""
    noise = syn[syn.expected == "catch"].kind.unique()
    noise = [k for k in noise if k not in ("periodic_motion_1p6Hz",)]
    rows = []
    for tac in (th["T_AC"], 0.15, 0.2, 0.25, 0.3, 0.35, 0.4):
        gr, gs = gate(t, tac, th["HR_AGREE"]), gate(syn, tac, th["HR_AGREE"])
        ev = t.half == "EVAL"
        r = {"T_AC": round(tac, 4), "frozen": tac == th["T_AC"],
             "EVAL_good_false_reject": float(gr.flagged[ev & (t.ref == "good")].mean()),
             "EVAL_bad_caught": float(gr.flagged[ev & (t.ref == "bad")].mean()),
             "EVAL_scored_flagged": float(gr.flagged[ev & t.scored].mean()),
             "EVAL_stress_flagged": float(gr.flagged[ev & (t.label == 2)].mean())}
        for k in noise:
            r[f"caught_{k}"] = float(gs.flagged[syn.kind == k].mean())
        r["clean_controls_flagged"] = float(gs.flagged[syn.expected == "pass"].mean())
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    t = real_table(Path(a.data))
    th = freeze_thresholds(t)
    print("FROZEN thresholds (TUNE good windows only):", json.dumps(th))
    tun = synth_table(TUNING_NOISE)
    gt = gate(tun, th["T_AC"], th["HR_AGREE"])
    print(f"sanity (tuning noise, inspected only): white_gauss seeds 0-99 flagged {int(gt.flagged.sum())}/100 "
          f"(existing rule {int(gt.G0_flag_ppg.sum())}/100)")
    g = gate(t, th["T_AC"], th["HR_AGREE"])
    real = summarise_real(t, g)
    syn = synth_table(TEST_SETS)
    gs = gate(syn, th["T_AC"], th["HR_AGREE"])
    srows = []
    for (kind, exp), idx in syn.groupby(["kind", "expected"], sort=False).groups.items():
        srows.append({"kind": kind, "expected": exp, "n": len(idx), "flagged_existing_G0": int(gs.G0_flag_ppg[idx].sum()),
                      "flagged_proposed": int(gs.flagged[idx].sum()), "G1": int(gs.G1_flat_railed[idx].sum()),
                      "G2": int(gs.G2_low_periodicity[idx].sum()), "G3": int(gs.G3_hr_disagree[idx].sum()),
                      "median_r_ac": float(syn.r_ac[idx].median()), "median_fake_hr": float(syn.hr_mean[idx].median())})
    synd = pd.DataFrame(srows)
    pd.concat([t, g], axis=1).to_csv(METRICS / "ppg_gate_real_windows.csv")
    real.to_csv(METRICS / "ppg_gate_real_summary.csv", index=False)
    pd.concat([syn, gs], axis=1).to_csv(METRICS / "ppg_gate_synthetic_windows.csv", index=False)
    synd.to_csv(METRICS / "ppg_gate_synthetic_summary.csv", index=False)
    # per-subject flag rate (scored) and r_ac distribution by reference class
    per = pd.concat([t, g], axis=1).groupby("subject").agg(n=("flagged", "size"), flagged=("flagged", "sum"),
                                                            existing=("G0_flag_ppg", "sum")).reset_index()
    per.to_csv(METRICS / "ppg_gate_per_subject.csv", index=False)
    dist = t.groupby(["half", "ref"]).r_ac.describe(percentiles=[0.01, 0.05, 0.5]).reset_index()
    dist.to_csv(METRICS / "ppg_gate_r_ac_distribution.csv", index=False)
    (METRICS / "ppg_gate_thresholds.json").write_text(json.dumps(th, indent=2))
    trade = tradeoff(t, syn, th)
    trade.to_csv(METRICS / "ppg_gate_tradeoff_POSTHOC.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(real.to_string(index=False)); print(synd.to_string(index=False)); print(dist.to_string(index=False))
        print(per.to_string(index=False))
        print("POST-HOC descriptive sweep (looks at EVAL and TEST noise; NOT used to choose anything):")
        print(trade.to_string(index=False))


if __name__ == "__main__":
    main()
