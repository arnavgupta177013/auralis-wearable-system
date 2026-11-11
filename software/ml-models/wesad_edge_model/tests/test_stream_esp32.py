"""
Tests for the streaming reference (src/stream_ref.py), the C export (src/export_esp32.py, models/stress_lr_D_w30.h)
and the proposed PPG gate (src/ppg_gate.py).  Hand-made signals are SYNTHETIC (generated here), not recordings.
WESAD tests are SKIPPED if the read-only dataset is absent; the C harness test is SKIPPED if no host C compiler exists.

    python -m unittest tests.test_stream_esp32 -v
"""

import json
import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps

from src import export_esp32 as X
from src import features as ft
from src import hw_parity as hwp
from src import ppg_gate as G
from src import preprocessing as pp
from src import stream_ref as S
from tests import synthetic_signals as syn

WESAD = Path(os.environ.get("WESAD_ROOT", r"C:\datasets\WESAD"))
HAVE_WESAD = (WESAD / "S5" / "S5.pkl").exists()
MODEL = json.loads((S.MODELS / "stress_lr_D_w30.json").read_text(encoding="utf-8"))
C = S.design_constants()


class BuildingBlocks(unittest.TestCase):
    def test_biquads_equal_scipy_sosfilt_steady_state_start(self):
        x = np.random.default_rng(0).normal(0, 50, 5000) + 300.0
        ref = pp.CausalSOS(pp.bvp_bandpass_sos()).process(x)
        f = S.Biquads(C["bvp_sos"], C["bvp_zi"])
        y = np.array([f.step(v) for v in x.tolist()])
        self.assertLessEqual(np.max(np.abs(y - ref)), 1e-12)

    def test_ema_equals_causal_ema(self):
        x = np.random.default_rng(1).normal(2, 0.1, 3000)
        ref = pp.CausalEMA(pp.EDA_TONIC_TAU_S, 4).process(x)
        e = S.EMA(C["ema_a"])
        self.assertLessEqual(np.max(np.abs(np.array([e.step(v) for v in x.tolist()]) - ref)), 1e-14)

    def test_find_peaks_port_equals_scipy(self):
        """Float signals: identical peaks.  Quantised signals (many equal-height maxima): identical when the port uses
        scipy's argsort tie order, which proves the stable-order port differs ONLY in tie order (deviation D1)."""
        rng = np.random.default_rng(2)
        n_cases, n_tie_diff = 0, 0
        for i in range(400):
            n = int(rng.integers(20, 2000))
            kind = i % 4
            if kind == 0:
                x = rng.normal(0, 1, n)
            elif kind == 1:
                x = sps.sosfilt(pp.bvp_bandpass_sos(), rng.normal(0, 1, n))
            elif kind == 2:                                   # quantised: plateaus of equal samples
                x = np.round(rng.normal(0, 3, n))
            else:
                x = np.sin(np.arange(n) / rng.uniform(2, 20)) + rng.normal(0, 0.1, n)
            dist = int(rng.integers(1, 30))
            prom = float(rng.uniform(0, 1.5) * np.std(x))
            ref, _ = sps.find_peaks(x, distance=dist, prominence=prom)
            got = S.find_peaks_port(x.tolist(), dist, prom)
            if kind == 2:
                self.assertEqual(list(ref), S.find_peaks_port(x.tolist(), dist, prom, tie_order="numpy"), f"case {i}")
                n_tie_diff += list(ref) != got
            else:
                self.assertEqual(list(ref), got, f"case {i} kind {kind}")
            n_cases += 1
        self.assertEqual(n_cases, 400)
        print(f"  [quantised cases where the stable tie order differs from scipy: {n_tie_diff}/100]")

    def test_find_peaks_plateau_midpoint(self):
        x = [0, 1, 3, 3, 3, 3, 1, 0, 2, 2, 0]
        self.assertEqual(S.find_peaks_port(x, 1, 0.0), list(sps.find_peaks(x, distance=1, prominence=0.0)[0]))

    def test_hr_window_equals_features_ppg(self):
        for seed, hr in ((0, 62), (1, 95), (2, 140)):
            b = pp.CausalSOS(pp.bvp_bandpass_sos()).process(syn.bvp(60, hr, seed=seed))[-1920:]
            ref = ft.ppg_features(b)
            got = S.hr_window(b.tolist())
            self.assertAlmostEqual(got["hr_mean"], ref["hr_mean"], delta=1e-9)
            self.assertEqual(got["n_beats"], ref["n_beats"])
            self.assertEqual(got["ppg_quality"], ref["ppg_quality"])

    def test_hr_window_noise_and_flat(self):
        self.assertTrue(np.isnan(S.hr_window([0.0] * 1920)["hr_mean"]))
        x = np.random.default_rng(3).normal(0, 1, 1920)
        a, b = S.hr_window(x.tolist()), ft.ppg_features(x)
        self.assertEqual(np.isnan(a["hr_mean"]), np.isnan(b["hr_mean"]))

    def test_slope_equals_polyfit(self):
        x = np.random.default_rng(4).normal(0, 1, 120).cumsum()
        self.assertAlmostEqual(S.slope_per_s(x.tolist()), np.polyfit(np.arange(120) / 4, x, 1)[0], delta=1e-12)

    def test_welford_equals_numpy_std(self):
        x = np.random.default_rng(5).normal(3, 0.2, 1200)
        w = S.Welford()
        for v in x.tolist():
            w.push(v)
        self.assertAlmostEqual(w.sd(), float(np.std(x)), delta=1e-13)

    def test_autocorr_port_equals_gate_numpy(self):
        for seed in range(5):
            b = pp.CausalSOS(pp.bvp_bandpass_sos()).process(np.random.default_rng(seed).normal(0, 1, 3000))[-1920:]
            r1, l1 = S.autocorr_sqi(b.tolist())
            r2, l2 = G.autocorr_sqi(b)
            self.assertEqual(l1, l2)
            self.assertAlmostEqual(r1, r2, delta=1e-12)


class StreamVsOffline(unittest.TestCase):
    """The streaming processor on a SYNTHETIC continuous recording equals the offline pipeline window by window."""

    def test_synthetic_stream_equals_offline(self):
        dur = 600
        bvp, eda = syn.bvp(dur, 72, seed=7), syn.eda(dur, scr_every_s=25, scr_amp=0.08, seed=8)
        i0, i1 = 60 * 4, 360 * 4
        st, cal, _ = S.run_stream(bvp, eda, i0, i1, C, MODEL)
        filt = pp.CausalSOS(pp.bvp_bandpass_sos()).process(bvp)
        lp, tonic, ph = hwp.eda_streams(eda)
        s_lp, s_ph = float(np.std(lp[i0:i1])), float(np.std(ph[i0:i1]))
        self.assertAlmostEqual(cal["sigma_lp_cal"], s_lp, delta=1e-12)
        self.assertAlmostEqual(cal["sigma_ph_cal"], s_ph, delta=1e-12)
        after = st[st.t_start_s >= 360]
        self.assertGreater(len(after), 40)
        for _, r in after.iterrows():
            t0 = r.t_start_s
            ref_hr = ft.ppg_features(filt[int(t0 * 64):int(t0 * 64) + 1920])["hr_mean"]
            e = hwp.eda_window(lp, tonic, ph, eda, t0, t0 + 30, s_lp, s_ph)
            self.assertAlmostEqual(r.hr_mean, ref_hr, delta=1e-9)
            self.assertAlmostEqual(r.eda_slope_rel, e["eda_slope_rel"], delta=1e-9)
            self.assertEqual(r.scr_count_rel, e["scr_count_rel"])
        self.assertTrue(st[st.t_start_s + 30 < 360].eda_slope_rel.isna().all(), "D4: NaN before calibration ends")
        self.assertFalse(np.isnan(st[st.t_start_s + 30 == 360].eda_slope_rel.iloc[0]), "window ending at cal end")

    def test_float32_stream_runs_and_is_close(self):
        bvp, eda = syn.bvp(420, 80, seed=9), syn.eda(420, seed=10)
        a, _, _ = S.run_stream(bvp, eda, 0, 1200, C, MODEL, T=float)
        b, _, _ = S.run_stream(bvp, eda, 0, 1200, C, MODEL, T=np.float32)
        ok = a.t_start_s >= 300
        self.assertLess(np.max(np.abs(a.hr_mean[ok] - b.hr_mean[ok])), 0.01)
        self.assertTrue((a.alert_z[ok].values == b.alert_z[ok].values).all())

    @unittest.skipUnless(HAVE_WESAD, "WESAD not available")
    def test_wesad_s5_stream_parity(self):
        name, _, df, _, _, _, _ = S._subject_job(("S5", WESAD / "S5" / "S5.pkl", "float64"))
        off = S.offline_table()
        fr, al, _ = S.compare(off[off.subject == "S5"], df, MODEL, "float64")
        for f in S.FEATS:
            self.assertLessEqual(fr.set_index("feature").loc[f, "max_abs_diff"], 1e-9)
        self.assertEqual(al["window_alert_differs"], 0)
        self.assertEqual(al["smoothed_4of5_alert_differs"], 0)


class StoredParityResults(unittest.TestCase):
    """The committed parity outputs (python -m src.stream_ref) say what NOTES.md says."""

    def test_no_alert_changes_float64_float32(self):
        a = pd.read_csv(S.METRICS / "stream_parity_alerts.csv").set_index("arithmetic")
        for dt in ("float64", "float32"):
            self.assertEqual(a.loc[dt, "n_scored_windows"], 5509)
            self.assertEqual(a.loc[dt, "window_alert_differs"], 0)
            self.assertEqual(a.loc[dt, "smoothed_4of5_alert_differs"], 0)


class Header(unittest.TestCase):
    def test_committed_header_is_regenerated_from_json(self):
        self.assertEqual(X.HEADER_PATH.read_text(encoding="utf-8"), X.header_text())

    def test_header_constants_equal_json(self):
        h = X.parse_header_constants(X.HEADER_PATH.read_text(encoding="utf-8"))
        self.assertEqual(h["WEIGHTS"], MODEL["weights_folded"])
        self.assertEqual(h["IMPUTER_MEDIANS"], MODEL["imputer_medians"])
        self.assertEqual(h["BIAS"], MODEL["bias_folded"])
        self.assertEqual(h["THRESHOLD_OP1"], MODEL["threshold_op1"])
        self.assertEqual(h["LOGIT_THRESHOLD_OP1"], MODEL["logit_threshold_op1"])
        self.assertEqual((h["WINDOW_S"], h["STEP_S"]), (MODEL["window_s"], MODEL["step_s"]))
        self.assertEqual((h["SMOOTH_M"], h["SMOOTH_N"]), (MODEL["smoothing"]["m"], MODEL["smoothing"]["n"]))
        self.assertEqual((h["FS_BVP_HZ"], h["FS_EDA_HZ"]), (MODEL["input_rates_hz"]["BVP"], MODEL["input_rates_hz"]["EDA"]))
        self.assertEqual(h["FEATURE_ORDER"], MODEL["feature_order"])
        spec = json.loads((S.MODELS / "feature_spec.json").read_text(encoding="utf-8"))
        self.assertEqual(h["FEATURE_ORDER"], [f for f in spec["order"] if f in h["FEATURE_ORDER"]])

    @unittest.skipUnless(X.find_host_compiler(), "no host C compiler: the C code is NOT compiled or run")
    def test_c_harness_agrees_with_python(self):
        r = X.c_test(verbose=False)
        for real, tol in (("float", 1e-6), ("double", 1e-12)):
            self.assertEqual(r[f"{real}_real300"]["n_rows"], 300)
            self.assertEqual(r[f"{real}_real300"]["alert_differs_vs_alert_op1_ref"], 0)
            self.assertLessEqual(r[f"{real}_real300"]["max_abs_p_diff_vs_p_ref"], tol)
            self.assertEqual(r[f"{real}_synthetic_nan40"]["alert_differs_vs_alert_op1_ref"], 0)
            self.assertEqual(r[f"{real}_smoother"]["differs_vs_runner_smooth"], 0)


class Gate(unittest.TestCase):
    TH = json.loads((S.METRICS / "ppg_gate_thresholds.json").read_text())

    def _flag(self, kind, seed):
        filt, raw = G.synth(kind, seed)
        d = pd.DataFrame([G.window_stats(filt, raw)])
        return bool(G.gate(d, self.TH["T_AC"], self.TH["HR_AGREE"]).flagged.iloc[0])

    def test_flat_railed_clipped_flagged(self):
        for k in ("flat_constant", "railed_full_scale", "clipped_pulse"):
            self.assertTrue(self._flag(k, 12345), k)

    def test_clean_pulse_passes(self):
        self.assertFalse(self._flag("clean_pulse", 12345))

    def test_thresholds_reproduce_from_stored_real_windows(self):
        t = pd.read_csv(S.METRICS / "ppg_gate_real_windows.csv", index_col=0)
        th = G.freeze_thresholds(t)
        self.assertAlmostEqual(th["T_AC"], self.TH["T_AC"], delta=1e-12)
        self.assertAlmostEqual(th["HR_AGREE"], self.TH["HR_AGREE"], delta=1e-12)
        self.assertFalse(set(G.TUNE) & set(G.EVAL))
        self.assertEqual(len(set(G.TUNE) | set(G.EVAL)), 15)


if __name__ == "__main__":
    unittest.main()
