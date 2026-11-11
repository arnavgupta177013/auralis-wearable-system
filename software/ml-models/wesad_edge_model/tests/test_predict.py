"""
Tests for the inference CLI (src/predict.py, Step E). Needs models/ (src.final_model) and, for the tests marked
WESAD, the read-only dataset at C:\\datasets\\WESAD (or env WESAD_ROOT); those are SKIPPED if it is absent.

Run from software/ml-models/wesad_edge_model:
    python -m unittest tests.test_predict -v

What "matches the stored outputs" means here (stated, not hidden):
  * The deployed model D was fitted on all 15 subjects and has no held-out predictions of its own. The LOSO held-out
    scores in results/metrics/final_predictions.csv come from 15 fold models that were not saved, so model D cannot
    reproduce them and is not compared with them.
  * Compared instead: (1) models/roundtrip_windows.csv `p_ref` = in-memory model D scores written by
    src.final_model at fit time (300 real WESAD windows, 17 significant digits): predict.py --features-csv must give
    them EXACTLY (tolerance 0); (2) the raw-signal path (--demo-wesad and --signals) must reproduce the training
    features in results/metrics/features_w30.csv + features_eda_w30.csv ("__clean") for every scored S5 window
    (rtol 1e-9, atol 1e-12, the hw_parity tolerance) and the S5 rows of roundtrip_windows.csv `p_ref` (atol 1e-12).
All hand-made inputs below are SYNTHETIC (tests/synthetic_signals.py or literal feature rows), not recordings.
"""

import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import predict as P
from tests import synthetic_signals as syn

ROOT = P.ROOT
METRICS = ROOT / "results" / "metrics"
WESAD = Path(os.environ.get("WESAD_ROOT", str(P.DEFAULT_WESAD)))
HAVE_WESAD = (WESAD / "S5" / "S5.pkl").exists()
ORDER = ["hr_mean", "eda_slope_rel", "scr_count_rel"]


def run_main(argv) -> tuple:
    """In-process CLI call -> (exit code, stdout)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = P.main(argv)
    return code, buf.getvalue()


def run_fresh(argv) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "src.predict"] + argv, cwd=ROOT, capture_output=True, text=True)


def training_features_w30() -> pd.DataFrame:
    f = pd.read_csv(METRICS / "features_w30.csv")
    e = pd.read_csv(METRICS / "features_eda_w30.csv", index_col="row")
    assert (e.index == f.index).all()
    f["eda_slope_rel"] = e["eda_slope_rel__clean"].values
    f["scr_count_rel"] = e["scr_count_rel__clean"].values
    return f


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="test_predict_"))
        cls.M = P.load_deployment()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def csv(self, name: str, text: str) -> Path:
        p = self.tmp / name
        p.write_text(text, encoding="utf-8")
        return p

    def assertRejected(self, argv, check: str, *fragments):
        code, out = run_main(argv)
        self.assertEqual(code, P.EXIT_REJECTED, out)
        self.assertIn(P.BANNER, out)
        self.assertIn(f"check failed: {check}", out)
        self.assertIn("What to do:", out)
        self.assertIn("No prediction was made.", out)
        for f in fragments:
            self.assertIn(f, out)
        msg = [l for l in out.splitlines() if l.startswith(("INPUT REJECTED", "  Problem", "  What to do"))]
        print("\n    " + "\n    ".join(msg), end=" ... ")
        return out


# ============================================================================ loading / constants
class Loading(Base):
    def test_loads_everything_from_models(self):
        M = self.M
        self.assertEqual(M["order"], ORDER)
        self.assertEqual((M["window_s"], M["step_s"], M["m"], M["n"]), (30.0, 5.0, 4, 5))
        self.assertEqual(M["threshold"], M["coef"]["threshold_op1"])
        self.assertEqual(M["rates"], {"BVP": 64, "EDA": 4})
        self.assertEqual(M["cal"]["duration_s"], 300)
        self.assertEqual(M["cal"]["stored_on_device"], ["sigma_lp_cal", "sigma_ph_cal"])

    def test_limits_derived_from_feature_spec(self):
        lim = self.M["limits"]
        self.assertAlmostEqual(lim["hr_mean"][0], 40.0)
        self.assertAlmostEqual(lim["hr_mean"][1], 60 / 0.33)
        self.assertEqual(lim["scr_count_rel_max"], 30.0)

    def test_train_range_constants_match_feature_csvs(self):
        f = training_features_w30()
        for c, (lo, hi) in P.TRAIN_RANGE.items():
            self.assertAlmostEqual(f[c].min(), lo, places=6, msg=c)
            self.assertAlmostEqual(f[c].max(), hi, places=6, msg=c)


# ============================================================================ stored-output agreement
class StoredOutputs(Base):
    def test_features_csv_reproduces_roundtrip_p_ref_exactly(self):
        """Fresh process, --features-csv on the 300 real rows of models/roundtrip_windows.csv: score == p_ref
        (tolerance 0) and window alert == alert_op1_ref."""
        ref = pd.read_csv(P.MODELS / "roundtrip_windows.csv")
        ref = ref[ref.synthetic_nan == 0].reset_index(drop=True)
        self.assertEqual(len(ref), 300)
        inp = self.tmp / "rt_features.csv"
        ref[ORDER].to_csv(inp, index=False, float_format="%.17g")
        out = self.tmp / "rt_out.csv"
        r = run_fresh(["--features-csv", str(inp), "--out", str(out)])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        got = pd.read_csv(out)
        np.testing.assert_array_equal(got.score.values, ref.p_ref.values)
        np.testing.assert_array_equal((got.alert == "stress").astype(int).values, ref.alert_op1_ref.values)
        self.assertTrue(set(got.alert) <= {"stress", "no stress"})

    @unittest.skipUnless(HAVE_WESAD, "WESAD not available")
    def test_demo_reproduces_training_features_and_scores(self):
        res = P.run_demo("S5", WESAD, self.M)
        t = res["table"]
        f = training_features_w30()
        ref = f[(f.subject == "S5") & f.scored]
        g = ref.merge(t, on="t_start_s", suffixes=("_ref", ""))
        self.assertEqual(len(g), len(ref))                    # every scored S5 window is produced
        for c in ORDER:
            np.testing.assert_allclose(g[c].values, g[c + "_ref"].values, rtol=1e-9, atol=1e-12, err_msg=c)
        rt = pd.read_csv(P.MODELS / "roundtrip_windows.csv")
        rt = rt[(rt.subject == "S5") & (rt.synthetic_nan == 0)].merge(t, on="t_start_s")
        self.assertGreater(len(rt), 0)
        np.testing.assert_allclose(rt.score.values, rt.p_ref.values, rtol=0, atol=1e-12)
        # windows withheld for quality inside the scored set = the training flag_ppg windows
        self.assertEqual(int((g.alert == "WITHHELD").sum()), int(ref.flag_ppg.sum()))

    @unittest.skipUnless(HAVE_WESAD, "WESAD not available")
    def test_signals_mode_on_exported_wesad_equals_training_features(self):
        """S5 raw E4 signals exported to CSV (calibration file = recording start .. end of the 5-min calibration,
        test file = the rest), then --signals in a fresh process: same features as the training CSVs."""
        from src import preprocessing as pp
        raw = pp.load_subject(WESAD / "S5" / "S5.pkl")
        c0, c1 = pp.calibration_period_s(raw["label"])
        d = self.tmp / "s5"
        for name, x, fs, hdr in (("bvp", raw["BVP"], 64, "bvp"), ("eda", raw["EDA"], 4, "eda_uS"),
                                 ("acc", raw["ACC"], 32, "acc_x_g,acc_y_g,acc_z_g")):
            k = int(round(c1 * fs))
            syn.write(d / "cal" / f"{name}.csv", x[:k], hdr, fs)
            syn.write(d / "test" / f"{name}.csv", x[k:], hdr, fs)
        out = self.tmp / "s5_signals_out.csv"
        r = run_fresh(["--signals", str(d / "cal"), str(d / "test"), "--out", str(out)])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        t = pd.read_csv(out)
        f = training_features_w30()
        ref = f[(f.subject == "S5") & f.scored]
        g = ref.merge(t, on="t_start_s", suffixes=("_ref", ""))
        self.assertEqual(len(g), len(ref))
        for c in ORDER:
            np.testing.assert_allclose(g[c].values, g[c + "_ref"].values, rtol=1e-9, atol=1e-12, err_msg=c)


# ============================================================================ representative SYNTHETIC inputs
class SyntheticRepresentative(Base):
    """Three hand-made SYNTHETIC feature rows (not from any person)."""
    ROWS = {"resting-like (SYNTHETIC)": (64.0, 0.0, 0.0),
            "stress-like (SYNTHETIC)": (108.0, 0.05, 6.0),
            "ambiguous (SYNTHETIC)": (95.0, 0.0, 1.0)}

    def _run(self):
        p = self.csv("synthetic3.csv", "hr_mean,eda_slope_rel,scr_count_rel\n" +
                     "".join(f"{a},{b},{c}\n" for a, b, c in self.ROWS.values()))
        code, out = run_main(["--features-csv", str(p), "--out", str(self.tmp / "syn3_out.csv")])
        self.assertEqual(code, 0, out)
        return pd.read_csv(self.tmp / "syn3_out.csv"), out

    def test_three_representative_rows(self):
        t, out = self._run()
        thr = self.M["threshold"]
        names = list(self.ROWS)
        print("\n    " + "\n    ".join(f"{n}: score {s:.4f} vs threshold {thr:.4f} -> {a}"
                                       for n, s, a in zip(names, t.score, t.alert)), end=" ... ")
        self.assertEqual(t.alert[0], "no stress")
        self.assertLess(t.score[0], thr - 0.2)
        self.assertEqual(t.alert[1], "stress")
        self.assertGreater(t.score[1], thr + 0.2)
        self.assertLess(abs(t.score[2] - thr), 0.05)           # ambiguous: within 0.05 of the threshold
        self.assertTrue((t.threshold == thr).all())
        self.assertEqual(list(t.smoothed_alert), ["no stress"] * 3)   # < 4 of 5 positives

    def test_output_has_no_probability_or_percentage(self):
        _, out = self._run()
        self.assertIn(P.BANNER, out)
        self.assertNotIn("%", out)
        self.assertNotIn("probability of stress", out.lower())
        self.assertIn("is not a probability", out)
        self.assertIn("NOT validated:", out)
        for s in ("Finger EDA in mV", "Fingertip PPG", "prototype hardware"):
            self.assertIn(s, out)

    def test_help_says_score_is_not_calibrated(self):
        r = run_fresh(["--help"])
        self.assertEqual(r.returncode, 0)
        self.assertIn("NOT a calibrated probability", r.stdout)

    def test_signals_rest_and_stress_synthetic(self):
        cal, test = syn.write_set(self.tmp / "rest")
        code, out = run_main(["--signals", str(cal), str(test), "--out", str(self.tmp / "rest_out.csv")])
        self.assertEqual(code, 0, out)
        t = pd.read_csv(self.tmp / "rest_out.csv")
        self.assertTrue((t.alert == "no stress").all())
        cal, test = syn.write_set(self.tmp / "stress", test_hr=110, test_scr_every=6)
        code, out = run_main(["--signals", str(cal), str(test), "--out", str(self.tmp / "stress_out.csv")])
        self.assertEqual(code, 0, out)
        t = pd.read_csv(self.tmp / "stress_out.csv")
        self.assertTrue((t.alert == "stress").all())
        self.assertEqual(list(t.smoothed_alert[:3]), ["no stress"] * 3)   # 4-of-5 needs 4 windows
        self.assertTrue((t.smoothed_alert[3:] == "stress").all())

    def test_signals_other_rates_are_resampled_with_warning(self):
        cal, test = syn.write_set(self.tmp / "rates", fs_bvp=100, fs_eda=10)
        code, out = run_main(["--signals", str(cal), str(test)])
        self.assertEqual(code, 0, out)
        self.assertIn("BVP at 100 Hz resampled to the model rate 64 Hz", out)
        self.assertIn("EDA at 10 Hz resampled to the model rate 4 Hz", out)


# ============================================================================ validation: features CSV
class ValidateFeaturesCSV(Base):
    HDR = "hr_mean,eda_slope_rel,scr_count_rel\n"
    OK = "70,0.01,1\n"

    def test_missing_column(self):
        p = self.csv("missing.csv", "hr_mean,scr_count_rel\n70,1\n")
        self.assertRejected(["--features-csv", str(p)], "missing columns", "eda_slope_rel")

    def test_wrong_order(self):
        p = self.csv("order.csv", "eda_slope_rel,hr_mean,scr_count_rel\n0.01,70,1\n")
        self.assertRejected(["--features-csv", str(p)], "wrong column order")

    def test_non_numeric(self):
        p = self.csv("nonnum.csv", self.HDR + self.OK + "seventy,0.01,1\n")
        self.assertRejected(["--features-csv", str(p)], "non-numeric value", "seventy")

    def test_nan(self):
        p = self.csv("nan.csv", self.HDR + self.OK + "NaN,0.01,1\n")
        self.assertRejected(["--features-csv", str(p)], "missing value (NaN/inf)", "file lines 3")

    def test_wrong_shape_ragged_row(self):
        p = self.csv("ragged.csv", self.HDR + self.OK + "70,0.01\n")
        self.assertRejected(["--features-csv", str(p)], "wrong shape", "different number of fields")

    def test_wrong_shape_extra_column(self):
        p = self.csv("extra.csv", "hr_mean,eda_slope_rel,scr_count_rel,rmssd\n70,0.01,1,40\n")
        self.assertRejected(["--features-csv", str(p)], "wrong shape", "rmssd")

    def test_wrong_shape_empty_and_header_only(self):
        self.assertRejected(["--features-csv", str(self.csv("empty.csv", ""))], "wrong shape", "empty")
        self.assertRejected(["--features-csv", str(self.csv("hdr.csv", self.HDR))], "wrong shape", "no data rows")

    def test_hr_out_of_range(self):
        p = self.csv("hr.csv", self.HDR + self.OK * 3 + "25,0.01,1\n")
        self.assertRejected(["--features-csv", str(p)], "hr_mean out of range", "30-220 bpm", "40.0-181.8")

    def test_hr_in_ms_units(self):
        p = self.csv("hrms.csv", self.HDR + "850,0.01,1\n820,0,0\n")
        self.assertRejected(["--features-csv", str(p)], "wrong units (hr_mean)", "ms")

    def test_scr_count_not_a_count(self):
        p = self.csv("scr.csv", self.HDR + "70,0.01,2.5\n")
        self.assertRejected(["--features-csv", str(p)], "scr_count_rel invalid")

    def test_eda_slope_rel_implausible(self):
        p = self.csv("slope.csv", self.HDR + "70,150,1\n")
        self.assertRejected(["--features-csv", str(p)], "eda_slope_rel implausible", "calibration")

    def test_t_start_not_on_grid(self):
        p = self.csv("tstart.csv", "t_start_s," + self.HDR + "0,70,0,0\n3,70,0,0\n")
        self.assertRejected(["--features-csv", str(p)], "t_start_s invalid")

    def test_outside_training_range_is_noted_not_rejected(self):
        p = self.csv("note.csv", self.HDR + "170,0.01,1\n")
        code, out = run_main(["--features-csv", str(p)])
        self.assertEqual(code, 0, out)
        self.assertIn("outside WESAD range: hr_mean", out)


# ============================================================================ validation: raw signals
class ValidateSignals(Base):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp(dir=self.tmp))
        self.cal, self.test = syn.write_set(self.d)

    def sig(self, *extra):
        return ["--signals", str(self.cal), str(self.test)] + list(extra)

    def test_valid_set_passes(self):
        code, out = run_main(self.sig())
        self.assertEqual(code, 0, out)
        self.assertIn("Checks passed.", out)
        self.assertNotIn("WARNING", out)

    def test_flat_eda_no_skin_contact(self):
        x = 2.0 + np.random.default_rng(3).normal(0, 1e-5, 420 * 4)
        syn.write(self.cal / "eda.csv", x, "eda_uS", 4)
        self.assertRejected(self.sig(), "flat EDA during calibration", "no skin contact?")

    def test_eda_too_low_no_skin_contact(self):
        t = np.arange(420 * 4) / 4
        x = 0.02 + 0.004 * np.sin(2 * np.pi * t / 60) + np.random.default_rng(8).normal(0, 0.0005, len(t))
        syn.write(self.cal / "eda.csv", x, "eda_uS", 4)                  # mean 0.02 uS, positive, not flat
        self.assertRejected(self.sig(), "EDA too low during calibration", "no skin contact?")

    def test_flat_eda_window_withheld(self):
        x = 2.0 + np.random.default_rng(4).normal(0, 1e-5, 300 * 4)
        syn.write(self.test / "eda.csv", x, "eda_uS", 4)
        code, out = run_main(self.sig("--out", str(self.d / "o.csv")))
        self.assertEqual(code, 0, out)
        t = pd.read_csv(self.d / "o.csv")
        self.assertTrue((t.alert == "WITHHELD").all())
        self.assertTrue(t.note.str.contains("no skin contact").all())

    def test_railed_bvp(self):
        x = syn.bvp(420, 64)
        syn.write(self.cal / "bvp.csv", np.clip(x, None, 0.6 * x.max()), "bvp", 64)
        self.assertRejected(self.sig(), "saturated / railed signal (BVP)")

    def test_railed_eda_mv(self):
        for p, s in ((self.cal, 420), (self.test, 300)):
            x = syn.eda(s) * 500
            x[: len(x) // 10] = 3200.0                      # ADC at its top
            syn.write(p / "eda.csv", x, "eda_mV", 4)
        self.assertRejected(self.sig(), "saturated / railed signal (EDA)")

    @staticmethod
    def no_pulse(dur, seed):
        """SYNTHETIC 'sensor off': slow 0.25 Hz baseline wander, no cardiac pulse."""
        t = np.arange(dur * 64) / 64
        return 200 * np.sin(2 * np.pi * 0.25 * t) + np.random.default_rng(seed).normal(0, 0.5, len(t))

    def test_low_ppg_quality_calibration(self):
        syn.write(self.cal / "bvp.csv", self.no_pulse(420, 5), "bvp", 64)
        self.assertRejected(self.sig(), "very low PPG quality during calibration")

    def test_low_ppg_quality_window_withheld(self):
        syn.write(self.test / "bvp.csv", self.no_pulse(300, 6), "bvp", 64)
        code, out = run_main(self.sig("--out", str(self.d / "o.csv")))
        self.assertEqual(code, 0, out)
        t = pd.read_csv(self.d / "o.csv")
        self.assertTrue((t.alert == "WITHHELD").all())
        self.assertIn("low PPG quality", out)

    @unittest.expectedFailure
    def test_KNOWN_LIMITATION_band_limited_noise_passes_ppg_quality(self):
        """KNOWN LIMITATION (documented, not fixed): the training PPG quality rule (features.ppg_features /
        quality_flags: >= 5 beats per 10 s, >= 50 % regular IBIs) is scale-invariant and band-passed Gaussian
        noise passes it (median quality 0.69, fake HR ~131 bpm). This test FAILS on purpose until a real signal
        quality index is added; it is marked expectedFailure so the gap stays visible in every test run."""
        syn.write(self.test / "bvp.csv", np.random.default_rng(6).normal(0, 50, 300 * 64), "bvp", 64)
        code, out = run_main(self.sig("--out", str(self.d / "o.csv")))
        t = pd.read_csv(self.d / "o.csv")
        self.assertTrue((t.alert == "WITHHELD").all())

    def test_eda_adc_counts_labelled_uS(self):
        for p, s in ((self.cal, 420), (self.test, 300)):
            syn.write(p / "eda.csv", np.round(syn.eda(s) * 1000), "eda_uS", 4)
        self.assertRejected(self.sig(), "wrong units (EDA)", "ADC counts", "uS")

    def test_eda_adc_counts_labelled_mV(self):
        for p, s in ((self.cal, 420), (self.test, 300)):
            syn.write(p / "eda.csv", np.round(syn.eda(s) * 1900), "eda_mV", 4)
        self.assertRejected(self.sig(), "wrong units (EDA)", "mV", "analogReadMilliVolts")

    def test_eda_header_without_unit(self):
        syn.write(self.cal / "eda.csv", syn.eda(420), "eda", 4)
        self.assertRejected(self.sig(), "wrong units / header (eda)", "eda_uS", "eda_mV")

    def test_acc_raw_lsb(self):
        syn.write(self.cal / "acc.csv", syn.acc(420) * 16384, "acc_x_g,acc_y_g,acc_z_g", 32)
        self.assertRejected(self.sig(), "wrong units (ACC)", "16384")

    def test_sampling_rate_missing(self):
        syn.write(self.cal / "eda.csv", syn.eda(420), "eda_uS", None)
        self.assertRejected(self.sig(), "sampling rate missing")

    def test_sampling_rate_flag_disagrees_with_header(self):
        self.assertRejected(self.sig("--eda-fs", "10"), "sampling rate mismatch")

    def test_calibration_too_short(self):
        cal, test = syn.write_set(self.d / "short", cal_s=200)
        self.assertRejected(["--signals", str(cal), str(test)], "calibration too short", "300")

    def test_too_little_test_data(self):
        cal, test = syn.write_set(self.d / "little", test_s=20)
        self.assertRejected(["--signals", str(cal), str(test)], "too little data")

    def test_gaps_nan(self):
        x = syn.eda(420)
        x[100:110] = np.nan
        syn.write(self.cal / "eda.csv", x, "eda_uS", 4)
        self.assertRejected(self.sig(), "gaps (too little data)", "NaN")

    def test_signal_lengths_disagree(self):
        syn.write(self.cal / "bvp.csv", syn.bvp(380, 64), "bvp", 64)
        self.assertRejected(self.sig(), "signal lengths disagree")

    def test_extreme_motion_during_calibration(self):
        syn.write(self.cal / "acc.csv", syn.acc(420, noise_g=0.3), "acc_x_g,acc_y_g,acc_z_g", 32)
        self.assertRejected(self.sig(), "extreme motion during calibration")

    def test_calibration_stats_far_outside_training(self):
        x = 2.0 + np.random.default_rng(7).normal(0, 1e-5, 420 * 4)
        x[1000] += 0.01                                       # range passes the flat check; sigma is tiny
        syn.write(self.cal / "eda.csv", x, "eda_uS", 4)
        self.assertRejected(self.sig(), "calibration statistics far outside the training range", "sigma_lp_cal")

    def test_warmup_warning(self):
        cal, test = syn.write_set(self.d / "nowarm", cal_s=300)
        code, out = run_main(["--signals", str(cal), str(test)])
        self.assertEqual(code, 0, out)
        self.assertIn("WARNING: Only 0 s of signal before the 5-min calibration", out)
        self.assertIn("first 60 s after power-on", out)

    def test_no_acc_warns_motion_unchecked(self):
        cal, test = syn.write_set(self.d / "noacc", with_acc=False)
        code, out = run_main(["--signals", str(cal), str(test)])
        self.assertEqual(code, 0, out)
        self.assertIn("motion during calibration was NOT checked", out)

    def test_mv_input_flagged_unvalidated(self):
        for p, s in ((self.cal, 420), (self.test, 300)):
            syn.write(p / "eda.csv", syn.eda(s) * 500, "eda_mV", 4)
        code, out = run_main(self.sig())
        self.assertEqual(code, 0, out)
        self.assertIn("EDA in mV", out)
        self.assertIn("UNVALIDATED", out)

    def test_missing_file(self):
        (self.test / "eda.csv").unlink()
        self.assertRejected(self.sig(), "missing file")


# ============================================================================ flat / railed / clipped PPG (Step F G1)
class PPGFlatRailed(Base):
    """The Step F gate G1, added to predict.py as a per-window withhold rule (G2/G3 periodicity checks NOT added).
    Hand-made signals here are SYNTHETIC (tests/synthetic_signals.py, src/ppg_gate.synth), not recordings."""

    def test_rule_equals_ppg_gate_G1_on_synthetic_test_sets(self):
        """Same decision as ppg_gate.gate()["G1_flat_railed"] on every SYNTHETIC Step F test window (1220); all flat,
        railed and clipped windows caught; no clean-pulse control flagged."""
        from src import ppg_gate as G
        rows = []
        for kind, seeds, _ in G.TEST_SETS:
            for s in seeds:
                filt, raw = G.synth(kind, s)
                rows.append({"kind": kind, "mine": P.ppg_flat_railed(raw), **G.window_stats(filt, raw)})
        t = pd.DataFrame(rows)
        ref = G.gate(t, 0.0, np.inf)["G1_flat_railed"]
        np.testing.assert_array_equal(t.mine.values, ref.values)
        caught = t.groupby("kind").mine.agg(["sum", "size"])
        print("\n    " + "; ".join(f"{k} {int(r['sum'])}/{int(r['size'])}" for k, r in caught.iterrows()), end=" ... ")
        for k in ("flat_constant", "railed_full_scale", "clipped_pulse"):
            self.assertEqual(caught.loc[k, "sum"], caught.loc[k, "size"], k)
        for k in ("clean_pulse", "clean_pulse_hrv"):
            self.assertEqual(caught.loc[k, "sum"], 0, k)

    def test_edge_cases(self):
        x = syn.bvp(30, 70)
        self.assertFalse(P.ppg_flat_railed(x))
        self.assertTrue(P.ppg_flat_railed(np.full(1920, 5.0)))                 # flat
        self.assertTrue(P.ppg_flat_railed(np.array([])))                        # no samples
        self.assertTrue(P.ppg_flat_railed(np.clip(x, None, 0.5 * x.max())))   # clipped at half the peak
        y = x.copy(); y[:38] = y.max()                     # 37 or 38 repeats of 1920 (<= 1.98 %) -> passes
        self.assertFalse(P.ppg_flat_railed(y))
        y[:40] = y.max()                                   # 39 or 40 repeats of 1920 (>= 2.03 %) -> flagged
        self.assertTrue(P.ppg_flat_railed(y))

    def _clipped_run(self, name, fs_bvp, damage):
        """Valid SYNTHETIC set; test BVP damaged in [120, 180) s of the test segment (stream time 540-600 s)."""
        cal, test = syn.write_set(self.tmp / name, fs_bvp=fs_bvp)
        x = syn.bvp(300, 66, fs_bvp, 10)
        a, b = 120 * fs_bvp, 180 * fs_bvp
        x[a:b] = damage(x[a:b])
        syn.write(test / "bvp.csv", x, "bvp", fs_bvp)
        code, out = run_main(["--signals", str(cal), str(test), "--out", str(self.tmp / f"{name}.csv")])
        self.assertEqual(code, 0, out)
        t = pd.read_csv(self.tmp / f"{name}.csv")
        inside = (t.t_start_s >= 540) & (t.t_end_s <= 600)
        self.assertEqual(int(inside.sum()), 7)
        return t, inside

    def test_intermittent_clipping_withheld_64hz(self):
        """Clipping in part of the recording only: the segment-level railed check (segment min/max) cannot see it,
        the per-window rule does. Exactly the 7 windows fully inside the clipped minute are flagged."""
        t, inside = self._clipped_run("clip64", 64, lambda s: np.minimum(s, 0.5 * s.max()))
        np.testing.assert_array_equal(self.flagged(t), inside.values)
        self.assertTrue((t.alert[inside] == "WITHHELD").all())
        self.assertTrue((t.alert[~inside] != "WITHHELD").all())
        # the training PPG rule alone accepts these clipped windows (plausible HR ~66 bpm): the new rule is needed
        self.assertFalse(self.notes(t)[inside].str.contains("low PPG quality").any())

    def test_intermittent_clipping_withheld_100hz_native_rate(self):
        """At 100 Hz the BVP is resampled to 64 Hz for the features; the check runs on the native samples."""
        t, inside = self._clipped_run("clip100", 100, lambda s: np.minimum(s, 0.5 * s.max()))
        np.testing.assert_array_equal(self.flagged(t), inside.values)

    def test_flat_section_withheld(self):
        t, inside = self._clipped_run("flat64", 64, lambda s: np.full(len(s), 3.0))
        np.testing.assert_array_equal(self.flagged(t), inside.values)
        self.assertTrue((t.alert[inside] == "WITHHELD").all())

    def test_clean_synthetic_set_not_flagged(self):
        cal, test = syn.write_set(self.tmp / "clean_g1")
        code, out = run_main(["--signals", str(cal), str(test), "--out", str(self.tmp / "clean_g1.csv")])
        self.assertEqual(code, 0, out)
        t = pd.read_csv(self.tmp / "clean_g1.csv")
        self.assertFalse(self.flagged(t).any())

    @staticmethod
    def notes(t: pd.DataFrame) -> pd.Series:
        return t.note.astype(object).where(t.note.notna(), "").astype(str)

    def flagged(self, t: pd.DataFrame) -> np.ndarray:
        return self.notes(t).str.contains("flat/railed/clipped PPG").to_numpy(bool)

    @unittest.skipUnless(HAVE_WESAD, "WESAD not available")
    def test_no_real_wesad_window_flagged(self):
        """Every 30 s window on the 5 s grid of all 15 complete WESAD recordings (all segments, not only the
        labelled ones): none is flagged by the rule predict.py uses."""
        from src import preprocessing as pp
        n, worst = 0, 0.0
        subs = pp.find_subjects(WESAD)
        self.assertEqual(len(subs), 15)
        for name, pkl in subs:
            b = pp.load_subject(pkl)["BVP"]
            for a in range(0, len(b) - 30 * 64 + 1, 5 * 64):
                w = b[a:a + 30 * 64]
                self.assertFalse(P.ppg_flat_railed(w), f"{name} window at {a / 64:.0f} s flagged")
                worst = max(worst, P._railed(w)); n += 1
        print(f"\n    {n} WESAD windows checked, 0 flagged; largest repeated-extreme fraction {worst:.5f} "
              f"(limit {P.PPG_RAIL_FRAC_WINDOW})", end=" ... ")
        self.assertGreater(n, 17000)

    @unittest.skipUnless(HAVE_WESAD, "WESAD not available")
    def test_demo_wesad_no_window_flagged(self):
        res = P.run_demo("S5", WESAD, self.M)
        t = res["table"]
        self.assertGreater(len(t), 1000)
        self.assertFalse(t.note.str.contains("flat/railed/clipped").any())


# ============================================================================ demo / determinism
class Determinism(Base):
    def _twice(self, argv):
        a, b = run_fresh(argv), run_fresh(argv)
        self.assertEqual(a.returncode, 0, a.stdout + a.stderr)
        self.assertEqual(a.stdout, b.stdout)
        return a.stdout

    def test_two_fresh_processes_identical_features_csv(self):
        p = self.csv("det.csv", "hr_mean,eda_slope_rel,scr_count_rel\n64,0,0\n108,0.05,6\n95,0,1\n")
        self._twice(["--features-csv", str(p)])

    def test_two_fresh_processes_identical_signals(self):
        cal, test = syn.write_set(self.tmp / "det_sig", test_hr=110, test_scr_every=6)
        self._twice(["--signals", str(cal), str(test)])

    @unittest.skipUnless(HAVE_WESAD, "WESAD not available")
    def test_two_fresh_processes_identical_demo(self):
        out = self._twice(["--demo-wesad", "S5", "--data", str(WESAD)])
        self.assertIn("S5 is a TRAINING subject", out)
        self.assertIn("NOT an accuracy estimate", out)
        self.assertIn("true_label", out)
        self.assertNotIn("%", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
