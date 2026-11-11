"""
Tests for the saved deployment model (Step D). Needs models/ written by `python -m src.final_model`.

Run from software/ml-models/wesad_edge_model:
    python -m unittest tests.test_final_model -v
"""

import json
import subprocess
import sys
import unittest

import numpy as np
import pandas as pd

from src import feature_spec as fs
from src.predict import MODELS, ROOT, check_feature_order, load, proba_joblib, proba_json


class FinalModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle, cls.coef, cls.spec = load()
        cls.ref = pd.read_csv(MODELS / "roundtrip_windows.csv")
        cls.X = cls.ref[cls.coef["feature_order"]]

    def test_fresh_process_roundtrip(self):
        r = subprocess.run([sys.executable, "-m", "src.predict", "--roundtrip"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ROUNDTRIP PASS", r.stdout)

    def test_feature_order_matches_spec(self):
        order = self.coef["feature_order"]
        self.assertEqual(order, ["hr_mean", "eda_slope_rel", "scr_count_rel"])
        self.assertEqual(order, [f for f in self.spec["order"] if f in set(order)])
        self.assertEqual(self.spec, json.loads(fs.SPEC_JSON.read_text()))     # same as the module-root spec
        with self.assertRaises(ValueError):
            check_feature_order(order[::-1], self.spec)
        with self.assertRaises(ValueError):                                   # excluded (HRV) feature
            check_feature_order(["hr_mean", "rmssd"], self.spec)
        with self.assertRaises(ValueError):                                   # gain-dependent feature
            check_feature_order(["hr_mean", "scl_mean"], self.spec)

    def test_wrong_input_rejected(self):
        o = self.coef["feature_order"]
        for bad in (self.X[o[::-1]], self.X[[o[1], o[0], o[2]]], self.X.to_numpy(), self.X[o[:2]]):
            with self.assertRaises(ValueError):
                proba_joblib(self.bundle, bad)
            with self.assertRaises(ValueError):
                proba_json(self.coef, bad)

    def test_json_equals_pipeline(self):
        p = proba_joblib(self.bundle, self.X)
        np.testing.assert_array_equal(p, self.ref.p_ref.values)
        np.testing.assert_allclose(proba_json(self.coef, self.X), p, rtol=0, atol=1e-12)
        self.assertTrue(self.X.isna().any(axis=1).any(), "fixed window set should include NaN rows (imputer path)")

    def test_json_is_self_consistent(self):
        c = self.coef
        w, mu, sd = (np.asarray(c[k]) for k in ("weights_standardised", "scaler_means", "scaler_sds"))
        np.testing.assert_allclose(c["weights_folded"], w / sd, rtol=1e-12)
        self.assertAlmostEqual(c["bias_folded"], c["bias_standardised"] - float(np.sum(w * mu / sd)), places=12)
        self.assertAlmostEqual(c["logit_threshold_op1"], float(np.log(c["threshold_op1"] / (1 - c["threshold_op1"]))), places=12)
        self.assertEqual((c["smoothing"]["m"], c["smoothing"]["n"]), (4, 5))
        self.assertEqual((c["window_s"], c["step_s"]), (30, 5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
