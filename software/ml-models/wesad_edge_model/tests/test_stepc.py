"""
Tests for the Step C extensions (src/runner.py, src/small_models.py, src/feature_spec.py).

Run from software/ml-models/wesad_edge_model:
    python -m unittest tests.test_stepc -v
"""

import unittest

import numpy as np
import pandas as pd

from src import feature_spec as fs
from src.baselines import logreg_spec
from src.runner import run_loso_ext, smooth, spec_threshold
from src.schemes import Config
from src.small_models import D, d_subsets, model_spec, primary_specs, reference_specs
from src.train import load_windows, run_loso


class StepC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_windows(20)

    def test_same_choice_as_step_b_harness(self):
        """run_loso_ext must pick the same parameters / OP1 threshold and give the same scores as train.run_loso."""
        for cfg in (Config("universal", True, True, False), Config("personal_z", True, True, False)):
            spec = logreg_spec(cfg)
            p_a, f_a = run_loso(self.df, spec)
            p_b, f_b = run_loso_ext(self.df, spec)
            self.assertEqual(list(f_a.params.map(lambda s: s.replace('"', "'"))), list(f_b.params))
            np.testing.assert_allclose(f_a.threshold.values, f_b.thr_op1.values, rtol=0, atol=1e-12)
            np.testing.assert_allclose(p_a.score.values, p_b.score.values, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(p_a.pred.values, p_b.pred_op1.values)

    def test_heldout_predictions_ignore_heldout_labels(self):
        """Flip every label of S2: S2's scores and all four decision rules must be unchanged."""
        spec = model_spec("logreg", "personal_z", D, "primary")
        bad = self.df.copy()
        m = bad.subject == "S2"
        bad.loc[m, "y"] = 1 - bad.loc[m, "y"]
        a, _ = run_loso_ext(self.df, spec)
        b, _ = run_loso_ext(bad, spec)
        a, b = a[a.subject == "S2"], b[b.subject == "S2"]
        np.testing.assert_array_equal(a.score.values, b.score.values)
        for c in ("pred_op1", "pred_op2", "pred_pw", "pred_sm"):
            np.testing.assert_array_equal(a[c].values, b[c].values)

    def test_smoothing_is_causal_m_of_n(self):
        t = np.arange(10) * 5.0
        p = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1])
        out = smooth(p, np.array(["S"] * 10), t, 2, 3)
        self.assertEqual(out.tolist(), [0, 0, 1, 1, 1, 0, 0, 0, 1, 1])
        p2 = p.copy(); p2[9] = 0                          # changing the future never changes the past
        self.assertEqual(smooth(p2, np.array(["S"] * 10), t, 2, 3)[:9].tolist(), out[:9].tolist())
        gap = np.array([0.0, 5.0, 100.0])                  # missing windows count as negative
        self.assertEqual(smooth(np.array([1, 1, 1]), np.array(["S"] * 3), gap, 2, 2).tolist(), [0, 1, 0])

    def test_spec_threshold(self):
        y = np.array([0] * 10 + [1] * 5)
        s = np.r_[np.arange(10) / 10, [0.5, 0.95, 0.99, 1.0, 1.0]]
        t = spec_threshold(y, s, 0.9)
        self.assertGreaterEqual(np.mean(s[y == 0] < t), 0.9)
        lower = np.max(s[s < t])
        self.assertLess(np.mean(s[y == 0] < lower), 0.9)  # the next lower candidate breaks the constraint

    def test_feature_order_enforced(self):
        with self.assertRaisesRegex(AssertionError, "order"):
            model_spec("logreg", "relative_only", ["scr_count", "hr_mean"], "primary")
        for s in primary_specs() + reference_specs():
            fs.assert_spec_order(s.features)
            self.assertFalse(set(s.features) & {"rmssd", "sdnn", "pnn50"})
        for s in primary_specs():
            self.assertFalse(set(s.features) & {"scl_mean", "eda_std", "temp_mean", "scr_amp_mean"})
        self.assertEqual(len(d_subsets()), 8)
        self.assertEqual(len(primary_specs()), 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
