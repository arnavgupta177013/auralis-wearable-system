"""
Negative tests for the stress detection leakage assertions (Step B).

Each test deliberately creates a violation and checks that the assertion fires.
Uses only the standard library (unittest) + the project code; reads results/metrics/features_w20.csv.

Run from software/ml-models/wesad_edge_model:
    python -m unittest tests.test_leakage -v
"""

import unittest

import numpy as np
import pandas as pd

from src import checks
from src.baselines import TIME_COL, Spec, logreg_spec
from src.features import FEATURES
from src.schemes import Config, personal_z
from src.train import fit_checked, load_windows, run_loso


class LeakageAssertions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_windows(20)
        cls.spec = logreg_spec(Config("universal", True, True, False))

    # ------------------------------------------------------------------ 1. subject overlap
    def test_subject_overlap_fires(self):
        with self.assertRaisesRegex(AssertionError, "both train and test"):
            checks.assert_no_subject_overlap({"S2", "S3", "S4"}, {"S3"})

    def test_fit_on_test_subject_fires(self):
        """Imputer/scaler/model fitted on rows that include the held-out subject."""
        sc = self.df[self.df.scored]
        rows = sc[sc.subject.isin(["S2", "S3"])]
        with self.assertRaisesRegex(AssertionError, "LEAKAGE"):
            fit_checked(self.spec, rows, rows.y.values, {"C": 1.0}, train_subj={"S3"}, test_subj={"S2"})

    # ------------------------------------------------------------------ 2. calibration window in the test set
    def test_calibration_window_in_test_set_fires(self):
        sc = self.df[self.df.scored & (self.df.subject == "S2")]
        cal = self.df[self.df.calib & (self.df.subject == "S2")].head(1)
        with self.assertRaisesRegex(AssertionError, "calibration"):
            checks.assert_no_calibration_rows(pd.concat([sc, cal]), "test set")

    def test_calibration_window_marked_scored_is_caught_by_harness(self):
        """Corrupt the data: mark one S2 calibration window as scored. S2 is the first test fold."""
        bad = self.df.copy()
        i = bad.index[bad.calib & (bad.subject == "S2")][0]
        bad.loc[i, "scored"] = True
        with self.assertRaisesRegex(AssertionError, "calibration window in test set"):
            run_loso(bad, self.spec)

    # ------------------------------------------------------------------ 3. time feature in the feature list
    def test_time_feature_fires(self):
        for f in ["t_start_s", TIME_COL, "label"]:
            with self.assertRaisesRegex(AssertionError, "forbidden"):
                checks.assert_no_time_feature(["hr_mean", f])

    def test_time_feature_in_candidate_model_is_caught_by_harness(self):
        s = self.spec
        bad = Spec("bad", "bad", "logreg", s.features + [TIME_COL], s.fit, s.score, grid=s.grid)
        with self.assertRaisesRegex(AssertionError, "forbidden"):
            run_loso(self.df, bad)

    # ------------------------------------------------------------------ 4. personal-z floor uses training subjects only
    def test_floor_ignores_test_subject(self):
        train = [s for s in self.df.subject.unique() if s != "S2"]
        rows = self.df[self.df.scored]
        _, floor_a, used = personal_z(self.df, rows, FEATURES, train)
        self.assertEqual(used, set(train))
        changed = self.df.copy()                              # inflate S2's calibration data 1000x
        m = changed.calib & (changed.subject == "S2")
        changed.loc[m, FEATURES] = changed.loc[m, FEATURES] * 1000
        _, floor_b, _ = personal_z(changed, rows, FEATURES, train)
        pd.testing.assert_series_equal(floor_a, floor_b)     # floor unchanged -> no test information
        _, floor_c, _ = personal_z(changed, rows, FEATURES, train + ["S2"])
        self.assertFalse(np.allclose(floor_a.values, floor_c.values))   # sanity: the test is sensitive

    # ------------------------------------------------------------------ positive control
    def test_clean_data_passes(self):
        checks.assert_no_subject_overlap({"S3", "S4"}, {"S2"})
        checks.assert_no_calibration_rows(self.df[self.df.scored], "scored windows")
        checks.assert_no_time_feature(self.spec.features)


if __name__ == "__main__":
    unittest.main(verbosity=2)
