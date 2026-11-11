"""
Leakage assertions for the stress detection validation harness.

Each function raises AssertionError (with a clear message) when a rule is broken.
They are called inside the training loop (src/train.py) on every outer AND inner fold,
and tests/test_leakage.py deliberately breaks each rule to show that the assertion fires.

Rules
-----
1. No subject is in both the training and the test set.
2. Anything fitted (imputer, scaler, personal-z SD floor, model) is fitted on training subjects only.
3. Calibration windows (and windows overlapping the calibration period) are never scored or trained on.
4. No time / position-in-session feature (or the label itself) is ever a model input.
"""

# Columns that encode time, session position or the target. None of them may be a model input.
FORBIDDEN_FEATURES = {
    "t_start_s", "t_end_s", "calib_start_s", "calib_end_s", "time_since_baseline_start_min",
    "label", "y", "subject", "purity", "calib", "in_calib_period", "scored",
}


def assert_no_subject_overlap(train_subjects, test_subjects) -> None:
    overlap = set(train_subjects) & set(test_subjects)
    assert not overlap, f"LEAKAGE: subject(s) {sorted(overlap)} in both train and test"


def assert_fitted_on_train_only(fit_subjects, train_subjects, test_subjects, what: str) -> None:
    """`fit_subjects` = subjects whose rows were actually used to fit `what`."""
    fit, train = set(fit_subjects), set(train_subjects)
    assert fit, f"{what}: fitted on no subjects"
    assert fit <= train, f"LEAKAGE: {what} fitted on non-training subject(s) {sorted(fit - train)}"
    assert not (fit & set(test_subjects)), f"LEAKAGE: {what} fitted on test subject(s) {sorted(fit & set(test_subjects))}"


def assert_no_calibration_rows(df, where: str) -> None:
    """No calibration window and no window overlapping the calibration period may be in `df`."""
    assert not df["calib"].any(), f"LEAKAGE: calibration window in {where}"
    assert not df["in_calib_period"].any(), f"LEAKAGE: window overlapping the calibration period in {where}"
    assert df["scored"].all(), f"LEAKAGE: non-scored window in {where}"


def assert_no_time_feature(features) -> None:
    bad = [f for f in features if f in FORBIDDEN_FEATURES or "time" in f.lower() or f.startswith("t_")]
    assert not bad, f"LEAKAGE: forbidden time/label feature(s) in the feature list: {bad}"
