"""
Feature schemes for the stress detection harness.

Three schemes (exactly):
    universal      raw features
    personal_z     each subject z-scored against their own first 5 min of baseline
                   (SD floor computed from TRAINING subjects only, inside each fold)
    relative_only  no absolute EDA level or absolute temperature

On/off factors:
    motion  acc_mag_mean, motion_energy                 (all schemes)
    drift   temp_mean, scl_mean                         (universal and personal_z only)
    hr_std  hr_std                                      (all schemes; failed like HRV in stress, Step A)
HRV (rmssd, sdnn, pnn50) is removed from every MAIN configuration because Step A found wrist
HRV does not track the chest-ECG reference. A with-HRV variant is kept only as a labelled reference.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features import FEATURES, RELATIVE_FEATURES

MOTION_FEATURES = ["acc_mag_mean", "motion_energy"]
DRIFT_FEATURES = ["temp_mean", "scl_mean"]
HRV_FEATURES = ["rmssd", "sdnn", "pnn50"]
SCHEMES = ("universal", "personal_z", "relative_only")


@dataclass(frozen=True)
class Config:
    scheme: str
    motion: bool = True
    drift: bool = True          # ignored (always False) for relative_only
    hrv: bool = True
    hr_std: bool = True

    @property
    def name(self) -> str:
        parts = [self.scheme, "motion" if self.motion else "nomotion"]
        if self.scheme != "relative_only":
            parts.append("drift" if self.drift else "nodrift")
        parts.append("hrstd" if self.hr_std else "nohrstd")
        parts.append("hrv" if self.hrv else "nohrv")
        return "+".join(parts)

    @property
    def is_reference(self) -> bool:
        """With-HRV configurations are references only (Step A: wrist HRV not trustworthy)."""
        return self.hrv


def feature_list(cfg: Config) -> list:
    """Model input columns for a configuration (never contains time or the label)."""
    base = list(RELATIVE_FEATURES) if cfg.scheme == "relative_only" else list(FEATURES)
    drop = set()
    if not cfg.motion:
        drop |= set(MOTION_FEATURES)
    if not cfg.drift or cfg.scheme == "relative_only":
        drop |= set(DRIFT_FEATURES)
    if not cfg.hrv:
        drop |= set(HRV_FEATURES)
    if not cfg.hr_std:
        drop.add("hr_std")
    return [f for f in base if f not in drop]


def all_configs() -> list:
    """Logistic-regression configurations per window length:
    20 main (no HRV; motion x drift x hr_std, drift only where it applies) + 3 with-HRV references."""
    out = []
    for scheme in SCHEMES:
        drifts = (True, False) if scheme != "relative_only" else (False,)
        for motion in (True, False):
            for drift in drifts:
                for hr_std in (True, False):
                    out.append(Config(scheme, motion, drift, hrv=False, hr_std=hr_std))
    for scheme in SCHEMES:                                  # references: full feature set incl. HRV
        out.append(Config(scheme, True, scheme != "relative_only", hrv=True, hr_std=True))
    return out


def calibration_stats(df: pd.DataFrame, features: list) -> tuple:
    """Per-subject mean and SD of each feature over that subject's calibration windows."""
    cal = df[df["calib"]]
    return cal.groupby("subject")[features].mean(), cal.groupby("subject")[features].std()


def personal_z(df_all: pd.DataFrame, rows: pd.DataFrame, features: list, floor_subjects) -> tuple:
    """z-score `rows` with each subject's own calibration mean/SD.

    df_all         : all windows (needed for the calibration windows of every subject)
    rows           : the windows to transform (scored windows of train and/or test subjects)
    floor_subjects : subjects whose calibration SDs define the SD floor -> MUST be training subjects only

    Returns (transformed copy of rows, floor Series, set of subjects used for the floor).
    """
    mu, sd = calibration_stats(df_all, features)
    floor_subjects = sorted(set(floor_subjects))
    floor = sd.loc[floor_subjects].median()                  # median calib SD of the training subjects
    out = rows.copy()
    for s, idx in rows.groupby("subject").groups.items():
        assert s in mu.index, f"{s}: no calibration windows"
        s_sd = np.maximum(sd.loc[s].fillna(0), floor.fillna(0)).replace(0, 1.0)
        out.loc[idx, features] = ((rows.loc[idx, features] - mu.loc[s]) / s_sd).values
    return out, floor, set(floor_subjects)
