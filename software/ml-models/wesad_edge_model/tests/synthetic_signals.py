"""
SYNTHETIC test signals for tests/test_predict.py (hand-made; NOT data from any person or device).

    BVP  sum of two harmonics at the heart rate + Gaussian noise (64 Hz unless stated)
    EDA  tonic level + slow drift + SCR-shaped bumps (rise tau 1 s, decay tau 4 s) + small noise (uS, 4 Hz)
    ACC  (0, 0, 1) g + noise (32 Hz)
"""

from pathlib import Path

import numpy as np


def bvp(duration_s: float, hr_bpm: float, fs: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(duration_s * fs))) / fs
    ph = 2 * np.pi * hr_bpm / 60.0 * t
    return 50 * np.sin(ph) + 20 * np.sin(2 * ph + 0.5) + rng.normal(0, 2.0, len(t))


def eda(duration_s: float, level: float = 2.0, scr_every_s: float = 40.0, scr_amp: float = 0.05,
        fs: int = 4, seed: int = 1, t_offset: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(duration_s * fs))) / fs
    x = level + 0.02 * np.sin(2 * np.pi * (t + t_offset) / 200.0) + rng.normal(0, 0.001, len(t))
    if scr_every_s:
        for t0 in np.arange(scr_every_s / 2, duration_s, scr_every_s):
            d = np.clip(t - t0, 0, None)
            x += scr_amp * (1 - np.exp(-d / 1.0)) * np.exp(-d / 4.0) * (t >= t0)
    return x


def acc(duration_s: float, noise_g: float = 0.01, fs: int = 32, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    return np.c_[np.zeros(n), np.zeros(n), np.ones(n)] + rng.normal(0, noise_g, (n, 3))


def write(path: Path, x: np.ndarray, header: str, fs=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x)
    x = x[:, None] if x.ndim == 1 else x
    with open(path, "w", newline="") as f:
        if fs is not None:
            f.write(f"# fs_hz={fs:g}\n")
        f.write(header + "\n")
        np.savetxt(f, x, delimiter=",", fmt="%.17g")
    return path


def write_set(root: Path, cal_s: float = 420.0, test_s: float = 300.0, test_hr: float = 66.0, cal_hr: float = 64.0,
              test_scr_every: float = 40.0, with_acc: bool = True, fs_bvp: int = 64, fs_eda: int = 4) -> tuple:
    """A VALID synthetic calibration + test pair (rest-like by default). Returns (cal_dir, test_dir)."""
    cal, test = Path(root) / "cal", Path(root) / "test"
    write(cal / "bvp.csv", bvp(cal_s, cal_hr, fs_bvp, 0), "bvp", fs_bvp)
    write(test / "bvp.csv", bvp(test_s, test_hr, fs_bvp, 10), "bvp", fs_bvp)
    write(cal / "eda.csv", eda(cal_s, fs=fs_eda, seed=1), "eda_uS", fs_eda)
    write(test / "eda.csv", eda(test_s, scr_every_s=test_scr_every, fs=fs_eda, seed=11, t_offset=cal_s), "eda_uS",
          fs_eda)
    if with_acc:
        write(cal / "acc.csv", acc(cal_s, seed=2), "acc_x_g,acc_y_g,acc_z_g", 32)
        write(test / "acc.csv", acc(test_s, seed=12), "acc_x_g,acc_y_g,acc_z_g", 32)
    return cal, test
